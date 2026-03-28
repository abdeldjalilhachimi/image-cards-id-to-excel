import asyncio
import io
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openpyxl
import pytesseract
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Algerian ID Card Extractor")
templates = Jinja2Templates(directory="templates")

# Temporary in-memory store: token → xlsx bytes
_download_store: dict[str, bytes] = {}

# Tesseract config: Arabic + French + English (digits/Latin), LSTM engine
_TESS_CONFIG = "--oem 3 --psm 3"
_TESS_LANG   = "ara+fra+eng"

# Tesseract is thread-safe and light on RAM — 4 parallel workers is safe
_executor = ThreadPoolExecutor(max_workers=4)

# ── Label sets ────────────────────────────────────────────────────────────────
# French and Arabic variants found on Algerian national ID cards
LASTNAME_LABELS = {
    "NOM",
    "اللقب",       # family name
    "النسب",
    "اسم العائلة",
}
FIRSTNAME_LABELS = {
    "PRÉNOM",
    "PRENOM",
    "الإسم",        # given name (Algerian ID spelling with hamza)
    "الاسم",        # alternate spelling without hamza
    "الإسم الشخصي",
    "الاسم الشخصي",
}
# All OTHER field labels on Algerian ID cards — must never be mistaken for names
NON_NAME_LABELS = {
    "الجنس",                 # gender
    "تاريخ الميلاد",         # date of birth
    "مكان الميلاد",          # place of birth
    "سلطة الإصدار",          # issuing authority
    "تاريخ الإصدار",         # issue date
    "تاريخ الانتهاء",        # expiry date
    "تاريخ الإنتهاء",
    "رقم التعريف الوطني",    # NIN label
    "رقم بطاقة التعريف",     # card number label
    "الجنسية",               # nationality
    "ذكر",                   # male
    "أنثى",                  # female
    "الجمهورية الجزائرية",   # header text
    "بطاقة التعريف الوطنية", # header text
    "Rh",                    # blood type label
    "RH",
}

# NIN on Algerian ID cards = 18 consecutive digits
NIN_RE = re.compile(r"\d{18}")
# Sometimes OCR inserts spaces inside the number
NIN_SPACED_RE = re.compile(r"\d[\d ]{16,20}\d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_arabic(text: str) -> str:
    """Strip Arabic diacritics and normalize visually-similar characters."""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    return text


def _norm(text: str) -> str:
    """Strip punctuation, normalize Arabic, collapse spaces, uppercase."""
    text = _normalize_arabic(text)
    return re.sub(r"[:\-–./]", "", text).strip().upper()


def _is_label(text: str) -> bool:
    n = _norm(text)
    all_labels = {_norm(l) for l in LASTNAME_LABELS | FIRSTNAME_LABELS | NON_NAME_LABELS}
    return n in all_labels


def _is_valid_name(text: str) -> bool:
    """Return True if the text could be a real name value (not a label, not digits)."""
    t = text.strip()
    if not t:
        return False
    if _is_label(t):
        return False
    if re.match(r'^[\d\s\-/.:,]+$', t):
        return False
    if ":" in t:
        return False
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 3


def _is_latin(text: str) -> bool:
    """Return True if text is mostly Latin script (French name preferred)."""
    latin = sum(1 for c in text if "\u0000" <= c <= "\u024F")
    return latin > len(text) * 0.5


def _clean_name(text: str) -> str:
    """
    Remove OCR noise tokens from an extracted name.
    Handles cases like 'سميرة 7 HS' → 'سميرة':
      - Pure-digit tokens  ('7', '19')
      - Short ASCII tokens (≤2 chars) mixed into Arabic text ('HS', 'ID', 'Rh')
      - Tokens that are mostly non-alphabetic (punctuation fragments)
    """
    if not text:
        return text
    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
    clean = []
    for tok in text.split():
        if re.fullmatch(r"\d+", tok):
            continue                                     # pure digit
        if has_arabic and len(tok) <= 2 and tok.isascii():
            continue                                     # short Latin noise in Arabic name
        alpha = [c for c in tok if c.isalpha()]
        if len(alpha) < max(1, len(tok) * 0.5):
            continue                                     # mostly punctuation / digits
        clean.append(tok)
    return " ".join(clean).strip()


# ── Core extraction ───────────────────────────────────────────────────────────

def _collect_same_row(i: int, texts: list, bboxes: list, rtl: bool = True) -> str:
    """
    Collect valid-name blocks on the same row as block i (the label).

    rtl=True  (Arabic labels): value is to the LEFT  of the label.
    rtl=False (French labels): value is to the RIGHT of the label.

    Proximity capped at max(300px, 4×label_width) to ignore far watermarks.
    """
    label_bbox   = bboxes[i]
    label_top    = label_bbox[0][1]
    label_bottom = label_bbox[2][1]
    label_center = (label_top + label_bottom) / 2
    label_h      = max(label_bottom - label_top, 1)
    label_left_x = label_bbox[0][0]
    label_right_x= label_bbox[1][0]
    label_w      = max(label_right_x - label_left_x, 1)
    max_x_gap    = max(300, label_w * 4)

    matches = []
    for j, (t, b) in enumerate(zip(texts, bboxes)):
        if j == i:
            continue
        cand_top     = b[0][1]
        cand_bottom  = b[2][1]
        cand_center  = (cand_top + cand_bottom) / 2
        cand_left_x  = b[0][0]
        cand_right_x = b[1][0]

        row_thresh = 1.8 * max(label_h, cand_bottom - cand_top, 1)
        if abs(cand_center - label_center) > row_thresh:
            continue

        if rtl:
            # Arabic RTL: value sits to the LEFT of the label
            if cand_right_x > label_left_x:
                continue
            if label_left_x - cand_right_x > max_x_gap:
                continue
        else:
            # French LTR: value sits to the RIGHT of the label
            if cand_left_x < label_right_x:
                continue
            if cand_left_x - label_right_x > max_x_gap:
                continue

        if _is_valid_name(t):
            matches.append((b[0][0], t))

    if not matches:
        return ""
    # RTL: read highest-X first; LTR: read lowest-X first
    matches.sort(key=lambda m: m[0], reverse=rtl)
    return " ".join(t for _, t in matches)


def _extract_inline_value(text: str, label: str) -> str:
    """
    On Algerian ID cards labels and values appear on the same line:
      اللقب: حداق   or   الإسم: مراد
    Split on ':' and return the part that is NOT the label.
    """
    if ":" not in text:
        return ""
    parts = [p.strip() for p in text.split(":")]
    label_norm = _normalize_arabic(label)
    for idx, part in enumerate(parts):
        if label_norm in _normalize_arabic(part):
            if idx + 1 < len(parts) and parts[idx + 1]:
                return parts[idx + 1].strip()
            if idx - 1 >= 0 and parts[idx - 1]:
                return parts[idx - 1]
    return ""


def extract_fields(ocr_results: list) -> dict:
    sorted_res = sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    texts  = [r[1].strip() for r in sorted_res]
    bboxes = [r[0]         for r in sorted_res]

    nin = lastname = firstname = ""

    # ── NIN ──────────────────────────────────────────────────────────────────
    full_digits_only = re.sub(r"\s", "", " ".join(texts))
    m = NIN_RE.search(full_digits_only)
    if m:
        nin = m.group()
    else:
        m2 = NIN_SPACED_RE.search(" ".join(texts))
        if m2:
            nin = re.sub(r"\s", "", m2.group())

    # ── Names ─────────────────────────────────────────────────────────────────
    lastname_labels_norm  = {_norm(l) for l in LASTNAME_LABELS}
    firstname_labels_norm = {_norm(l) for l in FIRSTNAME_LABELS}

    for i, text in enumerate(texts):
        norm    = _norm(text)
        label_y = bboxes[i][0][1]

        text_norm_ar = _normalize_arabic(text)

        # ── Strategy 1: inline "label: value" in the same OCR block ──────────
        if not lastname:
            for label in LASTNAME_LABELS:
                if _normalize_arabic(label) in text_norm_ar:
                    val = _clean_name(_extract_inline_value(text, label))
                    if val and _is_valid_name(val):
                        lastname = val
                        break

        if not firstname:
            for label in FIRSTNAME_LABELS:
                if _normalize_arabic(label) in text_norm_ar:
                    val = _clean_name(_extract_inline_value(text, label))
                    if val and _is_valid_name(val):
                        firstname = val
                        break

        # ── Strategy 2: standalone label → same-row search (RTL then LTR) ────
        is_lastname_label  = norm in lastname_labels_norm
        is_firstname_label = norm in firstname_labels_norm

        if is_lastname_label and not lastname:
            # Try RTL (Arabic) then LTR (French)
            val = _collect_same_row(i, texts, bboxes, rtl=True) or \
                  _collect_same_row(i, texts, bboxes, rtl=False)
            if val:
                lastname = _clean_name(val)
            else:
                # Fallback: scan blocks immediately below the label
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    candidate = _clean_name(candidate)
                    if _is_valid_name(candidate):
                        lastname = candidate
                        break

        elif is_firstname_label and not firstname:
            val = _collect_same_row(i, texts, bboxes, rtl=True) or \
                  _collect_same_row(i, texts, bboxes, rtl=False)
            if val:
                firstname = _clean_name(val)
            else:
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    candidate = _clean_name(candidate)
                    if _is_valid_name(candidate):
                        firstname = candidate
                        break

    return {"nin": nin, "lastname": lastname, "firstname": firstname}


# ── OCR engine (Tesseract) ────────────────────────────────────────────────────

def _run_ocr(img: Image.Image) -> list:
    """
    Run Tesseract on a PIL image and return results in the same format
    extract_fields() expects:  [ [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], text, conf ], …
    Words are grouped back into lines so inline "label: value" patterns work.
    """
    data = pytesseract.image_to_data(
        img,
        lang=_TESS_LANG,
        config=_TESS_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    # Group words into lines keyed by (block, paragraph, line)
    lines: dict = {}
    for i in range(len(data["text"])):
        if data["level"][i] != 5:          # level 5 = word
            continue
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if conf < 30 or not word:          # skip low-confidence garbage
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = {"words": [], "xs": [], "ys": [], "x2s": [], "y2s": []}
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        lines[key]["words"].append(word)
        lines[key]["xs"].append(x)
        lines[key]["ys"].append(y)
        lines[key]["x2s"].append(x + w)
        lines[key]["y2s"].append(y + h)

    results = []
    for line in lines.values():
        text = " ".join(line["words"])
        x1, y1 = min(line["xs"]), min(line["ys"])
        x2, y2 = max(line["x2s"]), max(line["y2s"])
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        results.append([bbox, text, 0.9])   # fixed conf; Tesseract line-level conf is unreliable

    return results


# ── Image processing ──────────────────────────────────────────────────────────

def _preprocess(img: Image.Image) -> Image.Image:
    """
    Prepare image for Tesseract:
    - Grayscale removes colour noise
    - Keep between 1200–1800 px wide (300 DPI equivalent for a standard ID card)
    - Contrast boost sharpens faint print
    - BILINEAR is 3× faster than LANCZOS with no visible quality loss here
    """
    gray = img.convert("L")
    w, h = gray.size
    if w < 1200:
        scale = 1200 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    elif w > 1800:
        scale = 1800 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    return gray


def process_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _preprocess(img)
    results = _run_ocr(img)
    return extract_fields(results)


# ── XLSX builder ──────────────────────────────────────────────────────────────

def build_xlsx(rows: list[dict]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ID Cards"

    headers = ["Filename", "NIN", "Last Name", "First Name"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)

    for row in rows:
        ws.append([
            row.get("filename", ""),
            row.get("nin", ""),
            row.get("lastname", ""),
            row.get("firstname", ""),
        ])

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max(max_len + 4, 12)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    """Return raw OCR lines so you can inspect what Tesseract detects."""
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _preprocess(img)
    results = _run_ocr(img)
    blocks = [
        {"text": r[1], "confidence": round(r[2], 3), "bbox": r[0]}
        for r in sorted(results, key=lambda r: r[0][0][1])
    ]
    return JSONResponse(blocks)


@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)):
    # Read all uploads first (async I/O, non-blocking)
    uploads = [(f.filename, await f.read()) for f in files]

    # Run OCR in parallel — Tesseract is thread-safe and low-memory
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(_executor, process_image, data) for _, data in uploads]
    ocr_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for (filename, _), data in zip(uploads, ocr_results):
        if isinstance(data, Exception):
            results.append({"filename": filename, "nin": "", "lastname": "", "firstname": "", "error": str(data)})
        else:
            results.append({"filename": filename, **data, "error": None})

    xlsx_bytes = build_xlsx(results)
    token = str(uuid.uuid4())
    _download_store[token] = xlsx_bytes
    return JSONResponse({"token": token})


@app.get("/download/{token}")
async def download(token: str):
    xlsx_bytes = _download_store.pop(token, None)
    if not xlsx_bytes:
        raise HTTPException(status_code=404, detail="File not found or already downloaded")
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=nin_data.xlsx"},
    )
