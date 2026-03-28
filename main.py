import io
import re
import uuid

import easyocr
import numpy as np
import openpyxl
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Algerian ID Card Extractor")
templates = Jinja2Templates(directory="templates")

# Temporary in-memory store: token → xlsx bytes
_download_store: dict[str, bytes] = {}

# Loaded once at startup — first run downloads ~100 MB of models
print("Loading OCR models (Arabic + Latin)…")
reader = easyocr.Reader(["ar", "en"], gpu=False)
print("OCR models ready.")

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
    # Remove harakat (diacritics: fatha, damma, kasra, etc.)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Normalize alef variants → bare alef (OCR often confuses أ / إ / آ / ا)
    text = re.sub(r"[أإآٱ]", "ا", text)
    # Normalize alef maqsura → ya
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
    # Reject pure digit / punctuation strings (NIN fragments, dates, codes)
    if re.match(r'^[\d\s\-/.:,]+$', t):
        return False
    # Reject anything containing a colon — names never have colons, label fragments always do
    if ":" in t:
        return False
    # Must have at least 3 meaningful characters (filters "Rh", "O+", single chars…)
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 3


def _is_latin(text: str) -> bool:
    """Return True if text is mostly Latin script (French name preferred)."""
    latin = sum(1 for c in text if "\u0000" <= c <= "\u024F")
    return latin > len(text) * 0.5


# ── Core extraction ───────────────────────────────────────────────────────────

def _collect_same_row(i: int, texts: list, bboxes: list) -> str:
    """
    Collect valid-name blocks on the same row as block i (the label).
    Rules:
      - Same row  : center-Y within 1.2 × max(label height, candidate height)
      - Direction : Arabic is RTL → value block must be to the LEFT of the label
      - Proximity : candidate's right edge must be within max(300px, 4×label_width)
                    of the label's left edge (ignores far-away watermarks / city names)
    Results joined in RTL reading order (highest X first = first Arabic word).
    """
    label_bbox    = bboxes[i]
    label_top     = label_bbox[0][1]
    label_bottom  = label_bbox[2][1]
    label_center  = (label_top + label_bottom) / 2
    label_h       = max(label_bottom - label_top, 1)
    label_left_x  = label_bbox[0][0]
    label_w       = max(label_bbox[1][0] - label_bbox[0][0], 1)
    max_x_gap     = max(300, label_w * 4)   # generous but bounded proximity limit

    matches = []
    for j, (t, b) in enumerate(zip(texts, bboxes)):
        if j == i:
            continue
        cand_top     = b[0][1]
        cand_bottom  = b[2][1]
        cand_center  = (cand_top + cand_bottom) / 2
        cand_right_x = b[1][0]

        # ── Row check ────────────────────────────────────────────────────────
        row_thresh = 1.8 * max(label_h, cand_bottom - cand_top, 1)
        if abs(cand_center - label_center) > row_thresh:
            continue

        # ── Direction check (RTL: value is to the LEFT of the label) ─────────
        if cand_right_x > label_left_x:
            continue   # block starts to the right of / overlaps the label

        # ── Proximity check (not a far-away unrelated block) ─────────────────
        x_gap = label_left_x - cand_right_x
        if x_gap > max_x_gap:
            continue

        if _is_valid_name(t):
            matches.append((b[0][0], t))   # (left-x, text)

    if not matches:
        return ""
    # Sort by X descending = RTL reading order (rightmost word first)
    matches.sort(key=lambda m: m[0], reverse=True)
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
        # Normalize both sides so إ/ا/ى differences don't cause misses
        if label_norm in _normalize_arabic(part):
            if idx + 1 < len(parts) and parts[idx + 1]:
                return parts[idx + 1].strip()
            if idx - 1 >= 0 and parts[idx - 1]:
                return parts[idx - 1]
    return ""


def extract_fields(ocr_results: list) -> dict:
    # Sort top → bottom, then left → right
    sorted_res = sorted(ocr_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    texts = [r[1].strip() for r in sorted_res]
    bboxes = [r[0] for r in sorted_res]

    nin = ""
    lastname = ""
    firstname = ""

    # ── NIN ──────────────────────────────────────────────────────────────────
    full_digits_only = re.sub(r"\s", "", " ".join(texts))
    m = NIN_RE.search(full_digits_only)
    if m:
        nin = m.group()
    else:
        full_spaced = " ".join(texts)
        m2 = NIN_SPACED_RE.search(full_spaced)
        if m2:
            nin = re.sub(r"\s", "", m2.group())

    # ── Names ─────────────────────────────────────────────────────────────────
    lastname_labels_norm = {_norm(l) for l in LASTNAME_LABELS}
    firstname_labels_norm = {_norm(l) for l in FIRSTNAME_LABELS}

    for i, text in enumerate(texts):
        norm = _norm(text)
        label_y = bboxes[i][0][1]

        # ── Strategy 1: inline "label: value" in the same OCR block ──────────
        text_norm_ar = _normalize_arabic(text)
        if not lastname:
            for label in LASTNAME_LABELS:
                if _normalize_arabic(label) in text_norm_ar:
                    val = _extract_inline_value(text, label)
                    if val and _is_valid_name(val):
                        lastname = val
                        break

        if not firstname:
            for label in FIRSTNAME_LABELS:
                if _normalize_arabic(label) in text_norm_ar:
                    val = _extract_inline_value(text, label)
                    if val and _is_valid_name(val):
                        firstname = val
                        break

        # ── Strategy 2: standalone label → search nearby blocks ──────────────
        if norm in lastname_labels_norm and not lastname:
            val = _collect_same_row(i, texts, bboxes)
            if val:
                lastname = val
            else:
                # Fallback: next blocks below
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    if _is_valid_name(candidate):
                        lastname = candidate
                        break

        elif norm in firstname_labels_norm and not firstname:
            val = _collect_same_row(i, texts, bboxes)
            if val:
                firstname = val
            else:
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    if _is_valid_name(candidate):
                        firstname = candidate
                        break

    return {"nin": nin, "lastname": lastname, "firstname": firstname}


# ── Image processing ──────────────────────────────────────────────────────────

def _preprocess(img: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy:
    - Convert to grayscale (removes colour noise)
    - Upscale to at least 1400 px wide (helps with PDF-converted or small images)
    - Mild contrast boost
    """
    gray = img.convert("L")
    w, h = gray.size
    if w < 1400:
        scale = 1400 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    return gray.convert("RGB")


def process_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = _preprocess(img)
    img_np = np.array(img)
    results = reader.readtext(img_np)
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
        ws.append(
            [
                row.get("filename", ""),
                row.get("nin", ""),
                row.get("lastname", ""),
                row.get("firstname", ""),
            ]
        )

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
    """Return raw OCR blocks so you can see exactly what EasyOCR detects."""
    from fastapi.responses import JSONResponse
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    results = reader.readtext(img_np)
    blocks = [
        {"text": r[1], "confidence": round(r[2], 3), "bbox": r[0]}
        for r in sorted(results, key=lambda r: r[0][0][1])
    ]
    return JSONResponse(blocks)


@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)):
    results = []
    for upload in files:
        image_bytes = await upload.read()
        try:
            data = process_image(image_bytes)
            results.append({"filename": upload.filename, **data, "error": None})
        except Exception as e:
            results.append({"filename": upload.filename, "nin": "", "lastname": "", "firstname": "", "error": str(e)})

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
