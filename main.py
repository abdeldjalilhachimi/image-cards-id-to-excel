import io
import re

import easyocr
import numpy as np
import openpyxl
from PIL import Image
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Algerian ID Card Extractor")
templates = Jinja2Templates(directory="templates")

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
    "الإسم",        # given name (Algerian ID spelling)
    "الاسم",        # alternate spelling without hamza
    "الإسم الشخصي",
    "الاسم الشخصي",
}

# NIN on Algerian ID cards = 18 consecutive digits
NIN_RE = re.compile(r"\d{18}")
# Sometimes OCR inserts spaces inside the number
NIN_SPACED_RE = re.compile(r"\d[\d ]{16,20}\d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Strip punctuation, collapse spaces, uppercase — for label comparison."""
    return re.sub(r"[:\-–./]", "", text).strip().upper()


def _is_label(text: str) -> bool:
    n = _norm(text)
    all_labels = {_norm(l) for l in LASTNAME_LABELS | FIRSTNAME_LABELS}
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
    # Must have at least 2 meaningful characters
    letters = [c for c in t if c.isalpha()]
    return len(letters) >= 2


def _is_latin(text: str) -> bool:
    """Return True if text is mostly Latin script (French name preferred)."""
    latin = sum(1 for c in text if "\u0000" <= c <= "\u024F")
    return latin > len(text) * 0.5


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_inline_value(text: str, label: str) -> str:
    """
    On Algerian ID cards labels and values appear on the same line:
      اللقب: حداق   or   الإسم: مراد
    Split on ':' and return the part that is NOT the label.
    """
    if ":" not in text:
        return ""
    parts = [p.strip() for p in text.split(":")]
    for idx, part in enumerate(parts):
        if label in part:
            # value is the adjacent part (before or after)
            if idx + 1 < len(parts) and parts[idx + 1]:
                return parts[idx + 1]
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
        if not lastname:
            for label in LASTNAME_LABELS:
                if label in text:
                    val = _extract_inline_value(text, label)
                    if val and _is_valid_name(val):
                        lastname = val
                        break

        if not firstname:
            for label in FIRSTNAME_LABELS:
                if label in text:
                    val = _extract_inline_value(text, label)
                    if val and _is_valid_name(val):
                        firstname = val
                        break

        # ── Strategy 2: standalone label → search nearby blocks ──────────────
        if norm in lastname_labels_norm and not lastname:
            # Same row first (Arabic RTL: value block is to the left)
            for j, (t, b) in enumerate(zip(texts, bboxes)):
                if j == i:
                    continue
                if abs(b[0][1] - label_y) <= 40 and _is_valid_name(t):
                    lastname = t
                    break
            # Fallback: next blocks below
            if not lastname:
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    if _is_valid_name(candidate):
                        lastname = candidate
                        break

        elif norm in firstname_labels_norm and not firstname:
            for j, (t, b) in enumerate(zip(texts, bboxes)):
                if j == i:
                    continue
                if abs(b[0][1] - label_y) <= 40 and _is_valid_name(t):
                    firstname = t
                    break
            if not firstname:
                for j in range(i + 1, min(i + 8, len(texts))):
                    candidate = texts[j].strip()
                    if abs(bboxes[j][0][1] - label_y) > 200:
                        break
                    if _is_valid_name(candidate):
                        firstname = candidate
                        break

    return {"nin": nin, "lastname": lastname, "firstname": firstname}


# ── Image processing ──────────────────────────────────────────────────────────

def process_image(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
            results.append(
                {
                    "filename": upload.filename,
                    **data,
                    "error": None,
                }
            )
        except Exception as e:
            results.append(
                {
                    "filename": upload.filename,
                    "nin": "",
                    "lastname": "",
                    "firstname": "",
                    "error": str(e),
                }
            )

    xlsx_bytes = build_xlsx(results)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=nin_data.xlsx"},
    )
