import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.keras")

app = FastAPI(title="Digit Recognition API")
model = keras.models.load_model(MODEL_PATH)
INPUT_RANK = len(model.input_shape)  # 2 = flat, 3 = (28,28), 4 = (28,28,1)


# ---------- shared ML helpers ----------
def preprocess(img: Image.Image):
    """White-on-black image -> MNIST-style 28x28 array (0..1).

    Crop to the digit, scale to fit 20x20, then center by center of mass in 28x28,
    exactly like the MNIST dataset was prepared.
    """
    arr = np.array(img.convert("L"), dtype=np.float32)
    mask = arr > 30
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray(arr.astype("uint8")).resize((nw, nh), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.array(small, dtype=np.float32)

    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    canvas = np.roll(canvas, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return canvas / 255.0


def to_model_input(xs: np.ndarray) -> np.ndarray:
    n = len(xs)
    if INPUT_RANK == 2:
        return xs.reshape(n, -1)
    if INPUT_RANK == 3:
        return xs.reshape(n, 28, 28)
    return xs.reshape(n, 28, 28, 1)


def predict_probs(xs: np.ndarray) -> np.ndarray:
    raw = model.predict(to_model_input(xs), verbose=0).astype(np.float64)
    bad = (raw.min(axis=1) < 0) | ~np.isclose(raw.sum(axis=1), 1.0, atol=1e-3)
    if bad.any():  # model returned logits -> softmax
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        raw = np.where(bad[:, None], e / e.sum(axis=1, keepdims=True), raw)
    return raw


# ---------- multi-digit segmentation ----------
def segment_digits(img_bgr: np.ndarray):
    """Find each handwritten digit in a photo/scan of paper.

    Handles uneven lighting, removes ruled notebook lines, and ignores
    page edges, logos and specks. Returns (rows of boxes in reading order, mask).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    H, W = gray.shape

    # 1. Threshold per-region so shadows / uneven light don't matter
    dark_ink = np.median(gray) >= 100  # light paper with dark ink (else: light ink on dark)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if dark_ink else cv2.THRESH_BINARY,
        51, 12 if dark_ink else -12,
    )

    # 2. Remove ruled lines: very long thin horizontal / vertical strokes
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(41, W // 12), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(61, H // 6)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk),
    )
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    mask = cv2.subtract(mask, lines)

    # 3. Repair digit strokes that were cut where they crossed a line
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))

    # 4. Candidate boxes, then filter out everything that isn't digit-shaped
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] >= 0.0002 * H * W and b[3] >= 0.02 * H]  # specks
    boxes = [b for b in boxes if b[2] <= 2.0 * b[3] and b[3] <= 0.4 * H]  # long bars / huge blobs
    boxes = [b for b in boxes
             if b[0] > 2 and b[1] > 2 and b[0] + b[2] < W - 2 and b[1] + b[3] < H - 2]  # page edges
    if not boxes:
        return [], mask

    # Typical digit height = the most common height; drop boxes far from it
    heights = np.array([b[3] for b in boxes], dtype=float)
    best = max(heights, key=lambda h0: np.sum((heights >= 0.7 * h0) & (heights <= 1.4 * h0)))
    boxes = [b for b in boxes if 0.6 * best <= b[3] <= 1.6 * best]
    if not boxes:
        return [], mask

    # 5. Group into rows (top to bottom), then left to right
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    rows = []
    for b in boxes:
        cy = b[1] + b[3] / 2
        if rows and abs(cy - np.mean([r[1] + r[3] / 2 for r in rows[-1]])) < 0.6 * best:
            rows[-1].append(b)
        else:
            rows.append([b])

    # A lone mark far away from a real line of digits is almost always a stray (logo, smudge)
    if max(len(r) for r in rows) >= 4:
        rows = [r for r in rows if len(r) >= 2]

    rows = [sorted(r, key=lambda b: b[0]) for r in rows]
    return rows, mask


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single digit drawn on the canvas."""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    x = preprocess(img)
    if x is None:
        raise HTTPException(status_code=422, detail="Canvas is empty")

    probs = predict_probs(x[None])[0]
    digit = int(np.argmax(probs))
    return {
        "digit": digit,
        "confidence": float(probs[digit]),
        "probabilities": {str(i): float(p) for i, p in enumerate(probs)},
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Photo/scan containing several digits."""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    rows, mask = segment_digits(img)
    if not rows:
        raise HTTPException(status_code=422, detail="No digits found in the image")

    flat, crops = [], []
    for ri, row in enumerate(rows):
        for (x, y, bw, bh) in row:
            pad = int(0.15 * max(bw, bh))
            crop = np.pad(mask[y:y + bh, x:x + bw], pad)
            arr = preprocess(Image.fromarray(crop))
            if arr is not None:
                crops.append(arr)
                flat.append((ri, x, y, bw, bh))

    if not crops:
        raise HTTPException(status_code=422, detail="No digits found in the image")

    probs = predict_probs(np.stack(crops))
    digits = []
    for (ri, x, y, bw, bh), p in zip(flat, probs):
        d = int(np.argmax(p))
        digits.append({
            "digit": d,
            "confidence": float(p[d]),
            "row": ri,
            "box": [x / W, y / H, bw / W, bh / H],  # fractions of image size
        })

    lines = []
    for ri in range(len(rows)):
        lines.append("".join(str(d["digit"]) for d in digits if d["row"] == ri))

    return {"count": len(digits), "lines": lines, "digits": digits}
