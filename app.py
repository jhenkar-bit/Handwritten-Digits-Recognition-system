import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_drawable_canvas import st_canvas

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")
st.title("✍️ Digit Recognizer")

try:
    requests.get(f"{API_URL}/health", timeout=2).raise_for_status()
except Exception:
    st.error("Backend is not running. Start it with: uvicorn backend:app --port 8000")
    st.stop()

tab_draw, tab_upload = st.tabs(["✏️ Draw a digit", "🖼️ Upload image with digits"])

# ---------------- Draw tab ----------------
with tab_draw:
    st.caption("Draw a single digit (0-9) big and thick, then click Predict.")
    left, right = st.columns(2)

    with left:
        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.caption("Use the trash icon under the canvas to clear.")
        predict = st.button("Predict", type="primary")

    with right:
        if predict:
            if canvas.image_data is None:
                st.warning("Draw a digit first.")
            else:
                img = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                try:
                    r = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("digit.png", buf, "image/png")},
                        timeout=15,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.metric("Prediction", data["digit"])
                        st.write(f"Confidence: **{data['confidence'] * 100:.1f}%**")
                        probs = pd.Series(data["probabilities"]).sort_index()
                        st.bar_chart(probs)
                    else:
                        st.warning(r.json().get("detail", "Prediction failed"))
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ---------------- Upload tab ----------------
with tab_upload:
    st.caption(
        "Upload a photo or scan with several handwritten digits. "
        "Best results: dark ink on plain light paper, good lighting, digits not touching."
    )
    up = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if up is not None:
        img = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        st.image(img, caption="Your image", use_column_width=True)

        if st.button("Recognize digits", type="primary"):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            buf.seek(0)
            try:
                with st.spinner("Reading digits..."):
                    r = requests.post(
                        f"{API_URL}/predict-image",
                        files={"file": ("image.jpg", buf, "image/jpeg")},
                        timeout=60,
                    )
                if r.status_code != 200:
                    st.warning(r.json().get("detail", "Could not read the image"))
                else:
                    data = r.json()

                    # Annotated image
                    out = img.copy()
                    W, H = out.size
                    draw = ImageDraw.Draw(out)
                    size = max(16, H // 22)
                    try:
                        font = ImageFont.load_default(size=size)
                    except TypeError:
                        font = ImageFont.load_default()
                    for d in data["digits"]:
                        x, y, w, h = d["box"]
                        x0, y0, x1, y1 = x * W, y * H, (x + w) * W, (y + h) * H
                        color = "lime" if d["confidence"] >= 0.7 else "orange"
                        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, W // 250))
                        draw.text((x0, max(0, y0 - size - 2)), str(d["digit"]), fill=color, font=font)
                    st.image(out, caption="Green = confident, orange = unsure", use_column_width=True)

                    st.subheader(f"Found {data['count']} digit(s)")
                    st.code("\n".join(data["lines"]), language=None)

                    table = pd.DataFrame(
                        [
                            {
                                "#": i + 1,
                                "digit": d["digit"],
                                "confidence %": round(d["confidence"] * 100, 1),
                            }
                            for i, d in enumerate(data["digits"])
                        ]
                    )
                    st.dataframe(table, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Request failed: {e}")
