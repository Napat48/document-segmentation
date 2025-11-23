import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import img2pdf
import tempfile
import os


def remove_shadow_preserve_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    bg = cv2.medianBlur(v, 31)
    diff = cv2.absdiff(v, bg)
    shadow = cv2.normalize(255 - diff, None, 0, 255, cv2.NORM_MINMAX)

    v2 = cv2.addWeighted(v, 0.85, shadow, 0.15, 0)

    final_hsv = cv2.merge([h, s, v2])
    final = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final


def safe_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)


def enhance_final_preserve_color(img):
    img = remove_shadow_preserve_color(img)
    img = safe_sharpen(img)
    return img


# start streamlit
st.title("📄 Multiple Document Scanner")
st.write("สแกนเอกสารหลายแผ่นได้อย่างรวดเร็วและแม่นยำ — เสร็จในคลิกเดียว")

uploaded = st.file_uploader("อัปโหลดภาพเอกสาร", type=["jpg","jpeg","png"])

if uploaded:

    # โหลดโมเดล
    model = YOLO("modelv2.pt")

    # โหลดภาพจาก upload
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    H, W = image.shape[:2]

    # Predict segmentation
    res = model.predict(image, conf=0.5)[0]

    if res.masks is None:
        st.error("❌ ไม่พบเอกสารในภาพ")
        st.stop()

    # Upsample masks
    masks = res.masks.data.cpu().numpy()
    upsampled_masks = []

    for m in masks:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        upsampled_masks.append(m)

    # หา contour
    contours = []
    for m in upsampled_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)

    # ---- กรองเอกสารที่เล็กเกินไป ----
    min_area = 50000  # ปรับตามต้องการ
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(contours) == 0:
        st.error("❌ ไม่พบเอกสารในภาพ")
        st.stop()
    # -------------------------------

    # เลือก 2 ใบใหญ่สุด
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],     # TL
            pts[np.argmin(diff)],  # TR
            pts[np.argmax(s)],     # BR
            pts[np.argmax(diff)]   # BL
        ], dtype="float32")


    A4_w, A4_h = 2480, 3508
    trim_border = 50

    output_images = []
    show_preview = st.checkbox("แสดงตัวอย่างผลลัพธ์ (Preview)", value=True)

    for i, c in enumerate(contours):

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect)

        src = order_points(approx.reshape(4,2).astype(np.float32))
        dst = np.array([[0,0],[A4_w-1,0],[A4_w-1,A4_h-1],[0,A4_h-1]], np.float32)

        H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        warped = cv2.warpPerspective(image, H_mat, (A4_w, A4_h))

        # enhance แบบ preserve color
        cropped = warped[
            trim_border:A4_h-trim_border,
            trim_border:A4_w-trim_border
        ]
        cropped = enhance_final_preserve_color(cropped)

        if show_preview:
            st.subheader(f"ผลลัพธ์หน้า {i+1}")
            st.image(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                caption=f"Document {i+1}",
                use_column_width=True
            )

        output_images.append(cropped)

    # สร้าง PDF ใน temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:

        image_paths = []

        for idx, img in enumerate(output_images):
            temp_path = f"/tmp/page_{idx}.jpg"
            cv2.imwrite(temp_path, img)
            image_paths.append(temp_path)

        tmp_pdf.write(img2pdf.convert(image_paths))

        for p in image_paths:
            os.remove(p)

        st.success("✔ สร้าง PDF สำเร็จ!")

        with open(tmp_pdf.name, "rb") as f:
            st.download_button(
                label="📥 ดาวน์โหลด PDF",
                data=f.read(),
                file_name="scanned_documents.pdf",
                mime="application/pdf"
            )
