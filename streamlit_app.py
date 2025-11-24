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

    bg = cv2.medianBlur(v, 61)
    diff = cv2.absdiff(v, bg)
    shadow = cv2.normalize(255 - diff, None, 0, 255, cv2.NORM_MINMAX)

    v2 = cv2.addWeighted(v, 0.85, shadow, 0.15, 0)

    final_hsv = cv2.merge([h, s, v2])
    final = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final

def remove_shadow_white_color(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 10
    )
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def safe_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def auto_shadow_removal(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    S_std = np.std(s)      
    V_mean = np.mean(v)     
    V_std  = np.std(v)      
    V_med  = np.median(v)
    
    gray_pixels = np.sum(s < 30)   
    total_pixels = s.size
    gray_ratio = gray_pixels / total_pixels

    if gray_ratio > 0.70 :
        return remove_shadow_white_color(img)
    elif V_med > 140 and V_std < 40 and S_std < 25:
        return remove_shadow_white_color(img)
    else:
        return remove_shadow_preserve_color(img)

def enhance_final_preserve_color(img):
    img = auto_shadow_removal(img)
    img = safe_sharpen(img)
    return img

# start streamlit
st.title("📄 Multiple Document Scanner")
st.write("สแกนเอกสารหลายแผ่นได้อย่างรวดเร็วและแม่นยำ — เสร็จในคลิกเดียว")

uploaded = st.file_uploader("อัปโหลดภาพเอกสาร", type=["jpg","jpeg","png"])

if uploaded:

    model = YOLO("modelv2.pt")

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    H, W = image.shape[:2]

    # Predict segmentation
    res = model.predict(image, conf=0.5)[0]

    if res.masks is None:
        st.error("❌ ไม่พบเอกสารในภาพ")
        st.stop()

    masks = res.masks.data.cpu().numpy()
    upsampled_masks = []

    for m in masks:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        upsampled_masks.append(m)

    contours = []
    for m in upsampled_masks:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)

    min_area = 50000
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(contours) == 0:
        st.error("❌ ไม่พบเอกสารในภาพ")
        st.stop()

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)]
        ], dtype="float32")

    A4_w, A4_h = 2480, 3508
    trim_border = 50

    output_images = []
    show_preview = st.checkbox("แสดงตัวอย่างผลลัพธ์ (Preview)", value=True)

    for i, c in enumerate(contours):

        # เตรียม mask ของใบนี้
        mask_i = upsampled_masks[i]
    
        # ขนาด mask
        h_mask, w_mask = mask_i.shape
        need_rotate = w_mask > h_mask  # แนวนอน → หมุน
    
        # สำรองภาพก่อนหมุน
        original_image = image.copy()
        original_mask  = mask_i.copy()
    
        if need_rotate:
            # หมุนภาพจริง
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
            # หมุน mask
            mask_i = cv2.rotate(mask_i, cv2.ROTATE_90_CLOCKWISE)
    
            # อัปเดต H,W ใหม่หลังหมุน
            H, W = image.shape[:2]
    
            # หา contour ใหม่บน mask หลังหมุน
            cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            c = max(cnts, key=cv2.contourArea)
    
        # หา approx polygon ใหม่ตามภาพปัจจุบัน (อาจถูกหมุน)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect)
        approx = approx.reshape(4,2).astype(np.float32)
    
        # จัดลำดับจุดใหม่
        src = order_points(approx)
    
        # จุดปลายทาง A4
        dst = np.array([
            [0,0],
            [A4_w-1,0],
            [A4_w-1,A4_h-1],
            [0,A4_h-1]
        ], np.float32)
    
        # Homography
        H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        warped = cv2.warpPerspective(image, H_mat, (A4_w, A4_h))
    
        # crop
        cropped = warped[
            trim_border:A4_h-trim_border,
            trim_border:A4_w-trim_border
        ]
    
        # enhance
        cropped = enhance_final_preserve_color(cropped)
    
        # หมุนกลับถ้าใบนี้เป็นแนวนอน
        if need_rotate:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
            # 🔥 restore ภาพหลักเพื่อทำใบถัดไป
            image = original_image
            mask_i = original_mask
            H, W = image.shape[:2]
    
        # แสดงผล
        if show_preview:
            st.subheader(f"ผลลัพธ์หน้า {i+1}")
            st.image(
                cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
                caption=f"Document {i+1}",
                use_column_width=True
            )
    
        output_images.append(cropped)


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
