import cv2
import numpy as np
import pytesseract
import imagehash
from PIL import Image
import csv
import argparse
import os

def is_black_image(image, threshold=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = cv2.mean(gray)[0]
    return mean_val < threshold

def detect_text_density(image, tesslang="eng"):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_img, lang=tesslang)
    num_chars = len(text.strip())
    return num_chars

def detect_map_like(image):
    # 간단히 색 분포와 edge를 보고 지도/캡처 비슷한 패턴인지 판별
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 채도가 낮고(회색 위주), edge가 많은 경우를 캡처로 가정
    sat_mean = np.mean(hsv[:,:,1])
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    return sat_mean < 60 and edge_density > 0.05

def analyze_image(url, tesslang="eng"):
    try:
        import requests
        resp = requests.get(url, timeout=15)
        img_array = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return "invalid", "이미지를 열 수 없음"

        if is_black_image(image):
            return "black", "검은 화면"
        text_chars = detect_text_density(image, tesslang)
        if text_chars > 15:  # 텍스트가 많으면 송장/문서 가능성
            return "text_only", f"글자 수 {text_chars}"
        if detect_map_like(image):
            return "map_capture", "지도/캡처 화면"
        return "ok", "정상"
    except Exception as e:
        return "error", str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input csv path")
    parser.add_argument("--output", required=True, help="output csv path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tesslang", type=str, default="eng")
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_rows = []
    for row in rows:
        pid = row["photo_id"]
        url = row["url"]
        label, reason = analyze_image(url, args.tesslang)
        out_rows.append({"photo_id": pid, "label": label, "reason": reason})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "label", "reason"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

if __name__ == "__main__":
    main()
