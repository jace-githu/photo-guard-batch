# photo_guard_free.py  (simple: too_dark | text_heavy | ok)
import cv2, numpy as np, pytesseract, csv, argparse, requests
from PIL import Image

# ---- 튜닝 가능한 임계값 ----
# 너무 어두움: V채널 평균/표준편차 + 저밝기 픽셀 비율로 판단
V_MEAN_THR = 40          # 평균 밝기(0~255)
V_STD_THR  = 20          # 밝기 분산이 너무 크면 제외 (플래시/밝은 영역 혼재 방지)
DARK_RATIO_THR = 0.80    # 매우 어두운 픽셀(<=25)이 80% 이상이면 too_dark

# 텍스트 과다: OCR 글자수 + 텍스트 박스 면적 비율
TEXT_CHAR_THR = 120      # 글자 수 기준
TEXT_AREA_RATIO_THR = 0.08  # 텍스트 bbox 총 면적 / 이미지 면적

def is_too_dark(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v_mean = float(np.mean(v))
    v_std  = float(np.std(v))
    dark_ratio = float(np.mean(v <= 25))
    return (v_mean < V_MEAN_THR and v_std < V_STD_THR) or (dark_ratio >= DARK_RATIO_THR)

def text_stats(img, lang="kor+eng"):
    # OCR 결과: 글자수 / 텍스트 박스 면적비
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, lang=lang, output_type=pytesseract.Output.DICT)
    text = pytesseract.image_to_string(pil, lang=lang)
    num_chars = len(text.strip())

    h, w = img.shape[:2]
    total = h * w
    box_area = 0
    for i in range(len(data['text'])):
        s = data['text'][i].strip()
        conf = data['conf'][i]
        conf = int(conf) if isinstance(conf, str) and conf.isdigit() else -1
        if s and conf >= 60:
            box_area += int(data['width'][i]) * int(data['height'][i])
    area_ratio = (box_area / total) if total > 0 else 0.0
    return num_chars, area_ratio

def analyze_image(url, lang="kor+eng"):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return "error", "이미지를 열 수 없음"

        # A) 너무 어두움
        if is_too_dark(img):
            return "too_dark", "어두움"

        # B) 텍스트 과다
        num_chars, area_ratio = text_stats(img, lang=lang)
        if num_chars >= TEXT_CHAR_THR and area_ratio >= TEXT_AREA_RATIO_THR:
            return "text_heavy", f"글자수 {num_chars}, 면적비 {area_ratio:.3f}"

        # C) 정상
        return "ok", "정상"
    except Exception as e:
        # Tesseract 미설치/실패 등도 여기로
        return "error", str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=4)  # 호환용(미사용)
    ap.add_argument("--tesslang", default="kor+eng")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        label, reason = analyze_image(r["url"], lang=args.tesslang)
        out.append({"photo_id": r["photo_id"], "label": label, "reason": reason})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["photo_id","label","reason"])
        w.writeheader(); w.writerows(out)

if __name__ == "__main__":
    main()
