import cv2
import numpy as np
from PIL import Image
import pyocr
import pyocr.builders

# PyOCR 도구 설정
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    exit(1)
tool = tools[0]

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    canny = cv2.Canny(blurred, 50, 150)  # Canny edge detection
    return blurred, canny


def find_circles(frame, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100):
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles

def calculate_circularity(circles):
    circularity = []
    for (x, y, r) in circles:
        area = np.pi * (r ** 2)
        perimeter = 2 * np.pi * r
        circ = (4 * np.pi * area) / (perimeter ** 2)
        circularity.append(circ)
    return circularity

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's 이진화
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
    return processed_image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    # 첫 번째 프레임에서 원 찾기
    blurred_frame, canny_frame = preprocess_frame(frame)
    # circles = find_circles(canny_frame)  # Canny edge image로 원 찾기
    circles = find_circles(canny_frame) 
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circularity = calculate_circularity(circles)
        
        # 가장 원에 가까운 원 두 개 선택
        sorted_circles = sorted(zip(circularity, circles), reverse=True, key=lambda x: x[0])
        best_circles = [circle for _, circle in sorted_circles[:1]]
        
        for (x, y, r) in best_circles:
            # 원의 영역 추출
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
            circle_roi = frame[y1:y2, x1:x2]

            # 원의 영역을 확대
            scale_factor = 2  # 확대 배율
            enlarged_circle_roi = cv2.resize(circle_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            # 전처리 적용
            processed_image = preprocess_for_ocr(enlarged_circle_roi)

            # PIL 이미지로 변환
            pil_image = Image.fromarray(processed_image)

            # OCR로 숫자 인식
            digits = tool.image_to_string(
                pil_image,
                lang='eng',
                builder=pyocr.builders.DigitBuilder()
            )
            if digits:
                print(f"Detected digits in the circle: {digits.strip()}")
            else:
                print("Failed to detect digits in the circle")

            # 확대된 영역을 화면에 표시
            # cv2.imshow(f"Enlarged Circle ROI {x}-{y}", enlarged_circle_roi)
            cv2.imshow(f"Processed Circle ROI {x}-{y}", processed_image)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)  # 원을 원본 프레임에 그림
            cv2.circle(frame, (x, y), 2, (0, 128, 255), 3)  # 원의 중심을 원본 프레임에 그림

    # 첫 번째 프레임을 화면에 표시
    cv2.imshow("First Frame with Circles", frame)
    # cv2.imshow("Canny Edge", canny_frame)
    # cv2.imshow("sharpened", blurred_frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

def main(video_path):
    process_video(video_path)

if __name__ == "__main__":
    main('/Users/jincheol/Desktop/디지털 영상 처리/project/Nick/shrug.mp4')
    # main('/Users/jincheol/Desktop/디지털 영상 처리/project/front_output.mp4')
