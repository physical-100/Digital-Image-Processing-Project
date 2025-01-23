import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # 샤프닝 필터 적용
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    # 대비 증가
    contrast = cv2.convertScaleAbs(sharpened, alpha=2.0, beta=0)
    canny = cv2.Canny(contrast,50,150)
    return canny


def find_circles(image):
    #원 탐지
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=300,
                               param1=150, param2=50, minRadius=30, maxRadius=100)
    return circles

# 이미지 불러오기
image_path = '/Users/jincheol/Desktop/디지털 영상 처리/project/스크린샷 2024-06-10 오후 4.39.16.png'
image = cv2.imread(image_path)

# 전처리 적용
processed_image = preprocess_image(image)

# 원 탐지
circles = find_circles(processed_image)

# 결과 출력
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.circle(image, (x, y), 2, (0, 128, 255), 3)
    print(f"Detected circles: {circles}")
else:
    print("No circles detected")

# 결과 이미지 표시
cv2.imshow("Processed Image", processed_image)
cv2.imshow("Detected Circles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

