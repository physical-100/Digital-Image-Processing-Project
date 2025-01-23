import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return blurred

def detect_motion(prev_frame, curr_frame, min_contour_area=3000):
    frame_delta = cv2.absdiff(prev_frame, curr_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    return motion_contours, thresh, frame_delta

def calculate_direction(flow, x, y, w, h):
    flow_region = flow[y:y+h, x:x+w]
    mag, ang = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
    avg_angle = np.mean(ang)
    if 0 <= avg_angle < np.pi / 2 or 3 * np.pi / 2 <= avg_angle < 2 * np.pi:
        return "up"
    else:
        return "down"

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    prev_frame = preprocess_frame(frame)
    largest_motion_contour = None
    roi = None
    direction = None
    repetitions = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame = preprocess_frame(frame)
        motion_contours, thresh, frame_delta = detect_motion(prev_frame, curr_frame)

        if motion_contours and roi is None:
            # 가장 큰 움직임이 나타나는 영역 찾기
            largest_motion_contour = max(motion_contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(largest_motion_contour)
            roi = (x, y, w, h)
            print(f"Detected motion at: {roi}")

        if roi:
            x, y, w, h = roi
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            new_direction = calculate_direction(flow, x, y, w, h)

            if direction is not None and new_direction != direction:
                repetitions += 1
            direction = new_direction

            # ROI 영역 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Repetitions: {repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)

        prev_frame = curr_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(video_path):
    process_video(video_path)

if __name__ == "__main__":
    main('/Users/jincheol/Desktop/디지털 영상 처리/project/ohp.MP4')
