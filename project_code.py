import cv2
import numpy as np
import time
import easyocr

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return blurred

def detect_motion(prev_frame, curr_frame, min_contour_area=4000):
    frame_delta = cv2.absdiff(prev_frame, curr_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    return motion_contours, thresh, frame_delta

def initialize_tracker(frame, bbox):
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    return tracker

def extract_digits(roi, reader):
    result = reader.readtext(roi, detail=0)
    digits = ''.join(result)
    print("Extracted digits:", digits)
    return digits.strip()

def invert_and_change_background(image):
    inverted_image = cv2.bitwise_not(image)
    inverted_image[image == 255] = 0
    inverted_image[image == 0] = 255
    return inverted_image

def save_image(image, path):
    cv2.imwrite(path, image)
    print(f"Image saved to {path}")

def process_video(video_path, output_path):
    reader = easyocr.Reader(['en'])
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    prev_frame = preprocess_frame(frame)
    tracker_initialized = False
    tracker = None
    bbox = None
    min_y = float('inf')
    max_y = float('-inf')
    repetitions = 0
    in_range = False
    repetition_times = []
    time_changes = []
    last_rep_time = None
    initial_binarized_roi = None
    extracted_digits = ""
    saved_image = False

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 360, 640)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame = preprocess_frame(frame)

        if not tracker_initialized:
            motion_contours, thresh, frame_delta = detect_motion(prev_frame, curr_frame)
            if motion_contours:
                largest_motion_contour = max(motion_contours, key=cv2.contourArea)
                bbox = cv2.boundingRect(largest_motion_contour)
                tracker = initialize_tracker(frame, bbox)
                tracker_initialized = True
                prev_y = bbox[1] + bbox[3] // 2
                min_y = min(min_y, prev_y)
                max_y = max(max_y, prev_y)

                roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, initial_binarized_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)

                initial_binarized_roi = invert_and_change_background(initial_binarized_roi)

                extracted_digits = extract_digits(initial_binarized_roi, reader)
                print(f"Extracted digits: {extracted_digits}")

                save_image(initial_binarized_roi, 'binarized_roi.png')
                saved_image = True

                print(f"Initialized tracker with bbox: {bbox}")
        else:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                curr_y = y + h // 2
                min_y = min(min_y, curr_y)
                max_y = max(max_y, curr_y)
                range_threshold = min_y + 0.1 * (max_y - min_y)

                if curr_y <= range_threshold:
                    if not in_range:
                        repetitions += 1
                        in_range = True
                        current_time = time.time()
                        if last_rep_time is not None:
                            time_taken = current_time - last_rep_time
                            repetition_times.append(time_taken)
                            if len(repetition_times) > 1:
                                time_change = repetition_times[-1] - repetition_times[-2]
                                time_changes.append(time_change)
                            else:
                                time_change = 0
                        last_rep_time = current_time
                else:
                    in_range = False

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Repetitions: {repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if repetition_times:
                    cv2.putText(frame, f"Time for last rep: {repetition_times[-1]:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if len(repetition_times) > 1:
                        cv2.putText(frame, f"Time change: {time_change:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if initial_binarized_roi is not None:
                    right_top_y = 10
                    right_top_x = frame.shape[1] - w - 10
                    frame[right_top_y:right_top_y+h, right_top_x:right_top_x+w] = cv2.cvtColor(initial_binarized_roi, cv2.COLOR_GRAY2BGR)
                    cv2.putText(frame, f"weight: {extracted_digits}", (right_top_x, right_top_y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                print("Tracking failed")

        out.write(frame)
        cv2.imshow("Frame", frame)

        prev_frame = curr_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if repetition_times:
        average_time = sum(repetition_times) / len(repetition_times)
        print(f"Average time per repetition: {average_time:.2f} seconds")
    else:
        print("No repetitions detected")

    if time_changes:
        variance_time_change = np.var(time_changes)
        print(f"Variance of time changes: {variance_time_change:.4f}")
    else:
        print("No time changes detected")

def main(video_path, output_path):
    process_video(video_path, output_path)
if __name__ == "__main__":
    #비디오 입력 경로
    main('/Users/jincheol/Desktop/4-1/디지털 영상 처리/project/test video/Nick.mp4','result_video.mp4')
