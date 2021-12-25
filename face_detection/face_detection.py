import mediapipe as mp
import cv2
import time


mp_fd = mp.solutions.face_detection
draw = mp.solutions.drawing_utils
face_detector = mp_fd.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture('../data/v1.mkv')
pre_time = 0
while True:
    _, image = cap.read()
    if _:
        curr_time = time.time()
        fps = int(1 / (curr_time - pre_time))
        pre_time = curr_time
        cv2.putText(image, f'FPS: {str(fps)}', (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_detector.process(img_rgb)
        if result.detections:
            for f_id, detect in enumerate(result.detections):
                print(detect)
                draw.draw_detection(image, detect)

        cv2.imshow('video', image)
        cv2.waitKey(40)

    else:
        break
