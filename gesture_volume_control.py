import cv2
from hand_tracking.hand_tracking import HandDetector

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detect = HandDetector()

    while True:
        _, image = cap.read()
        if _:
            detect.find_hands(image, draw=True)
            marks = detect.find_position(image)
            if len(marks) > 0:
                thumb = (marks[8][1], marks[8][2])
                finger = (marks[4][1], marks[4][2])
                cv2.circle(img=image,
                           center=thumb, radius=10,
                           color=(0, 0, 0),
                           thickness=cv2.FILLED,
                           )
                cv2.circle(img=image,
                           center=finger, radius=10,
                           color=(0, 0, 0),
                           thickness=cv2.FILLED
                           )
                cv2.line(img=image,
                         pt1=thumb,
                         pt2=finger,
                         color=(255, 0, 0),
                         thickness=3
                         )

            cv2.rectangle(img=image,
                          pt1=(20, 100),
                          pt2=(50, 400),
                          color=(0, 255, 0),
                          thickness=2
                          )
            cv2.imshow('image', image)

            cv2.waitKey(1)
