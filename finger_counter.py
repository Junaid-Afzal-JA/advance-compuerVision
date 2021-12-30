import cv2

from hand_tracking.hand_tracking import HandDetector


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detect = HandDetector()
    while True:
        _, image = cap.read()
        cv2.rectangle(img=image,
                      pt1=(0, 0),
                      pt2=(100, 100),
                      color=(0, 255, 0),
                      thickness=cv2.FILLED
                      )

        if _:
            detect.find_hands(image)
            result = detect.find_position(image=image)
            count = 0
            if len(result) > 0:
                if result[8][1] > result[6][1]:  # checking finger
                    count += 1
                if result[4][1] > result[3][1]:  # checking thumb
                    count += 1
                if result[12][1] > result[10][1]:  # checking middle finger
                    count += 1
                if result[16][1] > result[14][1]:  # checking 4th finger
                    count += 1
                if result[20][1] > result[18][1]:
                    count += 1

                print(count)
                cv2.putText(img=image,
                            text=str(count),
                            org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 0, 0),
                            thickness=2
                            )

            cv2.imshow('live', image)
            cv2.waitKey(1)
