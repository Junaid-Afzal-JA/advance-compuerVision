import mediapipe as mp
import cv2
import time


class FaceDetector:
    """
    Detect Face in Image
    """
    def __init__(self, min_detection_confidence=0.5):
        self.mp_fd = mp.solutions.face_detection
        self.draw = mp.solutions.drawing_utils
        self.face_detector = self.mp_fd.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect_face(self, image, draw=True):
        """
        Parameters:
            image: Image in the form of np.array
            draw: Draw Detection on Image or not.
        Returns:
             image: resultant image
             data: contains face_id, points, confidence score.
        """
        img_h, img_w, img_ch = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_detector.process(img_rgb)
        data = []
        if result.detections:
            for f_id, detect in enumerate(result.detections):
                score = round(detect.score[0], 2)
                x1 = detect.location_data.relative_bounding_box.xmin
                y1 = detect.location_data.relative_bounding_box.ymin

                x2 = detect.location_data.relative_bounding_box.width + x1
                y2 = detect.location_data.relative_bounding_box.height + y1

                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)

                x2 = int(x2 * img_w)
                y2 = int(y2 * img_h)

                face_position = ((x1, y1), (x2, y2))
                data.append([f_id, face_position, score])

                if draw:
                    self.__stylish_draw(image, detect.location_data.relative_bounding_box)

                    cv2.putText(
                                img=image,
                                text=f'{str(score*100)}%',
                                org=(x1, y1 - 5),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(100, 33, 33),
                                thickness=2
                    )

        return image, data

    @staticmethod
    def __stylish_draw(self, image, face_loc, line_length=20, line_thickness=4):
        """
        Draw boundaries around rectangle
        Parameters:
            image: Image on which draw
            face_loc: Points where rectangle draw
            line_length: boundary length
            line_thickness: boundary thickness
        Returns:
        """
        img_h, img_w, img_ch = image.shape
        x1 = face_loc.xmin
        y1 = face_loc.ymin

        x2 = face_loc.width + x1
        y2 = face_loc.height + y1

        x1_p = int(x1 * img_w)
        y1_p = int(y1 * img_h)
        x2_p = int(x2 * img_w)
        y2_p = int(y2 * img_h)

        cv2.rectangle(
            img=image,
            pt1=(x1_p, y1_p),
            pt2=(x2_p, y2_p),
            color=(0, 255, 0),
            thickness=1
        )
        # Drawing top left boundary
        cv2.line(
            img=image,
            pt1=(x1_p, y1_p),
            pt2=(x1_p + line_length, y1_p),
            color=(0, 255, 0),
            thickness=line_thickness
        )
        cv2.line(
            img=image,
            pt1=(x1_p, y1_p),
            pt2=(x1_p, y1_p + line_length),
            color=(0, 255, 0),
            thickness=line_thickness
        )

        # Drawing top right boundary
        cv2.line(
            img=image,
            pt1=(x2_p, y1_p),
            pt2=(x2_p - line_length, y1_p),
            color=(0, 255, 0),
            thickness=line_thickness
        )
        cv2.line(
            img=image,
            pt1=(x2_p, y1_p),
            pt2=(x2_p, y1_p + line_length),
            color=(0, 255, 0),
            thickness=line_thickness
        )

        # Drawing bottom right boundary
        cv2.line(
            img=image,
            pt1=(x2_p, y2_p),
            pt2=(x2_p - line_length, y2_p),
            color=(0, 255, 0),
            thickness=line_thickness
        )
        cv2.line(
            img=image,
            pt1=(x2_p, y2_p),
            pt2=(x2_p, y2_p - line_length),
            color=(0, 255, 0),
            thickness=line_thickness
        )

        # Drawing bottom left boundary
        cv2.line(
            img=image,
            pt1=(x1_p, y2_p),
            pt2=(x1_p + line_length, y2_p),
            color=(0, 255, 0),
            thickness=line_thickness
        )
        cv2.line(
            img=image,
            pt1=(x1_p, y2_p),
            pt2=(x1_p, y2_p - line_length),
            color=(0, 255, 0),
            thickness=line_thickness
        )


if __name__ == '__main__':
    detect = FaceDetector()
    cap = cv2.VideoCapture('../data/v5.mkv')
    pre_time = 0
    while True:
        _, image = cap.read()
        if _:
            img_h, img_w, img_ch = image.shape
            curr_time = time.time()
            fps = int(1 / (curr_time - pre_time))
            pre_time = curr_time
            img, result = detect.detect_face(image)

            cv2.putText(img, f'FPS: {str(fps)}', (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imshow('video', img)
            cv2.waitKey(40)

        else:
            break
