import mediapipe as mp
import cv2
import time


class PoseDetection:
    """
    Detect Pose of person
    """

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.__mp_pose = mp.solutions.pose

        self.__pose = self.__mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.__result = None
        self.__temp_image = None
        self.__draw = mp.solutions.drawing_utils

    def detect_pose(self, image, draw=True):
        """
        Message:
            Detect person pose
        Parameters:
             image: image to analyze
             draw: Draw landmarks not not.
        Returns:
            image: Image with detected pose
        """

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.__temp_image = image
        self.__result = self.__pose.process(rgb)
        if draw:
            self.__draw.draw_landmarks(image, self.__result.pose_landmarks, self.__mp_pose.POSE_CONNECTIONS)

        return image

    def find_pose_position(self, draw=False) -> list:
        """
        Message:
            Return the coordinates of detected pose used with detect_pose function.
        Parameters:
            draw: draw pose land_marks or not
        Returns:
             list
        """
        height, width, channel = self.__temp_image.shape
        marks = []
        if self.__result.pose_landmarks:
            for mark_id, mark in enumerate(self.__result.pose_landmarks.landmark):
                x_point = int(mark.x * width)
                y_point = int(mark.y * height)
                # print(mark_id, x_point, y_point)
                marks.append([mark_id, x_point, y_point])
                if draw:
                    cv2.circle(image, (x_point, y_point), 2, (200, 2, 82), 2)
        return marks


if __name__ == '__main__':

    cap = cv2.VideoCapture('data/cricket.MP4')
    pose_detector = PoseDetection()
    previous_time = 0
    while True:
        success, image = cap.read()
        if success:
            result = pose_detector.detect_pose(image, draw=False)
            coordinates = pose_detector.find_pose_position(draw=True)
            print(coordinates)

            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            fps = int(fps)
            previous_time = current_time

            cv2.putText(image, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (100, 100, 100), 3)
            cv2.imshow('Pose', result)
            cv2.waitKey(1)
        else:
            break

    print('end')
