import time

import mediapipe as mp
import cv2


class HandDetector:
    """
    Track hand in live time
    """
    def __init__(self, mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.my_hands = self.mp_hands.Hands(static_image_mode=mode,
                                            max_num_hands=max_num_hands,
                                            model_complexity=model_complexity,
                                            min_detection_confidence=min_detection_confidence,
                                            min_tracking_confidence=min_tracking_confidence)
        self.result = None
        self.temp_image = None
        self.draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        """
        Message:
            Take image in BGR and draw landmarks of hands on it.
        Parameters:
            image:
            draw: Draw landmarks or not
        Returns:
             image: image with landmarks
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.my_hands.process(rgb_image)
        self.temp_image = image
        if self.result.multi_hand_landmarks:
            for hands in self.result.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(image, hands, self.mp_hands.HAND_CONNECTIONS)

        return image

    def find_position(self, image, hand_no=0, draw=False):
        """
        Message:
            Return landmarks position of hands
        Parameters:
            image:
            hand_no: Which hand to identify
            draw: Draw landmark or not.
        """
        height, width, channel = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.my_hands.process(rgb_image)
        land_marks = []
        if result.multi_hand_landmarks:
            specific_hand = result.multi_hand_landmarks[hand_no]
            for mark_no, mark in enumerate(specific_hand.landmark):
                x_point = int(mark.x * width)
                y_point = int(mark.y * height)
                land_marks.append([mark_no, x_point, y_point])
                if draw:
                    cv2.circle(img=image,
                               center=(x_point, y_point), radius=8,
                               color=(210, 50, 50),
                               thickness=2
                               )

        return land_marks
