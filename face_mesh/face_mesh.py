import cv2
import mediapipe as mp
import time


class FaceMesher:
    """
    Detect facial landmarks in images.
    """
    def __init__(self,

                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 thickness=1,
                 circle_radius=1
                 ):
        """
        Initialize Object of FaceMesher.
        Parameters:
            static_image_mode (bool):
            max_num_faces (int): How many face to detect in image default to 1.
            refine_landmarks (bool): Either refine landmarks or nor, default False.
            min_detection_confidence: Minimum confidence to detect face.
            min_tracking_confidence (float): Minimum confidence to track face.
            thickness (int): drawing lines thickness
            circle_radius (int): Radius of circle that draw on face landmark.
        Returns:

        """

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                    max_num_faces=max_num_faces,
                                                    refine_landmarks=refine_landmarks,
                                                    min_detection_confidence=min_detection_confidence,
                                                    min_tracking_confidence=min_tracking_confidence
                                                    )
        self.draw = mp.solutions.drawing_utils
        self.specific_draw = self.draw.DrawingSpec(thickness=thickness, circle_radius=circle_radius)

    def get_face_mesh(self, image, draw=True, draw_mark_ids=False) -> list:
        """
        Parameters:
            image (np.array): Image to check for face mesh
            draw (bool): Either draw mask on detected face or not.
            draw_mark_ids (bool): Either draw mark_id or not.
        Returns: 
            faces_landmark (list): List for all faces landmarks 
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_h, im_w, i_c = image.shape
        result = self.face_mesh.process(rgb_image)
        faces_landmark = []

        # print(result.multi_face_landmarks)
        if result.multi_face_landmarks:
            # print(len(result.multi_face_landmarks))
            for landmarks in result.multi_face_landmarks:
                face = {}
                for mark_id, mark in enumerate(landmarks.landmark):
                    face[mark_id] = (int(mark.x * im_w), int(mark.y * im_h))
                    if draw_mark_ids:
                        cv2.putText(
                            img=image,
                            text=f'{mark_id}',
                            org=(int(mark.x * im_w), int(mark.y * im_h)),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=0.5,
                            color=(0, 255, 0),
                            thickness=1
                        )

                faces_landmark.append(face)
                if draw:
                    self.draw.draw_landmarks(
                        image,
                        landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        self.specific_draw,
                        self.specific_draw
                    )
        return faces_landmark

    @staticmethod
    def draw_mark(image, pos, radius=2, color=(0, 0, 0), thickness=5):
        """
        Parameters:
            image:
            pos:
            radius:
            color:
            thickness:
        :return:
        """
        cv2.circle(
            img=image,
            center=pos,
            radius=radius,
            color=color,
            thickness=thickness
        )


if __name__ == '__main__':
    mesher = FaceMesher()
    cap = cv2.VideoCapture('../data/v5.mkv')
    # cap = cv2.VideoCapture(0)
    pre_time = 0
    while True:
        _, image = cap.read()
        if _:
            curr_time = time.time()
            fps = int(1 / (curr_time - pre_time))
            pre_time = curr_time
            faces = mesher.get_face_mesh(image)
            if len(faces) > 0:
                # draw_mark(image, faces[0][385])
                mesher.draw_mark(image, faces[0][151])

            cv2.putText(
                img=image,
                text=f'FPS: {fps}',
                org=(8, 15),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2
            )
            cv2.imshow('image', image)
            cv2.waitKey(1)
