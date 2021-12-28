import cv2
import numpy as np
from hand_tracking.hand_tracking import HandDetector
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


if __name__ == '__main__':
    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    v_min, v_max, _ = volume.GetVolumeRange()
    cap = cv2.VideoCapture(0)
    detect = HandDetector()
    vol_bar = 400
    vol_bar_per = 0
    mid = 0, 0
    while True:
        _, image = cap.read()
        if _:
            detect.find_hands(image, draw=True)
            marks = detect.find_position(image)
            if len(marks) > 0:
                thumb = (marks[8][1], marks[8][2])
                finger = (marks[4][1], marks[4][2])
                cv2.circle(img=image,
                           center=thumb, radius=14,
                           color=(0, 0, 0),
                           thickness=cv2.FILLED,
                           )
                cv2.circle(img=image,
                           center=finger, radius=14,
                           color=(0, 0, 0),
                           thickness=cv2.FILLED
                           )
                cv2.line(img=image,
                         pt1=thumb,
                         pt2=finger,
                         color=(255, 0, 0),
                         thickness=3
                         )

                diff = int(math.hypot(thumb[0] - finger[0], thumb[1] - finger[1]))
                mid = int((thumb[0] + finger[0]) / 2), int((thumb[1] + finger[1]) / 2)
                vol_bar = int(np.interp(diff, [30, 180], [400, 100]))
                vol_bar_per = int(np.interp(diff, [30, 180], [0, 100]))
                new_vol_lvl = int(np.interp(diff, [30, 180], [v_min, v_max]))
                volume.SetMasterVolumeLevel(new_vol_lvl, None)

            cv2.rectangle(img=image,
                          pt1=(20, 100),
                          pt2=(50, 400),
                          color=(0, 255, 0),
                          thickness=2
                          )
            cv2.rectangle(img=image,
                          pt1=(50, 400),
                          pt2=(20, vol_bar),
                          color=(0, 255, 0),
                          thickness=cv2.FILLED
                          )
            cv2.putText(img=image,
                        text=str(vol_bar_per) + '%',
                        org=(20, 90),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(0, 255, 0),
                        thickness=2
                        )

            cv2.imshow('image', image)

            cv2.waitKey(1)
