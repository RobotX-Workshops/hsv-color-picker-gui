import cv2
import numpy as np
from typing import cast
import numpy.typing as npt


def apply_hsv_filter(image: npt.NDArray[np.uint8],
                     lower_hsv: npt.NDArray[np.uint8],
                     upper_hsv: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:

    hsv = cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask = cast(npt.NDArray[np.uint8], cv2.inRange(hsv, lower_hsv, upper_hsv))
    result = cast(npt.NDArray[np.uint8],
                  cv2.bitwise_and(image, image, mask=mask))
    return result
