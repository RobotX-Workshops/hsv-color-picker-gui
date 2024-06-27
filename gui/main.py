import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import Any, Literal, Optional, Union, cast
import numpy.typing as npt

from gui.utils import apply_hsv_filter


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("HSV Color Filter App")

    # Sidebar for input options and sliders
    with st.sidebar:
        st.header("Controls")
        source: Optional[Literal['Upload Image', 'Camera Feed']] = st.radio(
            "Select input source:",
            ("Upload Image", "Camera Feed")
        )

        st.subheader("HSV Color Filter")
        h_min: int = st.slider("Hue Min", 0, 179, 0)
        h_max: int = st.slider("Hue Max", 0, 179, 179)
        s_min: int = st.slider("Saturation Min", 0, 255, 0)
        s_max: int = st.slider("Saturation Max", 0, 255, 255)
        v_min: int = st.slider("Value Min", 0, 255, 0)
        v_max: int = st.slider("Value Max", 0, 255, 255)

    image: Optional[npt.NDArray[np.uint8]] = None
    cap: Union[cv2.VideoCapture, None] = None

    # Main area for image display
    col1, col2 = st.columns(2)

    if source == "Upload Image":
        uploaded_file: Optional[Any] = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_image: Image.Image = Image.open(uploaded_file)
            image = np.array(pil_image.convert('RGB'), dtype=np.uint8)
            image = cast(npt.NDArray[np.uint8], cv2.cvtColor(
                image, cv2.COLOR_RGB2BGR))
        else:
            st.warning("Please upload an image.")
            return
    else:  # source == "Camera Feed"
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Unable to access the camera.")
            return
        ret: bool
        frame: cv2.typing.MatLike
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to capture frame from camera.")
            return
        image = cast(npt.NDArray[np.uint8], frame)

    with col1:

        st.subheader("Original Image")

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 use_column_width=True)

    with col2:
        st.subheader("Filtered Image")
        lower_hsv: npt.NDArray[np.uint8] = np.array(
            [h_min, s_min, v_min], dtype=np.uint8)
        upper_hsv: npt.NDArray[np.uint8] = np.array(
            [h_max, s_max, v_max], dtype=np.uint8)
        filtered_image: npt.NDArray[np.uint8] = apply_hsv_filter(
            image, lower_hsv, upper_hsv)
        st.image(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB),
                 use_column_width=True)

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
