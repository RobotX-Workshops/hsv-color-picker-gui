import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import Any, Literal, Optional, Union, cast, List
import numpy.typing as npt
from gui.utils import apply_hsv_filter
import json
import io


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("HSV Color Filter App")

    # Sidebar for input options and sliders
    with st.sidebar:
        st.header("Controls")
        source: Optional[Literal['Upload Images', 'Camera Feed']] = st.radio(
            "Select input source:",
            ("Upload Images", "Camera Feed")
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
    uploaded_files: Optional[List[Any]] = []
    current_image_index: int = 0

    # Main area for image display
    col1, col2 = st.columns(2)

    if source == "Upload Images":
        uploaded_files = st.file_uploader(
            "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if not uploaded_files:
            st.warning("Please upload one or more images.")
            return

        st.success(f"Uploaded {len(uploaded_files)} images.")
        current_image_index = int(st.number_input(
            "Select image index", min_value=0, max_value=len(uploaded_files)-1, value=0, step=1))

        file = uploaded_files[current_image_index]
        pil_image = Image.open(io.BytesIO(file.read()))
        image = np.array(pil_image.convert('RGB'), dtype=np.uint8)
        image = cast(npt.NDArray[np.uint8],
                     cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

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

    # JSON Export and Display
    hsv_values = {
        "h_min": h_min,
        "h_max": h_max,
        "s_min": s_min,
        "s_max": s_max,
        "v_min": v_min,
        "v_max": v_max
    }
    json_string = json.dumps(hsv_values, indent=2)
    st.sidebar.subheader("HSV Values (JSON)")
    st.sidebar.code(json_string, language="json")

    if st.sidebar.button("Export HSV Values"):
        st.sidebar.download_button(
            label="Download HSV Values as JSON",
            data=json_string,
            file_name="hsv_values.json",
            mime="application/json"
        )

    if source == "Upload Images" and len(uploaded_files) > 1:
        if st.button("Next Image"):
            current_image_index = (
                current_image_index + 1) % len(uploaded_files)
            st.experimental_rerun()

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
