"""
Takes an mp4 file and outputs a series of images using OpenCV.
"""
import requests
import cv2
from io import BytesIO
from PIL import Image


def capture_frames() -> list:

    capture = cv2.VideoCapture("nz_cars_ALPR.mp4")

    frame_counter = 0
    fps = 30
    sample_rate = 1  # Set sample rate to 1fps.
    images = []

    while(capture.isOpened()):

        ret, frame = capture.read()
        if not ret:
            break
        if frame_counter % (fps // sample_rate) == 0:
            images.append(Image.fromarray(frame))
        frame_counter += 1

    capture.release()
    cv2.destroyAllWindows()
    return images


def evaluate_images(image_array: list):
    for img in image_array:
        byte_io = BytesIO()
        img.save(byte_io, 'png')
        byte_io.seek(0)  # seek start of I/O stream.


def main():
    image_array = capture_frames()
    evaluate_images(image_array)


if __name__ == "__main__":
    main()
