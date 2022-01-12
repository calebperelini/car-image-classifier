"""
Takes an mp4 file and outputs a series of images using OpenCV.
"""
import requests
import cv2
import database
from io import BytesIO
from PIL import Image


def capture_frames():

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
            # convert colour space from BGR to RGB.
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(RGB_frame))
        frame_counter += 1

    capture.release()
    cv2.destroyAllWindows()
    return images


# process images into bytestream, send to API.
def evaluate_images(image_array: list) -> list:
    responses = []
    for img in image_array:

        byte_io = BytesIO()
        img.save(byte_io, 'png')
        byte_io.seek(0)  # seek start of I/O stream.
        regions = ['nz']  # for greater region prediction.

        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),
            files=dict(upload=byte_io),
            headers={'Authorization': 'Token c5dde1d98512a9f4d442c16d38ac68f2b6bf36c7'})
        responses.append(response.json())

    return responses


def db_store(responses):
    entries = []
    for i in responses:
        try:
            entries.append([
                (i['results'][0]['plate']),
                (i['results'][0]['score']),
                (i['processing_time'])
            ])
        except IndexError:
            print("End of Entries")
            break
    database.init_db()
    database.add_many(entries)
    database.show_all()


def main():
    image_array = capture_frames()
    results = evaluate_images(image_array)  # to be stored in SQLite DB
    db_store(results)


if __name__ == "__main__":
    main()
