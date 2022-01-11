"""
Takes an mp4 file and outputs a series of images using OpenCV.
"""
import cv2

capture = cv2.VideoCapture("nz_cars_ALPR.mp4")


def capture_frames():
    capture = cv2.VideoCapture("nz_cars_ALPR.mp4")

    frame_counter = 0
    fps = 30
    sample_rate = 1  # Set sample rate to 1fps.

    while(capture.isOpened()):

        ret, frame = capture.read()
        if not ret:
            break
        if frame_counter % (fps // sample_rate) == 0:
            cv2.imwrite('data/test' + str(frame_counter) + '.jpg', frame)
        frame_counter += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
