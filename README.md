# Car Image Classification Model

This project aims to improve the accuracy of the [Plate Recognizer](https://platerecognizer.com/) ANPR by cross checking plate reads against an image classification model. 

The model itself was developed using Tensorflow with Keras, trained on the Vehicle Color Recognition Dataset, an annotated dataset of over 10,000 vehicles available [on kaggle.](https://www.kaggle.com/landrykezebou/vcor-vehicle-color-recognition-dataset)

## About

In setting out to design a system for improving the ANPR I looked to take an approach that was well suited to my abilities, whilst leaving room for growth and exploration. 

The code has been pre-compiled and executed in a series of jupyter notebooks for each respective step, the output of which can be viewed in each notebook. In addition, the source code for each step is also included in the repo for review.

### Part I. Extracting video frames and ANPR reads.

[Notebook](https://github.com/calebperelini/rushanpr/blob/main/partI.ipynb)

The first step was to extract frames from the provided `.mp4` and send them to the Plate Recognizer API for plate reads.

```python
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
```

Using the OpenCV library, frames were sampled from the file at 1 fps and returned as an array of image objects. 

An initial problem encountered was `capture.read()` returning an `image` object with BGR as it's default colour space. This was solved with the `cv2.cvtColor()` method, which allows us to convert the captured frame to RGB.

The next step was to convert the returned array of `image` objects into byte streams, which the API would accept.

```python
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
            headers={'Authorization': 'Token ' + SECRET_KEY}) # replace with your own key in a config.py file.
        responses.append(response.json())

    return responses
```

The API accepts `regions` as a parameter in the POST request, for greater accuracy when making reads on plates of specific regions. As we know all images in the set are of New Zealand cars, we can add the `'nz'` flag to the request.

Once we have an array of responses from the service, we can then filter the data we need and store them in a SQLite database.

```python

import database

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
```


The above method takes the plate read, confidence score, and processing time from the array of responses. It initialises a SQLite3 database, then adds them as entries by passing the data to a method in `database.py`.


```python
import sqlite3

# initialise db, create data table.
def init_db():
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.execute("""CREATE TABLE vehicles (
        plate_prediction text,
        score real,
        processing_time real
    )"""
              )
    print("Database initialised successfully.")
    con.commit()
    con.close()

def add_many(elements):
    con = sqlite3.connect('vehicles.db')
    c = con.cursor()
    c.executemany("INSERT INTO vehicles VALUES (?, ?, ?)",
                  (elements))
    con.commit()
    con.close()
```

### Part II. Evaluating Images with a model

[Notebook](https://github.com/calebperelini/rushanpr/blob/main/Model%20Training.ipynb)

This section was particularly challenging, due to a limited experience in Computer Vision and its associated tools. Given the time constraints and my ability, I decided to limit the scope of this step and focus on developing a model that would evaluate colour only.

The dataset used was the [Vehicle Colour Recognition Dataset](https://www.kaggle.com/landrykezebou/vcor-vehicle-color-recognition-dataset) (VCoR). This dataset consists was particularly useful due to its size and ease of use, with each class of colour pre-categorised into directories, and split 50-50 for training and testing.

Unclear on where to start, I referenced [Tensorflow's tutorial](https://www.tensorflow.org/tutorials/images/classification?fbclid=IwAR3dSGQ0W_EZEh_cr_LLXTNvGkZJqPsu1g6Li-ESI5jPffxvA0LABA9S6R8) for building image classification models with Tensorflow and Keras.



## Installation

Clone the repo and run `main.py`.
```python
git clone <this-repo>

python main.py

```


### Sources and external material referenced

- Plate Recognizer API Documentation
    - https://docs.platerecognizer.com/

- Extracting frames from video
    - https://www.askpython.com/python/examples/extract-images-from-video

- Preparing PIL objects for requests.
    - https://stackoverflow.com/questions/50350624/sending-pil-image-over-request-in-python

- Image Classification with Tensorflow
    - https://www.tensorflow.org/tutorials/images/classification?fbclid=IwAR3dSGQ0W_EZEh_cr_LLXTNvGkZJqPsu1g6Li-ESI5jPffxvA0LABA9S6R8



### Acknowledgements

- [The Vehicle Color Recognition Dataset](https://www.kaggle.com/landrykezebou/vcor-vehicle-color-recognition-dataset)
    - Panetta, Karen, Landry Kezebou, Victor Oludare, James Intriligator, and Sos Agaian. 2021. "Artificial Intelligence for Text-Based Vehicle Search, Recognition, and Continuous Localization in Traffic Videos" AI 2, no. 4: 684-704. (https://doi.org/10.3390/ai2040041)
    - Open access : (https://www.mdpi.com/2673-2688/2/4/41)