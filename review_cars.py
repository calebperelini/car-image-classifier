"""
For passing images to the model for review, and comparing against carjam output.
"""

import os
from model_eval import predict_image, model

def get_filenames(path: str) -> list:
    filenames_list = []
    for filename in os.listdir(path):
        file = os.path.join(path,filename)
        if os.path.isfile(file):
            filenames_list.append(file)
    return filenames_list


def get_predictions(filename_list: list) -> list:
    predictions = []
    for f in filename_list:
        predictions.append(predict_image(model, f))
    print(predictions)
    print(len(predictions))

get_predictions(get_filenames('images'))