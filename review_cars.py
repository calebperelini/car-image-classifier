"""
For passing images to the model for review, and comparing against carjam output.
"""

import os
import database
import carjam_soup
from model_eval import predict_image

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
        predictions.append(predict_image(f))
    return predictions

def compare_carjam(predictions: list):
    db = database.retreive_all()
    for pr in predictions:
        for entry in db:
            pr['plate_read'] = entry
    
    for car in predictions:
        carj_colour = carjam_soup.carjam_colour(car['plate_read'][1])
        if carj_colour is not None:
            if carj_colour == car['colour']:
                car['Match'] = True
            else:
                car['Match'] = False

def main():
    filenames = get_filenames('images')
    compare_carjam(get_predictions(filenames))

if __name__ == '__main__':
    main()