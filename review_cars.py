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

def compare_carjam(predictions: list) -> list:
    db = database.retreive_all()
    for i in range(len(predictions) - 1):
        predictions[i]['plate_read'] = db[i]
    
    for entry in predictions[:len(predictions) - 1]:
        entry_plate = entry['plate_read'][1]
        carj_colour = carjam_soup.carjam_colour(entry_plate)
        if carj_colour is not None:
            if carj_colour == entry['colour']:
                entry['Match'] = True
            else:
                entry['Match'] = False
    
    return predictions

def display(predictions: list):
    predictions = predictions[:-1]
    for pr in predictions:
        print("Plate Read: {}, Predicted Colour: {}, Match: {}".format(
            pr['plate_read'][1],
            pr['colour'], 
            pr['Match'])
            )


def main():
    filenames = get_filenames('images')
    predictions = compare_carjam(get_predictions(filenames))
    display(predictions)

if __name__ == '__main__':
    main()