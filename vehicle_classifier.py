import pickle
import cv2

class VehicleClassifier:
    def __init__(self):
        pass

    def save(self, fname):
        pickle.dump((None, ), open(fname, "wb"))

    @classmethod
    def from_file(cls, fname):
        _, = pickle.load(open(fname, "rb"))
        return cls()

    def classify(self, img):
        return False

    def train(self, img):
        return False
