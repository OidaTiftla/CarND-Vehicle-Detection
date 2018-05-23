import pickle
import cv2

class Camera:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def save(self, fname):
        pickle.dump((self.mtx, self.dist), open(fname, "wb"))

    def load(self, fname):
        self.mtx, self.dist = pickle.load(open(fname, "rb"))

    def undistort(self, img):
        # Undistorts an image
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
