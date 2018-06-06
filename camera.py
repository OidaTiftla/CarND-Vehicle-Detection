import pickle
import cv2

class Camera:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def save(self, fname):
        pickle.dump((self.mtx, self.dist), open(fname, "wb"))

    @classmethod
    def from_file(cls, fname):
        mtx, dist = pickle.load(open(fname, "rb"))
        return cls(mtx, dist)

    def undistort(self, img):
        # Undistorts an image
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
