from scipy.misc import imread
from glob2 import glob
import numpy as np

faces_dir = 'data/face/faces'
background_dir = 'data/face/background'

face_files = glob(faces_dir + "/*.jpg")
bg_files = glob(background_dir + "/*.jpg")
faces = np.vstack((imread(f)[None,] for f in face_files)).mean(axis=3)
bgs = np.vstack((imread(f)[None,] for f in bg_files)).mean(axis=3)

np.save('face.npy', faces)
np.save('nonface.npy', bgs)
