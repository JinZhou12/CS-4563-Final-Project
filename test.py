from sklearn.preprocessing import StandardScaler  # It is important in neural networks to scale the date
from sklearn.model_selection import train_test_split  # The standard - train/test to prevent overfitting and choose hyperparameters
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import numpy.random as r # We will randomly initialize our weights
import matplotlib.pyplot as plt 
from PIL import Image
import os

path = './NonsegmentedV2/'

im = Image.open(path + 'Maize' + '/' + '2.png').resize((64, 64))

img = np.asarray( im, dtype="int32" )[:,:,:3]

plt.matshow(img)
plt.show()