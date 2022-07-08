import os
import numpy as np
from tqdm import tqdm
from cv2 import cv2
import pandas as pd
import glob

from scipy.stats import kurtosis, skew, entropy

def get_variance(array):
    return np.var(array)

def get_mean(array):
    return np.mean(array)

def get_median(array):
    return np.median(array)

def get_deviation(array):
    return np.std(array)

def get_skewness(array):
    return skew(array)

def get_entropy(array):
    return entropy(array)

def get_kurtosis(array):
    return kurtosis(array)    

def read():
    cv_img = []
    for img in glob.glob(r"C:\Users\Win10\Downloads\FaceMaskDetection\Train\*.jpg"):
        n = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        cv_img.append(n)
    return cv_img

def read_images(smooth):
    histograms = []
    sobel_x = np.empty((3991, 64 * 64))
    sobel_y = np.empty((3991, 64 * 64))
    prewitt_x = np.empty((3991, 64 * 64))
    prewitt_y = np.empty((3991, 64 * 64))
    labels = np.empty((3991, 1), dtype=np.int8)
    i = 0

    print('Importing images...')

    img = read()
    print(len(img))
    for i in range(len(img)):
        if smooth == True:
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.erode(img[i], kernel, iterations=1)

        image = cv2.resize(img[i], (64, 64))

        sobel_x[i] = np.array(cv2.Sobel(image, cv2.CV_8U, 1, 0).flatten())
        sobel_y[i] = np.array(cv2.Sobel(image, cv2.CV_8U, 0, 1).flatten())

        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        prewitt_x[i] = np.array(cv2.filter2D(image, -1, kernel_x).flatten())
        prewitt_y[i] = np.array(cv2.filter2D(image, -1, kernel_y).flatten())

        histograms.append(cv2.calcHist([image], [0], None, [256], [0, 256]))
        i += 1
    llable = pd.read_csv(r"C:\Users\Win10\Downloads\FaceMaskDetection\train_labels.csv")   # loading the labels
    i=0
    fi = open('labels.txt', 'a')
    for f in llable['label']:
        if f == 'with_mask':
            label = 1
        else:
            label = 0
        fi.write(str(label))
        fi.write("\n")
        labels[i] = label
        i += 1
    print('Importing images completed')
    fi.close()
    print(labels)

    return histograms, sobel_x, sobel_y, prewitt_x, prewitt_y, labels


def construct_dataset(smooth):
    histogram, sobel_x, sobel_y, prewitt_x, prewitt_y, labels = read_images(smooth)

    variance = np.empty((3991, 1), dtype=np.float32)
    mean = np.empty((3991, 1), dtype=np.float32)
    median = np.empty((3991, 1), dtype=np.float32)
    std = np.empty((3991, 1), dtype=np.float32)
    skew = np.empty((3991, 1), dtype=np.float32)
    kurt = np.empty((3991, 1), dtype=np.float32)
    entropy = np.empty((3991, 1), dtype=np.float32)

    i = 0
    for hist in tqdm(histogram):
        variance[i] = get_variance(hist)
        mean[i] = get_mean(hist)
        median[i] = get_median(hist)
        std[i] = get_deviation(hist)
        skew[i] = get_skewness(hist)[0]
        kurt[i] = get_kurtosis(hist)[0]
        entropy[i] = get_entropy(hist)[0]
        i += 1

    print("feature extracted")

    features = np.empty((3991, 0))
    features = np.append(features, variance, axis=1)
    features = np.append(features, mean, axis=1)
    features = np.append(features, median, axis=1)
    features = np.append(features, std, axis=1)
    features = np.append(features, skew, axis=1)
    features = np.append(features, kurt, axis=1)
    features = np.append(features, entropy, axis=1)

    if smooth:
        np.save('less_feature_smooth', features)
    else:
        np.save('less_features', features)

    features = np.append(features, sobel_x, axis=1)
    features = np.append(features, sobel_y, axis=1)
    features = np.append(features, prewitt_x, axis=1)
    features = np.append(features, prewitt_y, axis=1)

    if smooth:
        np.save('feature_smooth', features)
    else:
        np.save('features', features)

    np.save('labels', labels)



construct_dataset(True)