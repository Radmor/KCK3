from os import listdir
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import data, color, filters, io, feature, measure,draw
from skimage import morphology
from scipy import ndimage



data_dir_path = 'data'

# settings

thresholdValue = 0.29
morphohologySquareWidth = 5
gaussianFilterSigma = 5.5
contoursLevel = 0.99
contourLineWidth = 2.5
centerOfMassCircleRadius = 6



# zwraca listę ścieżek do plików względem tego skryptu
def list_file_paths(dir_path):
    return [os.path.join(dir_path, file) for file in listdir(dir_path)]

def input_data(imageFilePath):
    return data.imread(imageFilePath)

def transform_image_to_grey(image):
    return color.rgb2grey(image)

def average_image(image):
    minValue = np.amin(image)
    meanValue = np.mean(image)

    for x in np.nditer(image, op_flags=['readwrite']):
        if x > minValue + thresholdValue:
            x[...] = meanValue

    return image

def compute_mass_center_coords(imageShape, rowPixelCoord, columnPixelCoord):
    massCenterTempArray = np.zeros(imageShape)
    massCenterTempArray[rowPixelCoord, columnPixelCoord] = 1
    return ndimage.measurements.center_of_mass(massCenterTempArray)

def perform_image_computations(image,index):
    filteredImage = transform_image_to_grey(image)

    filteredImage = average_image(filteredImage)

    filteredImage = filteredImage > filters.threshold_otsu(filteredImage)

    filteredImage = filters.gaussian(filteredImage,sigma=gaussianFilterSigma)

    contours = measure.find_contours(filteredImage, contoursLevel)
    subplot = plt.subplot(6, 3, index + 1)
    for contour in contours:
        rowCoords = contour[:, 0]
        columnCoords = contour[:, 1]
        rowPixelCoord, columnPixelCoord = draw.polygon(rowCoords, columnCoords)

        centerOfMassCoordX, centerOfMassCoordY, temp = compute_mass_center_coords(image.shape,rowPixelCoord,columnPixelCoord)

        #centr = compute_mass_center_coords(image.shape,rowPixelCoord,columnPixelCoord)
        #print(centr)

        circlePlot = plt.Circle((centerOfMassCoordY, centerOfMassCoordX), radius=centerOfMassCircleRadius, color='white')

        subplot.add_artist(circlePlot)
        subplot.plot(columnCoords, rowCoords, linewidth=contourLineWidth)


if __name__ == "__main__":
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.25, wspace=0.10)

    for index, imageFilePath in enumerate(list_file_paths(data_dir_path)):
        image = input_data(imageFilePath)
        perform_image_computations(image, index)
        plt.imshow(image, cmap="Greys_r")
        plt.axis("off")

    plt.savefig("part_II.pdf", bbox_inches="tight")


