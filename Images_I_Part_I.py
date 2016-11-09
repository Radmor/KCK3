from os import listdir
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import data, color, filters, io, feature, measure,draw
from skimage import morphology
from scipy import ndimage



data_dir_path = 'data'

# settings

# zwraca listę ścieżek do plików względem tego skryptu
def list_file_paths(dir_path):
    return [os.path.join(dir_path, file) for file in ['samolot01.jpg','samolot07.jpg','samolot08.jpg','samolot09.jpg','samolot10.jpg','samolot11.jpg']]

def input_data(imageFilePath):
    return data.imread(imageFilePath,as_grey=True)

def perform_image_computations(filteredImage,index):

    filteredImage = filteredImage > filters.threshold_otsu(filteredImage)

    filteredImage = filters.sobel(filteredImage)

    subplot = plt.subplot(2, 3, index + 1)

    return filteredImage


if __name__ == "__main__":
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0.25, wspace=0.10)

    for index, imageFilePath in enumerate(list_file_paths(data_dir_path)):
        image = input_data(imageFilePath)
        image = perform_image_computations(image, index)
        plt.imshow(image, cmap="Greys_r")
        plt.axis("off")

    plt.savefig("part_I.pdf", bbox_inches="tight")


