import cv2
import matplotlib.pyplot as plt
import numpy as np


from pandas import DataFrame
import pandas as pd
from IPython.display import display, HTML
from skimage.exposure import rescale_intensity
import plotly.graph_objects as go
import pandas as pd


def get_median_image(images):
    # stack all images together
    images_stacked = np.stack(images, axis=-1)
    median_image = np.median(images_stacked, axis=-1).astype('uint8')
    return median_image

def get_no_noise_map(images):
    # stack all images together
    images_stacked = np.stack(images, axis=-1)
    median_image = np.median(images_stacked, axis=-1).astype('uint8')
    no_noise_map = np.isclose(median_image[:,:,:,np.newaxis], images_stacked, atol=10, rtol=0.01)
    no_noise_map = np.all(no_noise_map, axis=(-2,-1)).astype('uint8')
    return no_noise_map


def Automedian(img_space, struct):
    img_space = img_space.astype('uint8')
    # first part same as close
    img_space_open = cv2.morphologyEx(img_space, cv2.MORPH_OPEN, struct)
    img_space_close = cv2.morphologyEx(img_space_open, cv2.MORPH_CLOSE, struct)
    img_space_open2 = cv2.morphologyEx(img_space_close, cv2.MORPH_OPEN, struct)
    # img_space_G = np.maximum(img_space, img_space_open2)

    img_space_close = cv2.morphologyEx(img_space, cv2.MORPH_CLOSE, struct)
    img_space_open = cv2.morphologyEx(img_space_close, cv2.MORPH_OPEN, struct)
    img_space_close2 = cv2.morphologyEx(img_space_open, cv2.MORPH_CLOSE, struct)
    img_space_Q = np.minimum(img_space, img_space_close2)

    return np.maximum(img_space_open2, img_space_Q)


def get_images_averaged(images):
    # stack all images together
    images_stacked = np.stack(images, axis=-1)
    # print(images_stacked[0,-1])
    median_image = np.mean(images_stacked, axis=-1).astype('uint8')
    return median_image


def add_images(image1, image2, weights):
    print(weights)
    norm = np.linalg.norm(weights)
    weights = weights/norm
    print(weights)
    image_sum = image1 * weights[0] + image2 * weights[1]
    print(image_sum.shape)
    
    
    return image_sum.astype('uint8')


def get_histogram(images, bins=256, density=True, print_=False):
    if type(images) == list:
        images_stacked = np.stack(images, axis=-1)
    else:
        images_stacked = images
    histogram = []
    color = ('b','g','r')
    if print_:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    for i,col in enumerate(color):
        array_flat = images_stacked[:,:,i,:].flatten()
        histr, bins = np.histogram(array_flat, bins=bins, density=density)
        histogram.append(histr)
        if print_:
            ax[i].hist(array_flat, bins=bins, color=col, density=density)

    if print_:    
        plt.show()
    return np.array(histogram)


def get_hist_of_certain_pixels(image, mask, bins=256):
    # flattening the arrays to mask 

    image_flat = image.reshape(-1, image.shape[-1])
    mask_flat = mask.flatten()
    image_filtered = image_flat[mask_flat.astype(bool)]
    
    print(image_filtered.shape)
    histogram = []
    color = ('b','g','r')
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    for i,col in enumerate(color):
        array_flat = image_filtered[:,i].flatten()
        histr, bins = np.histogram(array_flat, bins=bins)
        histogram.append(histr)
        ax[i].hist(array_flat, bins=bins, color=col)
        
    plt.show()
    return np.array(histogram)


def get_maximum_likely_image(images: list, bins=256, power=1, deterministic=False, masked_histogram=False, mask=None) -> np.ndarray:
    """_summary_

    Args:
        images (np.ndarray): The array of input images
        bins (int, optional): Number of bins for the histogram. Defaults to 256.
        power (int, optional): Regulates the discriminative power of ... . Defaults to 1.
        deterministic (bool, optional): When True - then use deterministic assignment of pixel values. Defaults to False.
        masked_histogram (bool, optional): When True then creates the histogram based on pixels that have almost equal values on each image. Defaults to False.
        mask (_type_, optional): ...  . Defaults to None.

    Returns:
        np.ndarray: Maximum Likely image
    """
    bin_span = int(256/bins)

    if masked_histogram:
        median_image = get_median_image(images)
        histogram = get_hist_of_certain_pixels(median_image, mask, bins)
    else:
        histogram = get_histogram(images, bins)
    images_stacked = np.stack(images, axis=-1)
    # for each color value in images_stacked prescribe the probability(frequency) of a given value
    # appearin in the images
    img_out = np.zeros(shape=images_stacked.shape[:-1])
    frequencies = np.zeros(shape=images_stacked.shape)
    shape = images_stacked.shape
    for row in range(shape[0]):
        for column in range(shape[1]):
            for color in range(shape[2]):
                for sample in range(shape[3]):
                    frequencies[row,column, color, sample] = histogram[color, images_stacked[row,column, color, sample]// bin_span]
                
                if deterministic:
                    img_out[row, column, color] = images_stacked[row,column,color, np.argmax(frequencies[row,column,color])]
                else:
                # chose the pixel rancomly, based on color frequencies
                    probabilities = frequencies[row,column, color]**power/ np.sum(frequencies[row,column, color]**power)
                    img_out[row, column, color] = np.random.choice(images_stacked[row,column, color], size=1, replace=False, p=probabilities)     
    return img_out.astype('uint8')


def get_image_masked(image, mask, reversed=False):
    image_cpy = image.copy()
    if reversed:
        image_cpy[mask.astype(bool)] = [0,0,0]
    else:
        image_cpy[~mask.astype(bool)] = [0,0,0]
    return image_cpy


def get_clean_image(images: list, bins=256, deterministic=False, histogramMask=None) -> np.ndarray:
    """

    Args:
        images (list): _description_
        bins (int, optional): _description_. Defaults to 256.
        deterministic (bool, optional): _description_. Defaults to False.
        histogramMask (_type_, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: _description_
    """
    bin_span = int(256/bins)

    if histogramMask is not None:
        raise NotImplementedError()
    else:
        histogram = get_histogram(images, bins, density=True)

    
