import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt

def get_no_noise_map(images: np.array) -> np.array:
    # stack all images together
    images_stacked = np.stack(images, axis=-1)
    median_image = np.median(images_stacked, axis=-1).astype('uint8')
    no_noise_map = np.isclose(median_image[:,:,:,np.newaxis], images_stacked, atol=10, rtol=0.01)
    no_noise_map = np.all(no_noise_map, axis=(-2,-1)).astype('uint8')
    return no_noise_map

def get_median_image(images):
    """Simplest median filter from multiple images

    :param images: Array of images
    :type images: np.array
    :return: One image - result of median filter
    :rtype: 3D np.array
    """
    images_stacked = np.stack(images, axis=-1)
    median_image = np.median(images_stacked, axis=-1).astype('uint8')
    return median_image

def average_closest_pixels(pixels):
    """Find two nearest pixels among all in the pixels array and compute their average value

    :param pixels: Array of pixels
    :type pixels: np.array
    :return: Average of two the nearest to each other pixels.
    :rtype: np.array
    """
    # TODO: Optimize
    closest = np.inf
    m = np.zeros(shape=(3,))
    for i in range(len(pixels)-1):
        for j in range(i+1, len(pixels)):
            a = np.sum(np.abs(pixels[i].astype(int)-pixels[j].astype(int)))
            if a < closest:
                closest = a
                m = np.mean(np.row_stack((pixels[i], pixels[j])), axis=0)
    return m.astype(np.uint8)

def get_available_pixels(images: np.array, y: int, x: int) -> np.array:
    """Get pixels from all images at certain position

    :param images: Array of all images from dataset
    :type images: np.array
    :param y: Row coordinate
    :type y: int
    :param x: Column coordinate
    :type x: int
    :return: Array of all pixels from that position
    :rtype: np.array
    """
    return np.array([image[y][x] for image in images])

def compute_closest_to_mean(row_mean: np.array, column_mean: np.array,pixels: np.array) -> np.array:
    """Get pixel closest to the row and column color mean.

    :param row_mean: Color mean from row where pixel exists
    :type row_mean: np.array
    :param column_mean: Color mean from column where pixel exists
    :type column_mean: np.array
    :param pixels: Array of pixels from all images on the same position
    :type pixels: np.array
    :return: Closest pixel
    :rtype: np.array
    """
    # TODO: Optimize
    m = np.inf
    w = pixels[0]
    for pixel in pixels:
        diff = np.sum(np.abs(row_mean.astype(int)-pixel.astype(int)))+np.sum(np.abs(column_mean.astype(int)-pixel.astype(int)))
        if diff<m:
            m = diff
            w = pixel
    return w.astype(np.uint8)

def detect_outlier(pixel) -> bool:
    # TODO think about lower boundary
    """Detect outlier pixel. A pixel is an outlier if any of the rgb channel is greater

    :param pixel: Single pixel array
    :type pixel: _type_
    :return: True if pixel is an outlier. Otherwise False
    :rtype: bool
    """
    pixel = np.sort(pixel)[::-1]
    if pixel[0] > pixel[1] + 30:
        return True
    return False
    
    
def clean_background(images: np.array) -> np.array:
    """Main function for cleaning background

    :param images: Array of all images
    :type images: np.array
    :return: Single output image
    :rtype: np.array
    """
    height, width = images[0].shape[0], images[0].shape[1]
    # First apply simple median filter.
    images_median = get_median_image(images)
    
    # First mask (larger one)
    no_noise_map = get_no_noise_map(images) * 255
    struct = np.ones((3,3), np.uint8)

    eroded = cv2.dilate(no_noise_map/255, struct, iterations=3)
    img_closed = cv2.erode(eroded, struct, iterations=6) * 255
    
    
    # Generate another image
    avg_images_median = images_median.copy()
    # TODO: Optimize to not use loops
    for y in range(height):
        for x in range(width):
            if img_closed[y][x] == 0.0:
                # TODO: Change get_available_pixels function
                pixels = get_available_pixels(images, y, x)
                avg_images_median[y][x] = average_closest_pixels(pixels)
                
    # Create another mask (smaller one) based on median image and average median image
    noise_map = get_no_noise_map(np.array([images_median, avg_images_median])) * 255
    struct = np.ones((1,1), np.uint8)

    eroded = cv2.dilate(noise_map/255, struct, iterations=3)
    new_diff = cv2.erode(eroded, struct, iterations=6) * 255
    
    # Final cleaning output image from as much noise as possible
    output = avg_images_median.copy()
    # TODO: Optimize not use loops
    for y in range(height):
        for x in range(width):
            if new_diff[y][x] == 0.0 and (np.all(avg_images_median[y][x]<65.0) or detect_outlier(avg_images_median[y][x])):
                row_mean = np.mean(avg_images_median[y], axis=0)
                column_mean = np.mean(avg_images_median[:, x, :], axis=0)
                output[y][x] = compute_closest_to_mean(row_mean, column_mean, get_available_pixels(images, y, x))
    return output
