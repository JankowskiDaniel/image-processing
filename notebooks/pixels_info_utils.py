import cv2
import matplotlib.pyplot as plt
import numpy as np


from pandas import DataFrame
import pandas as pd
from IPython.display import display, HTML
from skimage.exposure import rescale_intensity
import plotly.graph_objects as go
import pandas as pd

def get_available_pixels(generated: np.array, images: np.array, y: int, x: int, _print: bool = False) -> np.array:
    pixels = np.array([image[y][x] for image in images])
    if _print:
        print("Generated has value: ", generated[y][x])
        for i, _ in enumerate(images):
            print(f"Image {i}", _[y][x])
    return pixels

def get_2d_channel_average(pixels: np.array) -> np.array:
    return np.mean(pixels, axis=0)

def get_2d_channel_median(pixels: np.array) -> np.array:
    return np.median(pixels, axis=0)

def get_3d_channel_average(image: np.array) -> np.array:
    return np.mean(image, axis=(0, 1))

def plot_channels_linechart(data: np.array, if_row: bool = True):
    if if_row:
        y = np.arange(752)
    else:
        y = np.arange(500)
    r = data[:, 2]
    g = data[:, 1]
    b = data[:, 0]
    
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(5)
    plt.plot(y, r, color="red")
    plt.plot(y, g ,color="green")
    plt.plot(y, b, color="blue")
    plt.show()
    
def pixel_info(images: np.array, y: int, x: int, generated: np.array = None, _print: bool = False):
    pixels = get_available_pixels(generated, images, y, x, _print)
    mean = get_2d_channel_average(pixels)
    median = get_2d_channel_median(pixels)
    if _print:
        print("Mean: (B,G,R) ", mean)
        print("Median: (B,G,R", median)
        
    return pixels, mean, median

def get_proper_pixels_from_result(result, y: int, x: int) -> np.array:
    return result[y][x]