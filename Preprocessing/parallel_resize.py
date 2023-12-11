import pyspark
import cv2 as cv
import os
import argparse
import numpy as np
import math
import time

def fullpath(higher: str, current: str):
    return "".join([higher, current])

def crop_to_central_bounds(img, name:str):

    if img is None: return None

    # get width and height of the image in number of pixels
    im_width = img.shape[0]
    im_ht = img.shape[1]

    # take the mean of the image in the color direction
    colorless_image = np.mean(img, 2)
    
    # set a threshold for comparison 
    threshold = 8

    # crop the image to its central bounds (this assumes that the eyes are approximately centered)
    x_keep = [ids for ids, pix in enumerate(colorless_image[:, math.floor(im_ht/2)]) if pix > threshold]
    y_keep = [ids for ids, pix in enumerate(colorless_image[math.floor(im_width/2),:]) if pix > threshold]
    
    while len(x_keep) == 0 or len(y_keep) == 0:
        threshold -= 1
        x_keep = [ids for ids, pix in enumerate(colorless_image[:, math.floor(im_ht/2)]) if pix > threshold]
        y_keep = [ids for ids, pix in enumerate(colorless_image[math.floor(im_width/2),:]) if pix > threshold]

    cropped_img = img[x_keep[0]:x_keep[-1], y_keep[0]:y_keep[-1], :]
    
    if len(x_keep) < len(y_keep):
        extension = math.floor(abs(len(x_keep) - len(y_keep))/2)        
        append = np.zeros((extension, cropped_img.shape[1], 3), dtype=img.dtype)
        cropped_img = np.concatenate((append, cropped_img, append), 0)

    return cropped_img

def resize_image_if_applicable(img: np.array, desired_size: tuple):
    im_dimensions = (img.shape[0], img.shape[1])

    actual_size = [0, 0]
    for ii in range(2):
        actual_size[ii] = min(desired_size[ii], im_dimensions[ii])
        
    return cv.resize(img, actual_size)

def main_parallel(args):

    t0 = time.time()

    # if the output directory doesn't exist yet, make it
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not args.input_dir.endswith("/"):
        args.input_dir += "/"

    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    # support parameters
    file_list = os.listdir(args.input_dir)
    max_img_size = (args.size, args.size)
    n_images = len(file_list)

    # initialize pyspark
    sc = pyspark.SparkContext(appName="image_resizer", master=f"local[{args.ncores}]")
    
    # distribute the file list
    file_list = sc.parallelize(file_list)
    
    # perform parallel unit ops in sequence.
    images = file_list\
        .map(lambda img_name: (cv.imread( fullpath(args.input_dir, img_name) ), fullpath(args.output_dir, img_name)))\
        .map(lambda image: (crop_to_central_bounds(image[0], image[1]), image[1]))\
        .map(lambda image: (resize_image_if_applicable(image[0], max_img_size), image[1]))\
        .map(lambda image: cv.imwrite(image[1], image[0]))\
        .collect()
    
    print(f"{sum(images)} successful resizes out of {n_images} in {time.time() - t0} seconds")

if __name__ == "__main__":
    
    #parser
    p = argparse.ArgumentParser(description = 'Image Resizer', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input_dir', type=str, help='Input Directory containing Images')
    p.add_argument('output_dir', type=str, help="Output Directory")
    p.add_argument('size', type=int, help='Image Size (square with this dimension)')
    p.add_argument('--ncores', type=int, default = 10, help='Number of cores to run in Parallel')
    
    input_arguments = p.parse_args()
    # input_arguments = p.parse_args(["01_DataIn/test","01_DataIn/test_r","256"])
    
    #run inputs!
    main_parallel(input_arguments)