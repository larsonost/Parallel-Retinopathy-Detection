import cv2 as cv
import numpy as np
import pyspark
import argparse
import os
import math

def get_original_bounds(opencv_img: np.array):
    
    mean_img = np.mean(opencv_img, 2)
    r = math.floor(opencv_img.shape[0]/2)
    nz = np.nonzero(mean_img[:, r])
    return nz[0][0]

def transform(opencv_img: np.array):
        
        return cv.addWeighted(opencv_img, 4, cv.GaussianBlur(opencv_img, (0,0), 10), -4, 128)

def crop_circle(opencv_img: np.array, bounds: int):

    if opencv_img.shape[0] != opencv_img.shape[1]:
        raise Exception(f"Image was not a square")

    r = opencv_img.shape[0]
    r0 = opencv_img.shape[0]//2
    mask_init = np.zeros_like(opencv_img)
    mask = cv.circle(mask_init, (r0, r0), r0, (1, 1, 1), -1)
    
    cropped_img = opencv_img*mask
    cropped_img[:(bounds+10), :, :] = 0
    cropped_img[(r-bounds-10):, :, :] = 0

    return cropped_img

def filepath_only(path):

    return path.split("/")[-1]

class main:
    
    def __init__(self, args) -> None:
        
        self.inputs = args
        self.validate_args()
        self.sc = pyspark.SparkContext(appName="Parallel Filter Function", master=f"local[{args.ncores}]")
        self.load_files()
        self.parallel_transform()
        self.write_transformed()
        self.summarize()

    def validate_args(self):
        if not os.path.isdir(self.inputs.input_dir):
            raise Exception(f"Input directory ({self.inputs.input_dir}) not found.")
        
        if not self.inputs.input_dir.endswith("/"):
            self.inputs.input_dir += "/"

        if not self.inputs.output_dir.endswith("/"):
            self.inputs.output_dir += "/"
        
        if not os.path.isdir(self.inputs.output_dir):
            print(f"Output folder {self.inputs.output_dir} not found, creating it in the runtime directory")
            os.mkdir(self.inputs.output_dir)

        self.input = self.inputs.input_dir
        self.output = self.inputs.output_dir

        return self

    def load_files(self):
        
        file_list = ["".join([self.input, imgfile]) for imgfile in os.listdir(self.input)]
        
        self.n_images = len(file_list)
        self.files = self.sc.parallelize(file_list)

        return self

    def parallel_transform(self):
        
        self.images = self.files.map(lambda filepath: (filepath, cv.imread(filepath)))\
            .map(lambda tup: (tup[0], get_original_bounds(tup[-1]), tup[-1]))\
            .map(lambda tup: (tup[0], tup[1], transform(tup[-1])))\
            .map(lambda tup: (tup[0], crop_circle(tup[-1], tup[1])))
        
        return self

    def write_transformed(self):
        
        outp = self.output
        self.succ_or_fail = self.images\
            .map(lambda tup: cv.imwrite("".join([outp, "/", filepath_only(tup[0])]), tup[1]))\
            .collect()
    
        return self

    def summarize(self):

        print(f"Transformed {sum(self.succ_or_fail)} successfully out of {self.n_images}")

if __name__ == "__main__":
    
    p = argparse.ArgumentParser(description = 'Parallel Image Filter', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument('input_dir', type=str, help='Input Directory containing Images')
    p.add_argument('output_dir', type=str, help="Output Directory")
    p.add_argument('--ncores', type=int, default = 10, help='Number of cores to run in Parallel')
    
    input_arguments = p.parse_args()
    # input_arguments = p.parse_args(["./01_DataIn/test_r/", "./transformed_images/", "--ncores", "1"])
    main(input_arguments)

    # img = transform(cv.imread("01_DataIn/test_r/10005_left.jpeg"))
    # bnds = get_original_bounds(img)
    # bounds = crop_circle(img, bnds)
