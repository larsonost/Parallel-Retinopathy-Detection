import os
import cv2 as cv
import argparse

def main(args):

    file_list = ["".join([args.input_dir, "/", ii]) for ii in os.listdir(args.input_dir)]

    for fl in file_list:
        img = cv.imread(fl)
        if img.shape[0] != img.shape[1]:
            os.remove(fl)

if __name__ == "__main__":
    
    p = argparse.ArgumentParser(description = 'Parallel Image Filter', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input_dir', type=str, help='Input Directory containing Images')
    input_arguments = p.parse_args()
    
    main(input_arguments)