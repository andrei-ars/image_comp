"""
https://github.com/aleju/imgaug
https://imgaug.readthedocs.io/en/latest/source/overview/color.html
"""

#import math
import os
#import os.path
import sys
import numpy as np
import random
#from random import randint
from PIL import Image
from imgaug import augmenters as iaa
from imgaug import parameters as iap


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def resize_and_concat_images(im1, im2, image_size):
    im1 = im1.resize((image_size, image_size))
    im2 = im2.resize((image_size, image_size))
    return get_concat_h(im1, im2)



def augment_image(im):

    arr = np.asarray(im, dtype=np.uint8)
    blurer = iaa.GaussianBlur(1 + iap.Uniform(0.1, 3.0))

    seq = iaa.Sequential([
        #iaa.Crop(px=(1, 16), keep_size=False),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.ChangeColorTemperature((1100, 10000))
    ])

    #aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    #aug = iaa.EdgeDetect(alpha=(0.0, 1.0))
    #aug = iaa.ChangeColorTemperature((1100, 10000))

    aug_arr = seq(images=[arr])[0]
    im2 = Image.fromarray(aug_arr)
    #im2.show()
    #im.show()
    #Image.fromarray(np.hstack((np.array(im),np.array(im2)))).show()
    return im2


def augment_111():

    im = Image.open("1.jpg")
    print(im.size)
    print(im)
    arr = np.asarray(im, dtype=np.uint8)
    blurer = iaa.GaussianBlur(1 + iap.Uniform(0.1, 3.0))

    seq = iaa.Sequential([
        #iaa.Crop(px=(1, 16), keep_size=False),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.ChangeColorTemperature((1100, 10000))
    ])

    #aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    #aug = iaa.EdgeDetect(alpha=(0.0, 1.0))
    #aug = iaa.ChangeColorTemperature((1100, 10000))

    aug_arr = seq(images=[arr])[0]
    im2 = Image.fromarray(aug_arr)
    #im2.show()
    #im.show()
    Image.fromarray(np.hstack((np.array(im),np.array(im2)))).show()



def create_dataset(in_path, out_path):

    filenames = os.listdir(in_path)
    for i, filename1 in enumerate(filenames):
        im1 = Image.open(os.path.join(in_path, filename1))
        filenames_minus1 = filenames.copy()
        del filenames_minus1[i]
        filename2 = random.choice(filenames_minus1)
        im2 = Image.open(os.path.join(in_path, filename2))
        print(filename1, filename2)
        im = resize_and_concat_images(im1, augment_image(im1), image_size=100)
        im_path = os.path.join(os.path.join(out_path, "0"), "{}.png".format(i))
        im.save(im_path, "PNG")
        im = resize_and_concat_images(im1, augment_image(im2), image_size=100)
        im_path = os.path.join(os.path.join(out_path, "1"), "{}.png".format(i))
        im.save(im_path, "PNG")


if __name__ == '__main__':

    os.system("rm -r dataset")
    os.system("mkdir -p dataset")
    os.system("mkdir -p dataset/train")
    os.system("mkdir -p dataset/train/0")
    os.system("mkdir -p dataset/train/1")
    os.system("mkdir -p dataset/valid")
    os.system("mkdir -p dataset/valid/0")
    os.system("mkdir -p dataset/valid/1")
    create_dataset(in_path="raw_data/train", out_path="dataset/train")
    create_dataset(in_path="raw_data/valid", out_path="dataset/valid")