#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
from PIL import Image
from models import NeuralNetworkModel
from create_dataset import resize_and_concat_images
from data_factory import data_transforms

IMAGE_SIZE = 100
MODEL_PATH = "./saved_model/model.pt"


def compare_images(im1, im2, model):
    im = resize_and_concat_images(im1, im2, image_size=IMAGE_SIZE)
    transform = data_transforms['valid']
    input_tensor = transform(im).float().unsqueeze(0)
    output_tensor = model(input_tensor).detach()
    prediction = F.softmax(output_tensor, dim=1).numpy()[0]
    return prediction


if __name__ == "__main__":

    model = NeuralNetworkModel(image_size=IMAGE_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    im1 = Image.open("test_images/1.png")
    im2 = Image.open("test_images/2.png")
    prediction = compare_images(im1, im2, model)
    label = np.argmax(prediction)
    print("prediction:", prediction)
    print("label:", label)
