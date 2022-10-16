import os
import PIL.Image
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm.notebook import tqdm
import gradio as gr
import cv2 as cv
from face_detect import get_face


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


categories = {0 : 'Closed', 1 : 'Open', 2 : 'yawn', 3:  'no_yawn'}


def transform_images(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((120, 120)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Grayscale(num_output_channels=1),
    ])
    return transform(img)

PATH = "model_scripted.pt"

model = torch.jit.load(PATH)
model.eval()


def classify_image(img):
    image = transform_images(img).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return categories[int(predicted[0])]



out = get_face('715.jpg')
if len(out) > 0:
    cv.imwrite('out.jpg', out)
    image = PIL.Image.open('out.jpg')
    label = classify_image(image)
    print(label)



image = gr.inputs.Image(shape=(30,30))
label = gr.outputs.Label()
examples = ['_72.jpg', '_77.jpg', '115.jpg', '119.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)