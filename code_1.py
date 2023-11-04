from PIL import Image
import os
import cv2
image_path = 'path_to_your_image.jpg'
img = cv2.imread("Dataset\\train\\fox-resize-300\\00000245_300resized.png")
v, buffer = cv2.imencode(".png", img)