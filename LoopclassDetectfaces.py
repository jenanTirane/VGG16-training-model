import glob as glob
from torchvision import datasets
import os 
from sys import path_hooks
from mtcnn import MTCNN
import torch
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as model



class dataset_class():


    def __init__(self,path):
        self.path=path

    def __test__(self):
      relevant_path = self.path
      included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
      file_names = [fn for fn in os.listdir(relevant_path)
        if any(fn.endswith(ext) for ext in included_extensions)]
      for x in file_names:
         x = relevant_path + x 
         input_image = Image.open(x)       
         img = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
         detector = MTCNN()
         detections = detector.detect_faces(img)
         detections
         img_with_dets = img.copy()
         min_conf = 0.9
      for det in detections:
        if det['confidence'] >= min_conf:
          x, y, width, height = det['box']
          keypoints = det['keypoints']
          cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0,155,255), 2)
          plt.figure(figsize = (10,10))
          plt. imshow(img_with_dets)
          plt.axis('off')
          
cartoondataset= dataset_class('/content/drive/MyDrive/cartoondataset/training/face/')
cartoondataset.__test__()

