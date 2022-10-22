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
from torchvision import ops
 
def get_iou(ground_truth, pred):
      ix1 = torch.max(ground_truth[0][0], pred[0][0])
      iy1 = torch.max(ground_truth[0][1], pred[0][1])
      ix2 = torch.min(ground_truth[0][2], pred[0][2])
      iy2 = torch.min(ground_truth[0][3], pred[0][3])
      i_height = torch.max(iy2 - iy1 + 1, torch.tensor(0.))
      i_width = torch.max(ix2 - ix1 + 1, torch.tensor(0.))

      area_of_intersection = i_height * i_width
      gt_height = ground_truth[0][3] - ground_truth[0][1] + 1

      gt_width = ground_truth[0][2] - ground_truth[0][0] + 1


      pd_height = pred[0][3] - pred[0][1] + 1

      pd_width = pred[0][2] - pred[0][0] + 1
      area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

      iou = area_of_intersection / area_of_union
      return iou

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
          cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (255,0,0), 2)
          cv2.rectangle(img_with_dets, (140,123), (402,470), (0,255,0), 2)
          print( x , y , x+width , y+height)
          cv2.circle(img_with_dets, (keypoints['left_eye']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['right_eye']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['nose']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['mouth_left']), 2, (0,155,255), 2)
          cv2.circle(img_with_dets, (keypoints['mouth_right']), 2, (0,155,255), 2)
          plt.figure(figsize = (10,10))
          plt. imshow(img_with_dets)
          plt.axis('off')          
          ground_truth_bbox = torch.tensor([[140, 123, 402, 470]], dtype=torch.float)
          prediction_bbox = torch.tensor([[x, y, x+width, y+height]], dtype=torch.float)
          print(prediction_bbox)
          iou = get_iou(ground_truth_bbox , prediction_bbox)
          print('iou result is :' ,  iou)  

          


cartoondataset= dataset_class('/content/drive/MyDrive/cartoondataset/training/face/')
cartoondataset.__test__()
