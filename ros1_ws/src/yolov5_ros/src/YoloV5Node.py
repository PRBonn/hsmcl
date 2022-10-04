#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append(os.path.abspath('/home/nickybones/Code/yolo-v-5-pck/yolov5/'))
#sys.path.append(os.path.abspath('yolov5'))
#print(os.path.abspath('yolov5'))


from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.augmentations import letterbox


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import cv2

import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
from nmcl_msgs.msg import Yolo, YoloArray, YoloCombinedArray
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber


class YoloV5Node():

    def __init__(self)->None:

        cameraImgTopics = rospy.get_param('~cameraImgTopics')
        self.camIDS = rospy.get_param('~camIDS')
        self.camNum = len(self.camIDS)
        yoloTopic = rospy.get_param('yoloTopic')
        weights = rospy.get_param('~weights')
        self.conf_thres = rospy.get_param('~conf_thres')
        data = rospy.get_param('dataset')
        imgsize = rospy.get_param('~imgsize')
        #self.visualize = rospy.get_param('visualize')
      
        with open(data, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        self.semclasses = data_loaded['names']
        self.imgsz = (imgsize, imgsize)
        self.bridge = CvBridge()
        self.iou_thres=0.45
        self.max_det=1000
        self.classes=None
        self.agnostic_nms=False
        augment=False
        visualize=False
        update=False
        project=ROOT / 'runs/detect'
        name='exp'
        exist_ok=False
        line_thickness=3
        hide_labels=False
        hide_conf=False
        half=False
        dnn=False
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.stride = stride
        self.imgsz  = check_img_size(self.imgsz , s=self.stride)  # check image size
        bs = 1  # batch_size
        self.model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))  # warmup

        for i in range(self.camNum):
            cam_id = self.camIDS[i]
            if cam_id == 0:
                cam0_sub = Subscriber(cameraImgTopics[i], Image)
            elif cam_id == 1:
                cam1_sub = Subscriber(cameraImgTopics[i], Image)
            elif cam_id == 2:
                cam2_sub = Subscriber(cameraImgTopics[i], Image)
            elif cam_id == 3:
                cam3_sub = Subscriber(cameraImgTopics[i], Image)
            else:
                rospy.logerr("camID not valid!")



        self.ats = ApproximateTimeSynchronizer([cam0_sub, cam1_sub, cam2_sub, cam3_sub], queue_size=5, slop=0.05)
        self.ats.registerCallback(self.callbackCombined4)
        self.pred_pub = rospy.Publisher(yoloTopic, YoloCombinedArray, queue_size=10)
        self.pred_pub_debug = rospy.Publisher('yolov5_debug', Image, queue_size=10)

        rospy.loginfo("YOLONode ready!")



    def Process(self, img0, camID, stamp, h, w):

            img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]

            if img.shape[2] == 4:
                img = cv2.cvtColor(img0, cv2.COLOR_BGRA2BGR)

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            start = time.time()
            # Inference
            pred = self.model(img, augment=False, visualize=False)
            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            end = time.time()

            img_res = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img_res2 = np.zeros((h, w, 3)).astype(np.uint8)

            clr = cm.rainbow(np.linspace(0, 1, len(self.semclasses )))
            yoloarray = []

            # Process predictions
            for i, det in enumerate(pred):  # per image
             
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    det = det.cpu().detach().numpy()

                    for o in range(det.shape[0]):
                        objc = int(det[o, -1])
                        x1 = int(det[o, 0])
                        y1 = int(det[o, 1])
                        x2 = int(det[o, 2])
                        y2 = int(det[o, 3])
                        conf = det[o, 4]

                        color = 255 * clr[objc, :3]
                        cv2.rectangle(img_res,(x1,y1), (x2,y2), color, 2)
                        text1 = "{}".format(self.semclasses[objc])
                        text2 = "{:.2f}".format(conf)
                        cv2.putText(img_res, text1,(x1+2,y1+40),0,2.0,color, thickness = 3)
                        cv2.putText(img_res, text2,(x1+2,y1+100),0,2.0,color, thickness = 3)

                        yolo_msg = Yolo()
                        yolo_msg.semclass = objc
                        yolo_msg.confidence = conf
                        yolo_msg.xmin = x1
                        yolo_msg.ymin = y1
                        yolo_msg.xmax = x2
                        yolo_msg.ymax = y2
                        yoloarray.append(yolo_msg)


            img_res2[5:h-5, 5:w-5] = img_res[5:h-5, 5:w-5]
            yoloarray_msg = YoloArray()
            yoloarray_msg.detections = yoloarray
            yoloarray_msg.camID = camID
            yoloarray_msg.header.stamp = stamp
            
            return yoloarray_msg, img_res2


    def callbackCombined4(self, cam0_msg, cam1_msg, cam2_msg, cam3_msg):

        msgs_array = [cam0_msg, cam1_msg, cam2_msg, cam3_msg]
        h = cam0_msg.height
        w = cam0_msg.width
        debug_img = np.zeros((h, w*self.camNum, 3)).astype(np.uint8)

        combined = []

        for c in range(self.camNum):

            camID = self.camIDS[c]

            img0 = self.bridge.imgmsg_to_cv2(msgs_array[c], desired_encoding='passthrough')

            yoloarray_msg, img_res2 = self.Process(img0, camID, msgs_array[camID].header.stamp, h, w)
            combined.append(yoloarray_msg)
            debug_img[: , c*w:(c+1)*w] = img_res2

        combined_msg = YoloCombinedArray()
        combined_msg.views = combined
        combined_msg.header.stamp = rospy.get_rostime()
        self.pred_pub.publish(combined_msg)

        image_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        image_msg.header.stamp = rospy.get_rostime()
        self.pred_pub_debug.publish(image_msg)


if __name__ == "__main__":


    rospy.init_node('YoloV5Node', anonymous=True)
    #rate = rospy.Rate(10)
    posn = YoloV5Node()
    rospy.spin()