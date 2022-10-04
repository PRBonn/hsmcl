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

import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import yaml
from nmcl_msgs.msg import TextArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class TextVisualizerNode():

    def __init__(self)->None:

        cameraImgTopics = rospy.get_param('~cameraImgTopics')
        self.camIDS = rospy.get_param('~camIDS')
        self.camNum = len(self.camIDS)
        textTopic = rospy.get_param('textTopic')

        self.bridge = CvBridge()

        self.img0 = None
        self.img1 = None
        self.img2 = None
        self.img3 = None
      
        for i in range(self.camNum):
            cam_id = self.camIDS[i]
            if cam_id == 0:
                self.cam0_sub = rospy.Subscriber(cameraImgTopics[i], Image, self.callback0)
                self.cam0_pub = rospy.Publisher('textDebug0', Image, queue_size=10)
            elif cam_id == 1:
                self.cam1_sub = rospy.Subscriber(cameraImgTopics[i], Image, self.callback1)
                self.cam1_pub = rospy.Publisher('textDebug1', Image, queue_size=10)
            elif cam_id == 2:
                self.cam2_sub = rospy.Subscriber(cameraImgTopics[i], Image, self.callback2)
                self.cam2_pub = rospy.Publisher('textDebug2', Image, queue_size=10)
            elif cam_id == 3:
                self.cam3_sub = rospy.Subscriber(cameraImgTopics[i], Image, self.callback3)
                self.cam3_pub = rospy.Publisher('textDebug3', Image, queue_size=10)
            else:
                rospy.logerr("camID not valid!")


        #self.text_sub = rospy.Subscriber(textTopic, TextArray, self.callbackText)
        self.text_sub = rospy.Subscriber("confirmedDetection", TextArray, self.callbackTextConfirmed)
        self.text_pub = rospy.Publisher("textDetect", MarkerArray)
        
        rospy.loginfo("TextVisualizerNode ready!")


    def callback0(self, cam_msg):

        img = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='passthrough')
        self.img0 = img
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header.stamp = cam_msg.header.stamp
        self.cam0_pub.publish(image_msg)

    def callback1(self, cam_msg):

        img = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='passthrough')
        self.img1 = img
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header.stamp = cam_msg.header.stamp
        self.cam1_pub.publish(image_msg)

    def callback2(self, cam_msg):

        img = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='passthrough')
        self.img2 = img
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header.stamp = cam_msg.header.stamp
        self.cam2_pub.publish(image_msg)

    def callback3(self, cam_msg):

        img = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='passthrough')
        self.img3 = img
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        image_msg.header.stamp = cam_msg.header.stamp
        self.cam3_pub.publish(image_msg)


    def callbackTextConfirmed(self, text_msg):

        cam_id = text_msg.id
        text = text_msg.text
        num_detect = len(text)

        markerArray = MarkerArray()

        for c in range(num_detect):

            color = [0, 0, 0]

            txtmarker = Marker()
            txtmarker.header.frame_id = "map"
            txtmarker.type = txtmarker.TEXT_VIEW_FACING
            txtmarker.action = txtmarker.ADD
            txtmarker.text = text[c]
            txtmarker.scale.x = 1.0
            txtmarker.scale.y = 1.0
            txtmarker.scale.z = 1.0
            txtmarker.color.a = 1.0
            txtmarker.color.r = color[2]
            txtmarker.color.g = color[1]
            txtmarker.color.b = color[0]
            txtmarker.pose.orientation.w = 1.0
            txtmarker.pose.position.x = -18 
            txtmarker.pose.position.y =  - c * 1.2 
            txtmarker.pose.position.z = 0.01
            markerArray.markers.append(txtmarker)

        # Renumber the marker IDs
        id = 0
        for m in markerArray.markers:
            m.id = id
            id += 1

        # Publish the MarkerArray
        self.text_pub.publish(markerArray)


    def callbackText(self, text_msg):

        cam_id = text_msg.id
        text = text_msg.text
        points = text_msg.contours
        num_detect = int(len(points) / 4)

        contours = []
        for c in range(num_detect):
            cont = np.empty(shape=[0, 2], dtype=int)
            text_tag = text[c]
            found = ('room' in text_tag) or ('Room' in text_tag)
            #print(text_tag, found)
            if found:
                for p in range(4):
                    pnt = points[4 * c + p]
                    cont = np.append(cont, [[int(pnt.x), int(pnt.y)]], axis=0)
                contours.append(cont)

        if len(contours):

            if cam_id == 0:
                cv2.drawContours(image=self.img0, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                image_msg = self.bridge.cv2_to_imgmsg(self.img0, encoding='bgr8')
                image_msg.header.stamp = rospy.get_rostime()
                self.cam0_pub.publish(image_msg)
            elif cam_id == 1:
                cv2.drawContours(image=self.img1, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                image_msg = self.bridge.cv2_to_imgmsg(self.img1, encoding='bgr8')
                image_msg.header.stamp = rospy.get_rostime()
                self.cam1_pub.publish(image_msg)
            elif cam_id == 2:
                cv2.drawContours(image=self.img2, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                image_msg = self.bridge.cv2_to_imgmsg(self.img2, encoding='bgr8')
                image_msg.header.stamp = rospy.get_rostime()
                self.cam2_pub.publish(image_msg)
            elif cam_id == 3:
                cv2.drawContours(image=self.img3, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                image_msg = self.bridge.cv2_to_imgmsg(self.img3, encoding='bgr8')
                image_msg.header.stamp = rospy.get_rostime()
                self.cam3_pub.publish(image_msg)


            markerArray = MarkerArray()

            for c in range(num_detect):

                color = [0, 0, 0]

                txtmarker = Marker()
                txtmarker.header.frame_id = "map"
                txtmarker.type = txtmarker.TEXT_VIEW_FACING
                txtmarker.action = txtmarker.ADD
                txtmarker.text = text[c]
                txtmarker.scale.x = 1.0
                txtmarker.scale.y = 1.0
                txtmarker.scale.z = 1.0
                txtmarker.color.a = 1.0
                txtmarker.color.r = color[2]
                txtmarker.color.g = color[1]
                txtmarker.color.b = color[0]
                txtmarker.pose.orientation.w = 1.0
                txtmarker.pose.position.x = -18 
                txtmarker.pose.position.y =  - c * 1.2 
                txtmarker.pose.position.z = 0.01
                markerArray.markers.append(txtmarker)

            # Renumber the marker IDs
            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            # Publish the MarkerArray
            self.text_pub.publish(markerArray)

            #count += 1

            #rospy.sleep(0.01)




if __name__ == "__main__":


    rospy.init_node('TextVisualizerNode', anonymous=True)
    #rate = rospy.Rate(10)
    posn = TextVisualizerNode()
    rospy.spin()