#!/usr/bin/env python3

from GMAP import GMAP
import cv2
import yaml
import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import get_cmap
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
import json 
from std_msgs.msg import UInt16, Float32MultiArray


class RoomCategoryVizNode():

	def __init__(self)->None:

		data = rospy.get_param('data')
		dataFolder = rospy.get_param('dataFolder')
		roomTopic = rospy.get_param('roomTopic')
		markerTopic = rospy.get_param('roomCategoryTopic')


		publisher = rospy.Publisher(markerTopic, MarkerArray)
		f = open(dataFolder + "floor.config")
		config = json.load(f)

		
		roomCategories = config['semantic']['categories']

		msg = rospy.wait_for_message(roomTopic, Float32MultiArray)
		probs = msg.data

		txt = "Initialization probabilities: "

		for i, c in enumerate(roomCategories):

			txt += "{}: {:.2f}  ".format(c, probs[i])
			

		count = 0
		MARKERS_MAX = 100
		
		while not rospy.is_shutdown():

			markerArray = MarkerArray()

			txtmarker = Marker()
			txtmarker.header.frame_id = "map"
			txtmarker.type = txtmarker.TEXT_VIEW_FACING
			txtmarker.action = txtmarker.ADD
			txtmarker.text = txt
			txtmarker.scale.x = 1.5
			txtmarker.scale.y = 1.5
			txtmarker.scale.z = 1.5
			txtmarker.color.a = 1.0
			txtmarker.color.r = 1.0
			txtmarker.color.g = 1.0
			txtmarker.color.b = 1.0
			txtmarker.pose.orientation.w = 1.0
			txtmarker.pose.position.x = 0 
			txtmarker.pose.position.y = 5
			txtmarker.pose.position.z = 0.01
			markerArray.markers.append(txtmarker)


			# Renumber the marker IDs
			id = 0
			for m in markerArray.markers:
				m.id = id
				id += 1

			# Publish the MarkerArray
			publisher.publish(markerArray)

			#count += 1

			rospy.sleep(0.01)







if __name__ == "__main__":


    rospy.init_node('RoomCategoryVizNode', anonymous=True)
    #rate = rospy.Rate(10)
    posn = RoomCategoryVizNode()
    rospy.spin()