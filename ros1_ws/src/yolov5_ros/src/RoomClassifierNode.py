#!/usr/bin/env python3


import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import rospy
from nmcl_msgs.msg import Yolo, YoloArray, YoloCombinedArray
from std_msgs.msg import UInt16, Float32MultiArray
from message_filters import ApproximateTimeSynchronizer, Subscriber


class RoomClassifierNode():

    def __init__(self)->None:

        picklePath = rospy.get_param('picklePath')
        yoloTopic = rospy.get_param('yoloTopic')
        roomTopic = rospy.get_param('roomTopic')
        
        df = pd.read_pickle(picklePath)
        samples = df["samples"].to_numpy()
        predictions = df["predictions"].to_numpy()

        X = np.zeros((len(samples), samples[0].shape[0]))
        y = np.zeros(len(samples))

        for i in range(len(samples)):
            X[i] = samples[i]
            y[i] = predictions[i]


        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.knn = KNeighborsClassifier()
        self.knn.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'
             .format(self.knn.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'
             .format(self.knn.score(X_test, y_test)))
        self.yolo_sub = rospy.Subscriber(yoloTopic, YoloCombinedArray, self.callback)
        self.room_pub = rospy.Publisher(roomTopic, Float32MultiArray, queue_size=10)
        


    def callback(self, yolo_msg):

        detVec = np.zeros(14, dtype=int)
        for c in range(4):
            scan = yolo_msg.views[c].detections
            for obj in scan:
                semID = obj.semclass
                detVec[semID] += 1

        detVec = np.reshape(detVec, (1, -1))
        pred = self.knn.predict(detVec)
        prob = self.knn.predict_proba(detVec).flatten()
        msg = Float32MultiArray()
        msg.data = prob
        #msg.header.stamp = rospy.Time.now()
        self.room_pub.publish(msg)




if __name__ == "__main__":


    rospy.init_node('RoomClassifierNode', anonymous=True)
    #rate = rospy.Rate(10)
    posn = RoomClassifierNode()
    rospy.spin()