/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: TextRecoNode.cpp                                                       #
# ##############################################################################
**/


#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <nmcl_msgs/TextArray.h>
#include "opencv2/opencv.hpp"
#include <opencv2/dnn/dnn.hpp>
#include "TextSpotting.h"
#include <geometry_msgs/Point.h>


// rosrun image_transport republish compressed in:=/camera/image_raw raw out:=/camera/image_raw


#define DEBUG

class TextRecoNode
{
public:

	TextRecoNode(std::string name)
	{
		ros::NodeHandle nh(name);

		std::string jsonPath;
	    std::string textTopic;

	    std::vector<int> camIDS;  
	    std::vector<std::string> cameraImgTopics;  

	    nh.getParam("textSpottingConfig", jsonPath);
	    nh.getParam("cameraImgTopics", cameraImgTopics);
	    nh.getParam("camIDS", camIDS);
	    nh.getParam("textTopic", textTopic);

    	TextSpotting ts = TextSpotting(jsonPath);
    	o_textSpotter = std::make_shared<TextSpotting>(ts);
	    o_textPub = nh.advertise<nmcl_msgs::TextArray>(textTopic, 10);

    	for (int i = 0; i < camIDS.size(); ++i)
    	{
    		int id = camIDS[i];
    		//std::cout << cameraImgTopics[i] << std::endl;
    		switch (id)
    		{
    			case 0:
    				o_camSub0 = nh.subscribe(cameraImgTopics[i], 1, &TextRecoNode::callback0, this);

    				break;
    			case 1:
    				o_camSub1 = nh.subscribe(cameraImgTopics[i], 1, &TextRecoNode::callback1, this);

    				break;
    			case 2:
    				o_camSub2 = nh.subscribe(cameraImgTopics[i], 1, &TextRecoNode::callback2, this);

    				break;
    			case 3:
    				o_camSub3 = nh.subscribe(cameraImgTopics[i], 1, &TextRecoNode::callback3, this);

    				break;
    			default: 
    				ROS_INFO_STREAM("Unsupported camera ID");   
    				break;
    		}
    	}
	}

	~TextRecoNode()
  	{

  	}

	void callback0(const sensor_msgs::ImageConstPtr& imgMsg)
	{
		callback(imgMsg, 0);
	}

	void callback1(const sensor_msgs::ImageConstPtr& imgMsg)
	{
		callback(imgMsg, 1);
	}

	void callback2(const sensor_msgs::ImageConstPtr& imgMsg)
	{
		callback(imgMsg, 2);
	}

	void callback3(const sensor_msgs::ImageConstPtr& imgMsg)
	{
		callback(imgMsg, 3); 
	}

	void callback(const sensor_msgs::ImageConstPtr& imgMsg, int camID)
	{
		cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(imgMsg, sensor_msgs::image_encodings::TYPE_8UC3);
		cv::Mat frame = cvPtr->image;

		try
		{
			nmcl_msgs::TextArray msg;

#ifdef DEBUG
			std::vector< std::vector<cv::Point>> contours;
			std::vector<std::string> recRes = o_textSpotter->InferDebug(frame, contours);
			std::vector<geometry_msgs::Point> points;
			for (int i = 0; i < contours.size(); ++i)
			{
				for (int j = 0; j < contours[i].size(); ++j)
				{
					geometry_msgs::Point p;
					p.x = contours[i][j].x;
					p.y = contours[i][j].y;
					p.z = 0;
					points.push_back(p);
					//std::cout << p << std::endl;
				}
			}

			msg.contours = points;
#else
			std::vector<std::string> recRes = o_textSpotter->Infer(frame);
#endif			
			
			msg.header = imgMsg->header;
			msg.text =  recRes;
			msg.id = camID;
			o_textPub.publish(msg);
			//ROS_INFO_STREAM(std::string("Text from camera ") + std::to_string(camID));
		}
		catch (...) 
		{
  			ROS_INFO_STREAM(std::string("Failed to infer text from camera ") + std::to_string(camID));
		}
	
	}


private:

	std::shared_ptr<TextSpotting> o_textSpotter;
	ros::Publisher o_textPub;
	ros::Subscriber o_camSub0;
	ros::Subscriber o_camSub1;
	ros::Subscriber o_camSub2;
	ros::Subscriber o_camSub3;

};


int main(int argc, char** argv)
{
	std::string name = argv[1];
	ros::init(argc, argv, name);
	TextRecoNode tr = TextRecoNode(name);
	ros::spin();
	

	return 0;
}
