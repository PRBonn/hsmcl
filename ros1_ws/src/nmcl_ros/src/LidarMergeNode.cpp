/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: LidarMergeNode.cpp                                                    #
# ##############################################################################
**/

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <nmcl_msgs/MergedLaserScan.h>

#include "Utils.h"
#include "Lidar2D.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::LaserScan, sensor_msgs::LaserScan> LidarSyncPolicy;

class LidarMergeNode
{
public:
	LidarMergeNode()
	{
		ros::NodeHandle nh;
		int numParticles;

		std::string configFolder;
		std::string scanFrontTopic;
		std::string scanRearTopic;
		std::string mergedScanTopic;

		nh.getParam("configFolder", configFolder);
		nh.getParam("scanFrontTopic", scanFrontTopic);  
		nh.getParam("scanRearTopic", scanRearTopic); 
		nh.getParam("mergedScanTopic", mergedScanTopic); 

		std::string fldrName =  "front_laser";      
		std::string rldrName =  "rear_laser";  

		o_l2d_f = std::make_shared<Lidar2D>(Lidar2D(fldrName, configFolder));
		o_l2d_r = std::make_shared<Lidar2D>(Lidar2D(rldrName, configFolder));

		o_mergePub = nh.advertise<nmcl_msgs::MergedLaserScan>(mergedScanTopic, 10);

		o_laserFrontSub = std::make_shared<message_filters::Subscriber<sensor_msgs::LaserScan>>(nh, scanFrontTopic, 100);
		o_laserRearSub = std::make_shared<message_filters::Subscriber<sensor_msgs::LaserScan>>(nh, scanRearTopic, 100);

		o_lidarSync = std::make_shared<message_filters::Synchronizer<LidarSyncPolicy>>(LidarSyncPolicy(10), *o_laserFrontSub, *o_laserRearSub);

		o_lidarSync->registerCallback(boost::bind(&LidarMergeNode::callback, this, _1, _2));  
	}

	void callback(const sensor_msgs::LaserScanConstPtr& laserFront, const sensor_msgs::LaserScanConstPtr& laserRear)
	{
		std::vector<float> scanFront = laserFront->ranges;
		std::vector<float> scanRear = laserRear->ranges;

		std::vector<Eigen::Vector3f> points_3d = MergeScans(scanFront, *o_l2d_f, scanRear, *o_l2d_r, 1.0, 100.0);
		int len = points_3d.size();
		std::vector<float> xy(2 * len);

		for (int i = 0; i < len; ++i)
		{
			Eigen::Vector3f p = points_3d[i];
			xy[2 * i] = p(0);
			xy[2* i + 1] = p(1);
			// std::cout << x[i] << std::endl;
			// std::cout << y[i] << std::endl;
		}

		nmcl_msgs::MergedLaserScan msg;
		msg.header = laserRear->header;
		msg.xy = xy;
		o_mergePub.publish(msg);

	}

private:

	std::shared_ptr<Lidar2D> o_l2d_f;
	std::shared_ptr<Lidar2D> o_l2d_r;

	std::shared_ptr<message_filters::Subscriber<sensor_msgs::LaserScan>> o_laserFrontSub;
	std::shared_ptr<message_filters::Subscriber<sensor_msgs::LaserScan>> o_laserRearSub;
	std::shared_ptr<message_filters::Synchronizer<LidarSyncPolicy>> o_lidarSync;

	ros::Publisher o_mergePub;

};


int main(int argc, char** argv)
{
	ros::init(argc, argv, "LidarMergeNode");
	LidarMergeNode merge = LidarMergeNode();
	ros::spin();
	

	return 0;
}