/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ConfigNMCLNode.cpp                                                   #
# ##############################################################################
**/

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/LaserScan.h>
#include <nmcl_msgs/TextArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2/LinearMath/Quaternion.h> 
#include <tf2_ros/transform_listener.h>  
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <tf/transform_broadcaster.h>

#include <mutex> 
#include <sstream>  

#include "Camera.h"
#include "Utils.h"
#include "ReNMCL.h"
#include "RosUtils.h"
#include "PlaceRecognition.h"
#include <boost/archive/text_iarchive.hpp>
#include <nlohmann/json.hpp>
#include "NMCLFactory.h"
#include "LidarData.h"
#include "SemanticData.h"
#include <nmcl_msgs/LidarScanMask.h>
#include <nmcl_msgs/YoloArray.h>
#include <nmcl_msgs/Yolo.h>
#include <nmcl_msgs/YoloCombinedArray.h>
#include "std_msgs/Float32MultiArray.h"

#define DEBUG


class ConfigNMCLNode
{
public:
	ConfigNMCLNode()  
	{
		
		ros::NodeHandle nh;
		int numParticles;

		if (!ros::param::has("dataFolder"))
		{
			ROS_FATAL_STREAM("Data folder not found!");
		}


		std::string dataFolder;
		std::string scanTopic;
		std::string odomTopic;
		std::vector<double> odomNoise;  
		std::string textTopic;
		std::string poseTopic;
		std::string nmclconfig;
		std::string yoloTopic;
		std::string sensorConfigFolder;
		std::string roomTopic;
 
		nh.getParam("dataFolder", dataFolder);		
		nh.getParam("scanTopic", scanTopic); 
		nh.getParam("odomTopic", odomTopic); 
		nh.getParam("mapTopic", o_mapTopic);
		nh.getParam("odomNoise", odomNoise);  
		nh.getParam("textTopic", textTopic);
		nh.getParam("dsFactor", o_dsFactor); 
		nh.getParam("triggerDist", o_triggerDist);
		nh.getParam("triggerAngle", o_triggerAngle);
		nh.getParam("poseTopic", poseTopic); 
		nh.getParam("nmclconfig", nmclconfig);  
		nh.getParam("baseLinkTF", o_baseLinkTF);
		nh.getParam("yoloTopic", yoloTopic);
		nh.getParam("configFolder", sensorConfigFolder);
		nh.getParam("maskTopic", o_maskTopic); 
		nh.getParam("roomTopic", roomTopic);   

		o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam0.config"))); 
		o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam1.config")));
		o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam2.config")));
		o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam3.config"))); 
		o_occludedAngles = std::vector<std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>>(4, std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>());

		//srand48(21);
		o_mtx = new std::mutex();   
		o_renmcl = NMCLFactory::Create(dataFolder + nmclconfig); 

		o_dict = o_renmcl->GetFloorMap()->GetRoomNames(); 
		o_placeRec = std::make_shared<PlaceRecognition>(PlaceRecognition(o_dict, dataFolder + "/TextMaps/"));

		nav_msgs::OdometryConstPtr odom = ros::topic::waitForMessage<nav_msgs::Odometry>(odomTopic, ros::Duration(60)); 
		o_prevPose = OdomMsg2Pose2D(odom);
		o_roomProbabilities = {0.0, 0.0, 0.0, 0.0};

		sensor_msgs::PointCloud2ConstPtr pcl = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(scanTopic, ros::Duration(60));
		int scanSize = pcl->width;
		std::vector<double> mask(scanSize, 1.0);
		o_scanMask = Downsample(mask, o_dsFactor);  

		o_maskPub = nh.advertise<sensor_msgs::PointCloud2>(o_maskTopic, 10);
		o_scanSub = nh.subscribe(scanTopic, 1, &ConfigNMCLNode::observationCallback, this);  
		o_odomSub = nh.subscribe(odomTopic, 1, &ConfigNMCLNode::motionCallback, this);   
		o_odomNoise = Eigen::Vector3f(odomNoise[0], odomNoise[1], odomNoise[2]); 		

		o_posePub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(poseTopic, 10); 
		o_particlePub = nh.advertise<geometry_msgs::PoseArray>("Particles", 10);
		o_textSub = nh.subscribe(textTopic, 10, &ConfigNMCLNode::relocalizationCallback, this);   
		o_yoloSub =  nh.subscribe(yoloTopic, 10, &ConfigNMCLNode::semanticCallback, this);  
		o_roomSub = nh.subscribe(roomTopic, 1, &ConfigNMCLNode::roomCallback, this);

#ifdef DEBUG
		o_textPub = nh.advertise<nmcl_msgs::TextArray>("confirmedDetection", 10);
#endif 

		ROS_INFO_STREAM("Engine running!");    
 
	} 

	void roomCallback(const std_msgs::Float32MultiArray::ConstPtr& roomMsg)
	{
		std::vector<float> roomProb = roomMsg->data;

		for(int i = 0; i < o_roomProbabilities.size(); ++i) 
		{
			o_roomProbabilities[i] = (o_roomInitCnt * o_roomProbabilities[i] + roomProb[i]) / (o_roomInitCnt + 1);
		}


		if (o_roomInit) 
		{ 
			if (o_roomInitCnt == 10)
			{
				for (int t = 0; t < o_roomProbabilities.size(); ++t) std::cout << o_roomProbabilities[t] <<  " ";
				std::cout << std::endl;

				o_renmcl->RoomInit(roomProb);
				o_roomInit = false;   	
			} 
		} 
		else
		{
			//o_renmcl->SetRoomProbabilities(o_roomProbabilities);
			//o_renmcl->SetPredictStrategy(ReNMCL::Strategy::BYROOM);
		} 
		o_roomInitCnt++;
	} 
  

	void updateScanMask(std::vector<Eigen::Vector3f>& points_3d)
	{
		std::vector<double> tempMask(o_scanMask.size(), 1.0);
		
		int scanSize = o_scanMask.size();
		std::vector<Eigen::Vector3f> colors(scanSize);

		o_mtx->lock(); 
		for(int i = 0; i < scanSize ; ++i)
		{
			 Eigen::Vector3f p = points_3d[i];
			 Eigen::Vector2f xy_v(p(0), p(1));
			 xy_v = xy_v.normalized();
			 Eigen::Vector3f rgb(0, 255, 0);

		 	for(int d = 0; d < o_occludedFlat.size(); ++d)
		 	{
		 		std::pair<Eigen::Vector2f, Eigen::Vector2f> occ_ang = o_occludedFlat[d];
		 		Eigen::Vector2f xy_l = occ_ang.first;
		 		Eigen::Vector2f xy_r = occ_ang.second;

		 		float norm_gap = xy_l.dot(xy_r);
		 		float norm1 = xy_l.dot(xy_v);
		 		float norm2 = xy_v.dot(xy_r);
		 		if ((norm_gap < norm1) && (norm_gap < norm2))
		 		{
		 			tempMask[i] = 0.0;
		 			rgb = Eigen::Vector3f(255, 0, 0);
		 			break;
		 		}
		 	}
		 	colors[i] = rgb;
		}
		o_mtx->unlock(); 

		o_scanMask = tempMask;

		pcl::PointCloud<pcl::PointXYZRGB> pcl2 = Vec2RGBPointCloud(points_3d, colors);
		sensor_msgs::PointCloud2 pcl_msg2;
		pcl::toROSMsg(pcl2, pcl_msg2);
		pcl_msg2.header.stamp = ros::Time::now();
		pcl_msg2.header.frame_id = o_maskTopic;
		o_maskPub.publish(pcl_msg2);

		tf::Transform transform;
	    transform.setOrigin( tf::Vector3(0.0, 0.0, 0.1) );
		tf::Quaternion q;
		q.setRPY(0, 0, 0);
		transform.setRotation(q);
		o_tfMaskBroadcast.sendTransform(tf::StampedTransform(transform.inverse(), ros::Time::now(), o_baseLinkTF, o_maskTopic));		

	}


 
	void relocalizationCallback(const nmcl_msgs::TextArrayConstPtr& msg)  
	{
		std::vector<std::string> places;
		places = msg->text;
		int camID = msg->id;
		
		std::vector<int> matches = o_placeRec->Match(places);   
		int numMatches = matches.size();
		std::vector<int> validMatches;

		if (numMatches)
		{	
			// float camAngle = 0;
			// if (camID == 1) camAngle = -0.5 * M_PI;
			// else if (camID == 2) camAngle = M_PI;
			// else if (camID == 3) camAngle = 0.5 * M_PI;

			float camAngle = o_cameras[camID]->Yaw();

			for(int i = 0; i < matches.size(); ++i) 
			{
				int id = matches[i];
				//std::cout << o_dict[id] << " detected" << std::endl; 

				if ((camID != o_lastDetectedCamera) || (id != o_lastDetectedMatch))				   
				{	
					o_lastDetectedCamera = camID;
					o_lastDetectedMatch = id;
					validMatches.push_back(id);	
				}
			}

			if (validMatches.size())
			{
				//for(int p = 0; p < places.size(); p++) std::cout << places[p] << std::endl;
				std::vector<std::string> confirmedMatches;
	 			TextData textData = o_placeRec->TextBoundingBoxes(validMatches, confirmedMatches);
				o_mtx->lock(); 
				o_renmcl->Relocalize(textData.BottomRight(), textData.TopLeft(), textData.Orientation(), camAngle);
				o_mtx->unlock();
				for (int p = 0; p < confirmedMatches.size(); ++p)
				{
					ROS_INFO_STREAM("Particles initialized for " << confirmedMatches[p]);   
				}
#ifdef DEBUG

				nmcl_msgs::TextArray new_msg;
				new_msg.header = msg->header;
				new_msg.text =  confirmedMatches;
				new_msg.id = camID;
				o_textPub.publish(new_msg);
#endif
			}


		}   
	}

	void motionCallback(const nav_msgs::OdometryConstPtr& odom)
	{
		Eigen::Vector3f currPose = OdomMsg2Pose2D(odom);

		Eigen::Vector3f delta = currPose - o_prevPose;

		if((((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > o_triggerDist) || (fabs(delta(2)) > o_triggerAngle)) || o_first)
		{
			//ROS_INFO_STREAM("NMCL first step!");
			o_first = false;
			Eigen::Vector3f u = o_renmcl->Backward(o_prevPose, currPose);

			std::vector<Eigen::Vector3f> command{u};
			o_mtx->lock();
			o_renmcl->Predict(command, o_odomWeights, o_odomNoise);
			o_mtx->unlock(); 

			o_prevPose = currPose;   
			o_step = true; 
		}
	}

	Eigen::Vector2f bb2pnt2(const Eigen::Vector4f& semScan, int camID)
	{
		float u1 = 0.5 * (semScan(0) + semScan(2));
		float v1 = 0.5 * (semScan(1) + semScan(3));

		Eigen::Vector3d xyz =  o_cameras[camID]->UV2CameraFrame(Eigen::Vector2f(u1, v1));

		return Eigen::Vector2f(xyz(0), xyz(1));
	}

	Eigen::Vector2f bb2pnt(float u, float v, int camID)
	{
		// float camAngle = 0;
		// if (camID == 1) camAngle = -0.5 * M_PI;
		// else if (camID == 2) camAngle = M_PI;
		// else if (camID == 3) camAngle = 0.5 * M_PI;

		float camAngle = o_cameras[camID]->Yaw();

		Eigen::Vector3d xyz =  o_cameras[camID]->UV2CameraFrame(Eigen::Vector2f(u, v));

		float x = xyz(0);
		float y = xyz(1);		
		float x_ = cos(-camAngle) * x + sin(-camAngle) * y;
        float y_ = -sin(-camAngle) * x + cos(-camAngle) * y;

		return Eigen::Vector2f(x_, y_);
	}


 	void semanticCallback(const nmcl_msgs::YoloCombinedArrayConstPtr& combined_msg)
 	{
 		std::vector<int> labels;
		std::vector<Eigen::Vector2f> poses;
		std::vector<float> confidences;

 		for( int v = 0; v < combined_msg->views.size(); ++v)
 		{
 			nmcl_msgs::YoloArray msg = combined_msg->views[v];
	 		int camID = msg.camID;
	 		std::vector<nmcl_msgs::Yolo> detections = msg.detections;

			o_occludedAngles[camID].clear();

	 		for(int d = 0; d < detections.size(); ++d)
	 		{
	 			int semclass = detections[d].semclass;
	 			float confidence = detections[d].confidence;
	 			float u1 = detections[d].xmin;
	 			float v1 = detections[d].ymin;
	 			float u2 = detections[d].xmax;  
	 			float v2 = detections[d].ymax;

	 			Eigen::Vector2f xy1 = bb2pnt(u1, v1, camID);
 				Eigen::Vector2f xy2 = bb2pnt(u2, v2, camID);

 				float x = 0.5 * (xy1(0) + xy2(0));
				float y = 0.5 * (xy1(1) + xy2(1)); 

 				xy1 = xy1.normalized();
 				xy2 = xy2.normalized();

 				if ((semclass == 5) || (semclass == 12) || (semclass == 10) || (semclass == 8)) 
 				{	
 					o_occludedAngles[camID].push_back(std::pair<Eigen::Vector2f, Eigen::Vector2f>(xy1, xy2));
 				}

				if ((semclass == 5) || (semclass == 12) || (semclass == 10)) continue; 

	            poses.push_back(Eigen::Vector2f(x, y));
	            labels.push_back(int(semclass));
				confidences.push_back(confidence);
	 		}
	 	}

	 	o_mtx->lock(); 
 		o_occludedFlat.clear();
			for(int c = 0; c < o_occludedAngles.size(); ++c)
		{
			o_occludedFlat.insert(std::end(o_occludedFlat), std::begin(o_occludedAngles[c]), std::end(o_occludedAngles[c]));
		}
		o_mtx->unlock(); 


 		if (labels.size())
		{
			SemanticData data = SemanticData(labels, poses, confidences);
			auto t1 = std::chrono::high_resolution_clock::now();
			o_renmcl->CorrectSemantic(std::make_shared<SemanticData>(data));
			auto t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
			//ROS_INFO_STREAM("sem infer : " + std::to_string(fp_ms.count()));   


			SetStatistics stas = o_renmcl->Stats();
			Eigen::Matrix3d cov = stas.Cov();
			Eigen::Vector3d pred = stas.Mean(); 
			if(pred.array().isNaN().any() || cov.array().isNaN().any() || cov.array().isInf().any())
			{ 
				ROS_FATAL_STREAM("NMCL fails to Localize!");
				o_renmcl->Recover();
			}			
		}

 	}



	void observationCallback(const sensor_msgs::PointCloud2ConstPtr& pcl_msg)
	{
		if (o_step)
		{
			int scanSize = o_scanMask.size();
			std::vector<Eigen::Vector3f> points_3d(scanSize);

			pcl::PCLPointCloud2 pcl;
    		pcl_conversions::toPCL(*pcl_msg, pcl);
    		pcl::PointCloud<pcl::PointXYZ> cloud;
    		pcl::fromPCLPointCloud2(pcl, cloud);

    		for(int i = 0; i < scanSize ; ++i)
    		{
    			points_3d[i] = Eigen::Vector3f(cloud.points[i * o_dsFactor].x, cloud.points[i * o_dsFactor].y, cloud.points[i * o_dsFactor].z);
    		}
			updateScanMask(points_3d);

			float sum = std::accumulate(o_scanMask.begin(), o_scanMask.end(), 0.0); 
			{
				LidarData data = LidarData(points_3d, o_scanMask);
				o_mtx->lock();
				auto t1 = std::chrono::high_resolution_clock::now();
				o_renmcl->Correct(std::make_shared<LidarData>(data)); 
				auto t2 = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
				//ROS_INFO_STREAM("lidar infer : " + std::to_string(fp_ms.count()));   


				SetStatistics stas = o_renmcl->Stats();   
				std::vector<Particle> particles = o_renmcl->Particles();          
				o_mtx->unlock();    
			}
			o_step = false;    

			o_mtx->lock();
			SetStatistics stas = o_renmcl->Stats();   
			std::vector<Particle> particles = o_renmcl->Particles();          
			o_mtx->unlock();  
			Eigen::Matrix3d cov = stas.Cov();
			Eigen::Vector3d pred = stas.Mean();   
			o_pred = pred;   


			if(pred.array().isNaN().any() || cov.array().isNaN().any() || cov.array().isInf().any())
			{ 
				ROS_FATAL_STREAM("NMCL fails to Localize!");
				o_renmcl->Recover();
			}
			else    
			{
				geometry_msgs::PoseWithCovarianceStamped poseStamped = Pred2PoseWithCov(pred, cov);
				poseStamped.header.frame_id = o_mapTopic;
				poseStamped.header.stamp = ros::Time::now(); 
				o_posePub.publish(poseStamped); 

				geometry_msgs::PoseArray posearray;
				posearray.header.stamp = ros::Time::now();  
				posearray.header.frame_id = o_mapTopic;
				posearray.poses = std::vector<geometry_msgs::Pose>(particles.size());

				for (int i = 0; i < particles.size(); ++i)
				{
					geometry_msgs::Pose p;
					p.position.x = particles[i].pose(0); 
					p.position.y = particles[i].pose(1);
					p.position.z = 0.1; 
					tf2::Quaternion q;
					q.setRPY( 0, 0, particles[i].pose(2)); 
					p.orientation.x = q[0];
					p.orientation.y = q[1];
					p.orientation.z = q[2];
					p.orientation.w = q[3];

					posearray.poses[i] = p;
				}

				o_particlePub.publish(posearray);
			}	
		}	

		tf::Transform transform;
	    transform.setOrigin( tf::Vector3(o_pred(0), o_pred(1), 0.0) );
		tf::Quaternion q;
		q.setRPY(0, 0, o_pred(2));
		transform.setRotation(q);
		o_tfBroadcast.sendTransform(tf::StampedTransform(transform.inverse(), ros::Time::now(), o_baseLinkTF, o_mapTopic));		
	}


private:

	tf::TransformBroadcaster o_tfBroadcast;
	tf::TransformBroadcaster o_tfMaskBroadcast;
	ros::Publisher o_maskPub;
	ros::Publisher o_posePub;
	ros::Publisher o_particlePub;
	ros::Subscriber o_textSub;
	ros::Subscriber o_yoloSub;
	ros::Subscriber o_scanSub;
	ros::Subscriber o_odomSub;
	ros::Subscriber o_roomSub;
	ros::Publisher o_textPub;

	std::vector<float> o_odomWeights = {1.0};

	Eigen::Vector3f o_prevPose = Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f o_odomNoise = Eigen::Vector3f(0.02, 0.02, 0.02);
	Eigen::Vector3d o_pred;
	std::string o_maskTopic;
	
	std::shared_ptr<ReNMCL> o_renmcl;
	std::shared_ptr<PlaceRecognition> o_placeRec;

	int o_dsFactor = 10;
	std::vector<double> o_scanMask;
	std::string o_mapTopic;
	std::string o_baseLinkTF;
	bool o_first = true;
	bool o_step = false;
	bool o_roomInit = true;
	long int o_roomInitCnt = 0;
	std::vector<float> o_roomProbabilities;
	std::mutex* o_mtx;
	std::vector<std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>> o_occludedAngles;
	std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>> o_occludedFlat;
	std::vector<std::shared_ptr<Camera>> o_cameras;
	float o_triggerDist = 0.05;
	float o_triggerAngle = 0.05;
	int o_lastDetectedCamera = -1;
	int o_lastDetectedMatch = -1;
	int o_cnt = 0;
	std::vector<std::string> o_dict; 
};



int main(int argc, char** argv)
{
	ros::init(argc, argv, "ConfigNMCLNode");
	ConfigNMCLNode nmcl = ConfigNMCLNode();
	ros::spin();
	

	return 0;
}
