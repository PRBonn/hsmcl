/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLEngine.h      	            		                               #
# ##############################################################################
**/

#ifndef NMCLENGINE_H
#define NMCLENGINE_H

#include "ReNMCL.h"
#include "TextSpotting.h"
#include "PlaceRecognition.h"
#include "Camera.h"
#include <fstream>
//#include "CustomEigenVec.h"



class NMCLEngine
{
public:

	//! A constructor to the localization engine
    /*!
     \param nmclConfigPath is a string describing the name of the config file for the NMCL factory 
     \param sensorConfigFolder is a string describing the name of the folder where the sensor descriptions are contained 
     \param textMapDir is a a string describing the folder in which the text maps are contained 
    */
	NMCLEngine(const std::string& nmclConfigPath, const std::string& sensorConfigFolder, const std::string& textMapDir);


	//! Wraps the predict functionality of ReNMCL, while maintaining certain conditions, like trigger distance/bearing change before application
	/*!
	  \param odom is an vector of odometry (x, y, yaw)
	*/

	void Predict(Eigen::Vector3f odom);

	//! Wraps the correct functionality of ReNMCL, while maintaining certain conditions, like having sufficient valid beams
	/*!
	  \param scan is a vector of homogeneous points (x, y, 1), in the base_link frame. So the sensor location is (0, 0, 0)
	  		Notice that for the LaserScan messages you need to first transform the ranges to homo points, and then center them 
	*/
	int Correct(const std::vector<Eigen::Vector3f>& scan);


	//! Wraps the correctSemantic functionality of ReNMCL
	/*!
	  \param combinedScan is a vector of vector of points (cls, u1, v1, u2, v2, conf). cls is the semantic label, (u1, v1, u2, v2) is the bounding box coordinates in 
	  			image frame, and conf is the prediction confidence. Each vector of points corresponds to detections from a single camera.
	*/

	int CorrectSemantic(const std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>& combinedScan);

	//! Wraps the UpdateConsistency functionality of ReNMCL
	/*!
	 \param particle is the Particle whose pose corresponds to the base_link pose when the semantic detection is inferred
	  \param combinedScan is a vector of vector of points (cls, u1, v1, u2, v2, conf). cls is the semantic label, (u1, v1, u2, v2) is the bounding box coordinates in 
	  			image frame, and conf is the prediction confidence. Each vector of points corresponds to detections from a single camera.
	*/

	void UpdateConsistency(const Particle& particle, const std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>& combinedScan);

	const std::vector<Eigen::Vector2f>& ClassConsistency() const
	{
		return o_renmcl->ClassConsistency();
	}


	//! Wraps the relocalize functionality of ReNMCL, while ensuring certain conditions, like not repeating particle injection for the same place and the same camera.
	// This function also performs the text spotting 
	/*!
	  \param img is an image from the camera
	  \param camID is the ID of the camera from which the image was taken
	*/
	void TextMask(const cv::Mat& img, int camID);

	//! Wraps the relocalize functionality of ReNMCL, while ensuring certain conditions, like not repeating particle injection for the same place and the same camera
	// This function does not perform text spotting, only matches predictions to the map
	/*!
	  \param places is a vector of strings, which are the text predictions detected by an external text spotting algorithm
	  \param camID is the ID of the camera from which the image was taken
	*/
	void TextMask(const std::vector<std::string>& places, int camID);


	//Eigen::Vector3d PoseEstimation() const;

	SetStatistics PoseEstimation() const
	{
		return o_renmcl->Stats();
	}

	std::vector<Particle> Particles() const
	{
		return o_renmcl->Particles();
	}

	std::vector<double> ScanMask() const
	{
		return o_scanMask;
	}

	Eigen::Vector2f bb2pnt(float u, float v, int camID);

private:	

	void updateScanMask(const std::vector<Eigen::Vector3f>& points_3d);

	std::shared_ptr<TextSpotting> o_textSpotter;
	std::shared_ptr<ReNMCL> o_renmcl;
	std::vector<double> o_scanMask;
	std::shared_ptr<PlaceRecognition> o_placeRec;
	Eigen::Vector3f o_wheelPrevPose = Eigen::Vector3f(0, 0, 0);
	std::vector<std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>> o_occludedAngles;
	std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>> o_occludedFlat;
	bool o_first = true;
	int o_dsFactor = 10;
	Eigen::Vector3f o_odomNoise = Eigen::Vector3f(0.15, 0.15, 0.15);
	std::vector<float> o_odomWeights = {1.0};
	float o_triggerDist = 0.1;
	//float o_triggerDist = 0.05;
	float o_triggerAngle = 0.03;
	bool o_step = false;
	bool o_initPose = true;


	std::vector<std::shared_ptr<Camera>> o_cameras;
	//std::vector<cv::Mat> o_textMaps;
	int o_lastDetectedCamera = -1;
	int o_lastDetectedMatch = -1;
};

#endif
