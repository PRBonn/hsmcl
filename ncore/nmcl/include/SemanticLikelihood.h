/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SemanticLikelihood.h                      		                       #
# ##############################################################################
**/


#ifndef SEMANTICLIKELIHOOD_H
#define SEMANTICLIKELIHOOD_H

#include <memory>
#include <string>
#include <Particle.h>
#include "GMap.h"
#include "FloorMap.h"
#include "SemanticData.h"


class SemanticLikelihood
{
	public:



		//! A constructor
	    /*!
	      \param FloorMap is a ptr to a FloorMap object, which holds the floor map
	      \param sigma is a float that determine how forgiving the model is (small sigma will give a very peaked likelihood)
	      \param maxRange is a float specifying up to what distance from the sensor a reading is valid
	    */

		SemanticLikelihood(std::shared_ptr<FloorMap> floorMap, const std::string& semMapDir, float sigma = 8, float maxRange = 15);


			//! Computes weights for all particles based on how well the observation matches the map
		/*!
		  \param particles is a vector of Particle elements
		  \param SensorData is an abstract container for sensor data. This function expects LidarData type
		*/

		void ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<SemanticData> data);

	private:

		float getLikelihood(float distance);
		bool isOccluded(Eigen::Vector3f pose, Eigen::Vector2f sUV);
		void createEDT(float maxRange, const std::string& semMapDir);


		Eigen::Vector2f scan2Map(Eigen::Vector3f pose, Eigen::Vector3f scan);


		std::shared_ptr<FloorMap> o_floorMap;
		std::shared_ptr<GMap> o_gmap;
		float o_maxRange = 15;
		float o_sigma = 8;
		float o_coeff = 1;
		std::vector<cv::Mat> o_distMaps;


};





#endif