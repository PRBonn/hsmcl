/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                            		   #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                     				   #
#                                                                              #
#  File: SemanticVisibility.h                                                  #
# ##############################################################################
**/

#ifndef SEMANTICVISIBILITY_H
#define SEMANTICVISIBILITY_H




#include <vector>
#include <eigen3/Eigen/Dense>
#include <Particle.h>
#include <map>

#include "SemanticData.h"
#include "GMap.h"


class SemanticVisibility
{
	public:

		//! A constructor
	    /*!
	      \param Gmap is a ptr to a GMAP object, which holds the gmapping map
	    */

		SemanticVisibility(std::shared_ptr<GMap> Gmap, int beams, const std::string& semMapDir, const std::vector<std::string>& classes, const std::vector<float>& confidences);

		//! Computes weights for all particles based on how well the observation matches the map
		/*!
		  \param particles is a vector of Particle elements
		  \param SensorData is an abstract container for sensor data. This function expects SemanticData type
		*/
		// the data is already in base_link coordiantes
		void ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<SemanticData> data);

		void UpdateConsistency(const Particle& particle, std::shared_ptr<SemanticData> data);


		const std::vector<Eigen::Vector2f>& ClassConsistency() const
		{
			return o_classConsistency;
		}


	private:

		int cellID(int x, int y);
		bool isTraced(const cv::Mat& currMap, Eigen::Vector2f pose, Eigen::Vector2f bearing);

		std::vector<std::map<int, std::vector<Eigen::Vector2f>>> o_visibilityMap;
		std::shared_ptr<GMap> o_gmap;
		cv::Size o_mapSize;
		std::vector<float> o_confidenceTH;
		std::vector<Eigen::Vector2f> o_classConsistency;
		std::vector<cv::Mat> o_classMaps;

};

#endif