/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: BeamEnd.h                      		                               #
# ##############################################################################
**/


#ifndef BEAMEND_H
#define BEAMEND_H

#include <memory>
#include <string>
#include "GMap.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include "LidarData.h"
#include "Particle.h"

class BeamEnd
{
	public:

		enum class Weighting 
		{   
			NAIVE = 0, 
		    INTEGRATION = 1, 
		    LAPLACE = 2,
		    GEOMETRIC = 3,
		    GPOE = 4,
		    GIORGIO = 5
		};


		//! A constructor
	    /*!
	      \param Gmap is a ptr to a GMAP object, which holds the gmapping map
	      \param sigma is a float that determine how forgiving the model is (small sigma will give a very peaked likelihood)
	      \param maxRange is a float specifying up to what distance from the sensor a reading is valid
	      \param Weighting is an int specifying which weighting scheme to use
	    */

		BeamEnd(std::shared_ptr<GMap> Gmap, float sigma = 8, float maxRange = 15, Weighting weighting = Weighting::LAPLACE);

		//! Computes weights for all particles based on how well the observation matches the map
		/*!
		  \param particles is a vector of Particle elements
		  \param SensorData is an abstract container for sensor data. This function expects LidarData type
		*/

		void ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<LidarData> data);
		
		//! Returns truth if a particle is in an occupied grid cell, false otherwise. Notice that for particles in unknown areas the return is false.
		/*!
		  \param Particle is a particle with pose and weight
		*/


		void plotParticles(std::vector<Particle>& particles, std::string title, bool show=true); 
	
	
	private:	


		float getLikelihood(float distance);

		void plotScan(Eigen::Vector3f laser, std::vector<Eigen::Vector2f>& zMap); 

		std::vector<Eigen::Vector2f> scan2Map(Eigen::Vector3f pose, const std::vector<Eigen::Vector3f>& scan);

		double naive(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);

		double geometric(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);

		double gPoE(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);

		double giorgio(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);


		double integration(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);

		double laplace(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask);




		std::shared_ptr<GMap> Gmap;
		float maxRange = 15;
		float sigma = 8;
		cv::Mat edt;
		Weighting o_weighting;
		float o_coeff = 1;

};

#endif