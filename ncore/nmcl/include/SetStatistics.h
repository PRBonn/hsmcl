/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SetStatistics.h          			                           		   #
# ##############################################################################
**/


#ifndef SETSTATISTICS_H
#define SETSTATISTICS_H

#include <eigen3/Eigen/Dense>
#include <vector>
#include "Particle.h"

class SetStatistics
{
	public:

		SetStatistics(Eigen::Vector3d m = Eigen::Vector3d::Zero(), Eigen::Matrix3d c = Eigen::Matrix3d::Zero())
		{
			mean = m;
			cov = c;
		}

		Eigen::Vector3d Mean()
		{
			return mean;
		}

		Eigen::Matrix3d Cov()
		{
			return cov;
		}

		static SetStatistics ComputeParticleSetStatistics(const std::vector<Particle>& particles);

	private:

		Eigen::Vector3d mean;
		Eigen::Matrix3d cov;
	
};


#endif