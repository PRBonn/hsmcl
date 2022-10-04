/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ParticleFilter.h             		                           		   #
# ##############################################################################
**/
 

#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include "Particle.h"
#include "GMap.h"
#include "SetStatistics.h"
#include "FloorMap.h"

class ParticleFilter
{
public:

	ParticleFilter(std::shared_ptr<FloorMap> floorMap);

	void InitByRoomType(std::vector<Particle>& particles, int n_particles, const std::vector<float>& roomProbabilities);

	void InitUniform(std::vector<Particle>& particles, int n_particles);

	void InitGaussian(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector3f>& initGuess, const std::vector<Eigen::Matrix3d>& covariances);

	void RemoveWeakest(std::vector<Particle>& particles, int n_particles);

	void AddUniform(std::vector<Particle>& particles, int n_particles);

	Eigen::Vector3f CreateSingleUniform();

	void AddGussian(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector3f>& initGuess, const std::vector<Eigen::Matrix3d>& covariances);

	void AddBoundingBox(std::vector<Particle>& particles, int n_particles, const std::vector<Eigen::Vector2f>& tls, const std::vector<Eigen::Vector2f>& brs, const std::vector<float>& yaws);

	SetStatistics ComputeStatistics(const std::vector<Particle>& particles);

	void NormalizeWeights(std::vector<Particle>& particles);

	std::vector<Particle>& Particles()
	{
		return o_particles;
	}

	void SetParticle(int id, Particle p)
	{
		o_particles[id] = p;
	}


	SetStatistics Statistics()
	{
		return o_stats;
	}



private:



	std::shared_ptr<FloorMap> o_floorMap;
	std::shared_ptr<GMap> o_gmap;
	std::vector<Particle> o_particles;
	SetStatistics o_stats;
};

#endif