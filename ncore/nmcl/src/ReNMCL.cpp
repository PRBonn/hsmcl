/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ReNMCL.cpp                    			                               #
# ##############################################################################
**/


#include "ReNMCL.h"
#include <numeric>
#include <functional> 
#include <iostream>


ReNMCL::ReNMCL(std::shared_ptr<FloorMap> fm, std::shared_ptr<MixedFSR> mm, std::shared_ptr<BeamEnd> sm, 
			std::shared_ptr<Resampling> rs, std::shared_ptr<SemanticVisibility> sem, int n, float injectionRatio)
{
	o_motionModel = mm;
	o_beamEndModel = sm;
	o_resampler = rs;
	o_numParticles = n;
	o_injectionRatio = injectionRatio;
	o_floorMap = fm;
	o_gmap = o_floorMap->Map();

	o_particleFilter = std::make_shared<ParticleFilter>(ParticleFilter(o_floorMap));
	o_particleFilter->InitUniform(o_particles, o_numParticles);
	o_stats = SetStatistics::ComputeParticleSetStatistics(o_particles);
//	std::string mapFolder = "/home/nickybones/Code/OmniNMCL/ncore/data/floor/SMap/SemMaps/";

//	o_semanticModel = std::make_shared<SemanticLikelihood>(SemanticLikelihood(o_floorMap, 6, 255));
	o_semanticModel2 = sem;

}

ReNMCL::ReNMCL(std::shared_ptr<FloorMap> fm, std::shared_ptr<MixedFSR> mm, std::shared_ptr<BeamEnd> sm, 
			std::shared_ptr<Resampling> rs, std::shared_ptr<SemanticVisibility> sem, int n, 
			std::vector<Eigen::Vector3f> initGuess, std::vector<Eigen::Matrix3d> covariances,
			float injectionRatio)
{
	o_motionModel = mm;
	o_beamEndModel = sm;
	o_resampler = rs;
	o_numParticles = n;
	o_injectionRatio = injectionRatio;
	o_floorMap = fm;
	o_gmap = o_floorMap->Map();
	
	o_particleFilter = std::make_shared<ParticleFilter>(ParticleFilter(o_floorMap));
	o_particleFilter->InitGaussian(o_particles, o_numParticles, initGuess, covariances);
	o_stats = o_particleFilter->ComputeStatistics(o_particles);
	o_semanticModel2 = sem;
}

void ReNMCL::RoomInit(const std::vector<float>& roomProbabilities)
{
	o_particleFilter->InitByRoomType(o_particles, o_numParticles, roomProbabilities);
}

void ReNMCL::CorrectSemantic(std::shared_ptr<SemanticData> data)
{
	//o_semanticModel->ComputeWeights(o_particles, data);
	o_semanticModel2->ComputeWeights(o_particles, data);

	o_particleFilter->NormalizeWeights(o_particles);
	o_resampler->Resample(o_particles);
	o_stats = SetStatistics::ComputeParticleSetStatistics(o_particles);
}


void ReNMCL::UpdateConsistency(const Particle& particle, std::shared_ptr<SemanticData> data)
{
	o_semanticModel2->UpdateConsistency(particle, data);
}



void ReNMCL::Predict(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	switch(o_predictStrategy) 
	{
	    case Strategy::UNIFORM : 
	    	predictUniform(u, odomWeights, noise);
	    	break;
	    case Strategy::GAUSSIAN : 
	    	predictGaussian(u, odomWeights, noise);
	    	break;
	    case Strategy::GIORGIO : 
	    	predictGiorgio(u, odomWeights, noise);
	    	break;
	    case Strategy::BYROOM : 
	    	predictRoom(u, odomWeights, noise);
	    	break;
	 }
}

void ReNMCL::predictUniform(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	for(int i = 0; i < o_numParticles; ++i)
	{
		Eigen::Vector3f pose = o_motionModel->SampleMotion(o_particles[i].pose, u, odomWeights, noise);
		o_particles[i].pose = pose;
	
		//particle pruning - if particle is outside the map, we replace it
		while (!o_gmap->IsValid(o_particles[i].pose))
		{
			std::vector<Particle> new_particle;
			o_particleFilter->InitUniform(new_particle, 1);
			new_particle[0].weight = 1.0 / o_numParticles;
			o_particles[i] = new_particle[0];
		}
	}
}

void ReNMCL::predictRoom(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	for(int i = 0; i < o_numParticles; ++i)
	{
		Eigen::Vector3f pose = o_motionModel->SampleMotion(o_particles[i].pose, u, odomWeights, noise);
		o_particles[i].pose = pose;
	
		//particle pruning - if particle is outside the map, we replace it
		while (!o_gmap->IsValid(o_particles[i].pose))
		{
			std::vector<Particle> new_particle;
			o_particleFilter->InitByRoomType(new_particle, 1, o_roomProbabilities);
			new_particle[0].weight = 1.0 / o_numParticles;
			o_particles[i] = new_particle[0];
		}
	}
}

void ReNMCL::predictGaussian(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	Eigen::Matrix3d cov;
	cov << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;
	std::vector<Eigen::Matrix3d> covariances{cov};
	Eigen::Vector3d mean = o_stats.Mean();
	std::vector<Eigen::Vector3f> initGuesses{Eigen::Vector3f(mean(0), mean(1), mean(2))};

	for(int i = 0; i < o_numParticles; ++i)
	{
		Eigen::Vector3f pose = o_motionModel->SampleMotion(o_particles[i].pose, u, odomWeights, noise);
		o_particles[i].pose = pose;
		//particle pruning - if particle is outside the map, we replace it
		while (!o_gmap->IsValid(o_particles[i].pose))
		{
			std::vector<Particle> new_particle;
			o_particleFilter->InitGaussian(new_particle, 1, initGuesses, covariances);
			new_particle[0].weight = 1.0 / o_numParticles;
			o_particles[i] = new_particle[0];
		}
	}	
}

void ReNMCL::predictGiorgio(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise)
{
	for(int i = 0; i < o_numParticles; ++i)
	{
		Eigen::Vector3f pose = o_motionModel->SampleMotion(o_particles[i].pose, u, odomWeights, noise);
		o_particles[i].pose = pose;
	}
}

void ReNMCL::Correct(std::shared_ptr<LidarData> data)
{
	o_beamEndModel->ComputeWeights(o_particles, data);

	o_particleFilter->NormalizeWeights(o_particles);
	o_resampler->Resample(o_particles);
	o_stats = SetStatistics::ComputeParticleSetStatistics(o_particles);

}


void ReNMCL::Recover()
{
	o_particleFilter->InitUniform(o_particles, o_numParticles);
}


void ReNMCL::Relocalize(const std::vector<Eigen::Vector2f>& br, const std::vector<Eigen::Vector2f>& tl, const std::vector<float>& orientations, float camAngle)
{
	int numMatches = tl.size();

	if(numMatches)
	{	
		int numInject = int(o_numParticles * o_injectionRatio);
		int perInject = int(numInject) / numMatches;
		int numRemove = perInject * numMatches;
		int numKeep = o_numParticles - perInject * numMatches;

		o_particleFilter->RemoveWeakest(o_particles, numRemove);

		std::vector<float> yaw;
		for(int i = 0; i < numMatches; ++i)
		{
			yaw.push_back(orientations[i] - camAngle);
		}

		o_particleFilter->AddBoundingBox(o_particles ,perInject, tl, br, yaw);
	}
}




