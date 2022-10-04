/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ReNMCL.h      	          				                               #
# ##############################################################################
**/

#ifndef RENMCL_H
#define RENMCL_H

#include "MixedFSR.h"
#include "BeamEnd.h"
#include "Resampling.h"
#include "SetStatistics.h"
#include "FloorMap.h"
#include "PlaceRecognition.h"
#include <memory>
#include "SemanticData.h"
#include "LidarData.h"
#include "ParticleFilter.h"
#include "SemanticLikelihood.h"
#include "SemanticVisibility.h"

class ReNMCL
{
	public:


		enum class Strategy 
		{   
			UNIFORM = 0, 
		    GAUSSIAN = 1, 
		    GIORGIO = 2,
		    BYROOM = 3
		};

		//! A constructor
	    /*!
	     \param fm is a ptr to a FloorMap object
	      \param mm is a ptr to a MotionModel object, which is an abstract class. FSR is the implementation 
	      \param sm is a ptr to a BeamEnd object, which is an abstract class. BeamEnd is the implementation 
	      \param rs is a ptr to a Resampling object, which is an abstract class. LowVarianceResampling is the implementation 
	      \param n_particles is an int, and it defines how many particles the particle filter will use
	      \param injectionRatio is an float, and it determines which portions of the particles are replaced when relocalizing
	    */
		ReNMCL(std::shared_ptr<FloorMap> fm, std::shared_ptr<MixedFSR> mm, std::shared_ptr<BeamEnd> sm, 
			std::shared_ptr<Resampling> rs, std::shared_ptr<SemanticVisibility> sem, int n_particles, float injectionRatio = 0.2);


		//! A constructor
	    /*!
	     * \param fm is a ptr to a FloorMap object
	      \param mm is a ptr to a MotionModel object, which is an abstract class. FSR is the implementation 
	      \param sm is a ptr to a BeamEnd object, which is an abstract class. BeamEnd is the implementation 
	      \param rs is a ptr to a Resampling object, which is an abstract class. LowVarianceResampling is the implementation 
	      \param n_particles is an int, and it defines how many particles the particle filter will use
	      \param initGuess is a vector of initial guess for the location of the robots
	      \param covariances is a vector of covariances (uncertainties) corresponding to the initial guesses
	      \param injectionRatio is an float, and it determines which portions of the particles are replaced when relocalizing
	    */
		ReNMCL(std::shared_ptr<FloorMap> fm, std::shared_ptr<MixedFSR> mm, std::shared_ptr<BeamEnd> sm, 
			std::shared_ptr<Resampling> rs, std::shared_ptr<SemanticVisibility> sem, int n_particles, std::vector<Eigen::Vector3f> initGuess, 
			std::vector<Eigen::Matrix3d> covariances, float injectionRatio = 0.2);


		//! A getter for the mean and covariance of the particles
		/*!
		   \return an object SetStatistics, that has fields for mean and covariance
		*/
		SetStatistics Stats()
		{
			return o_stats;
		}


		//! A getter particles representing the pose hypotheses 
		/*!
		   \return A vector of points, where each is Eigen::Vector3f = (x, y, theta)
		*/
		std::vector<Particle> Particles()
		{
			return o_particles;
		}


		//! Advanced all particles according to the control and noise, using the chosen MotionModel's forward function
		/*!
		  \param control is a 3d control command. In the FSR model it's (forward, sideways, rotation)
		  \param odomWeights is the corresponding weight to each odometry source
	      \param noise is the corresponding noise to each control component
		*/
		void Predict(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);

		//! Considers the beamend likelihood of observation for all hypotheses, and then performs resampling 
		/*!
		  \param scan is a vector of homogeneous points (x, y, 1), in the sensor's frame. So the sensor location is (0, 0, 0)
		  		Notice that for the LaserScan messages you need to first transform the ranges to homo points, and then center them 
		*/
		void Correct(std::shared_ptr<LidarData> data);


		//! Considers the semantic likelihood of observation for all hypotheses, and then performs resampling. 
		/*!
		  \param data is an SemanticData ptr that hold the labels and center locations for all detected objects. The center locations are in the base_link frame
		*/
		void CorrectSemantic(std::shared_ptr<SemanticData> data);

		void UpdateConsistency(const Particle& particle, std::shared_ptr<SemanticData> data);

		const std::vector<Eigen::Vector2f>& ClassConsistency() const
		{
			return o_semanticModel2->ClassConsistency();
		}

		void RoomInit(const std::vector<float>& roomProbabilities);




		//! Removes the fraction of weakest particles, and injects an equal number in the places were text cues were detected and matched the map (given by bounding boxes)
		/*!
		  \param br is a vector of 2D points, where each point is the bottom-left corner of a bounding box defining the injection area according to the TextMaps
		   \param tl is a vector of 2D points, where each point is the top-left corner of a bounding box defining the injection area according to the TextMaps
		   \param orientations is a vector of float, where each value is the particle's desired orientation, in the map frame, based on the TextMaps
		  \param camAngle is a float describing the orientation of the camera from which the image was taken.
		*/
		void Relocalize(const std::vector<Eigen::Vector2f>& br, const std::vector<Eigen::Vector2f>& tl, const std::vector<float>& orientations, float camAngle);

		//! Initializes filter with new particles upon localization failure
		void Recover();

		Eigen::Vector3f Backward(Eigen::Vector3f p1, Eigen::Vector3f p2)
		{
			return o_motionModel->Backward(p1, p2);
		}

		//! A setter for the injection ration for particle, in case of text spotting. The ratio represents the fraction of particles that will be removed, and the same number
		// of particles will be injected at the location of detected text. The removed particles are always the ones with the lowest weights. 
		
		void SetInjRation(float ratio)
		{
			o_injectionRatio = ratio;
		}

		const std::shared_ptr<FloorMap>& GetFloorMap()
		{
			return o_floorMap;
		}


		void SetPredictStrategy(Strategy strategy)
		{
			o_predictStrategy = strategy;
		}

		void SetRoomProbabilities(std::vector<float>& roomProbabilities)
		{
			o_roomProbabilities = roomProbabilities;
		}


	private:

		void predictUniform(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);
		void predictGaussian(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);
		void predictGiorgio(const std::vector<Eigen::Vector3f>& control, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);
		void predictRoom(const std::vector<Eigen::Vector3f>& u, const std::vector<float>& odomWeights, const Eigen::Vector3f& noise);


	
		Strategy o_predictStrategy = Strategy(0);
		std::shared_ptr<SemanticLikelihood> o_semanticModel;
		std::shared_ptr<SemanticVisibility> o_semanticModel2;
		std::shared_ptr<ParticleFilter> o_particleFilter;
		std::shared_ptr<GMap> o_gmap;

		std::shared_ptr<MixedFSR> o_motionModel;
		std::shared_ptr<BeamEnd> o_beamEndModel;
		std::shared_ptr<Resampling> o_resampler; 
		std::shared_ptr<FloorMap> o_floorMap;
		int o_numParticles = 0;
		std::vector<Particle> o_particles;
		SetStatistics o_stats;
		float o_injectionRatio = 0.5;
		std::vector<float> o_roomProbabilities;
};

#endif