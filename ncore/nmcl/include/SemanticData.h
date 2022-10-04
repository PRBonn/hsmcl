/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SemanticData.h          		                           		       #
# ##############################################################################
**/


#ifndef SEMANTICDATA_H
#define SEMANTICDATA_H

#include <vector>
#include <eigen3/Eigen/Dense>


class SemanticData 
{

	public: 

		SemanticData(const std::vector<int>&  labels, const std::vector<Eigen::Vector2f>& poses, const std::vector<float>& confidences)
		{
			o_labels = labels;
			o_poses = poses;
			o_confidences = confidences;
		}

		//virtual ~LidarData(){};

		const std::vector<Eigen::Vector2f>& Pos() const
		{
			return o_poses;
		}

		const std::vector<int>& Label() const
		{
			return o_labels;
		}

		const std::vector<float>& Confidence() const
		{
			return o_confidences;
		}

	private:

		std::vector<Eigen::Vector2f> o_poses;
		std::vector<int> o_labels;
		std::vector<float> o_confidences;

};

#endif