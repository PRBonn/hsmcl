/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: LidarData.h          			                           		       #
# ##############################################################################
**/


#ifndef LIDARDATA_H
#define LIDARDATA_H

#include <vector>
#include <eigen3/Eigen/Dense>

class LidarData 
{

	public: 

		LidarData(const std::vector<Eigen::Vector3f>& scan, const std::vector<double>& mask)
		{
			o_scan = scan;
			o_mask = mask;
		}

		//virtual ~LidarData(){};

		std::vector<Eigen::Vector3f>& Scan()
		{
			return o_scan;
		}

		std::vector<double>& Mask()
		{
			return o_mask;
		}

	private:

		std::vector<Eigen::Vector3f> o_scan;
		std::vector<double> o_mask;

};

#endif