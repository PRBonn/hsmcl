/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: SemanticLikelihood.cpp                   		                       #
# ##############################################################################
**/

#include "SemanticLikelihood.h"
#include "Utils.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "SemanticData.h"


SemanticLikelihood::SemanticLikelihood(std::shared_ptr<FloorMap> floorMap, const std::string& semMapDir, float sigma, float maxRange)
{
	o_floorMap = floorMap;
	o_gmap = o_floorMap->Map();

	o_maxRange = maxRange;
	o_coeff = 1.0 / sqrt(2 * M_PI * sigma);

	createEDT(maxRange, semMapDir);
}


void SemanticLikelihood::createEDT(float maxRange, const std::string& semMapDir)
{
	o_distMaps = std::vector<cv::Mat>(1);

	for (int c = 0; c < o_distMaps.size(); ++c )
	{
		//load class block image
		cv::Mat objLoc = cv::imread(semMapDir + "Sink.png", 0);
		cv::Mat edt(objLoc.size(), CV_32F, cv::Scalar(255)); 

		int roomNum = o_floorMap->GetRoomsNum();
		for(int r = 0; r < roomNum; ++r)
		{
			Room room = o_floorMap->GetRoom(r);
			std::vector<Object> objects = room.Objects();
			std::vector<Object>::iterator iter = std::find_if(objects.begin(), objects.end(), [c](const auto& o){ return o.SemLabel() == c; } );
			if (iter != objects.end())
			{
				std::cout << "Room" + std::to_string(r) + ".png" << std::endl;
				cv::Mat room = cv::imread(semMapDir + "Room" + std::to_string(r) + ".png", 0);
				cv::Mat roomObjLoc;
				objLoc.copyTo(roomObjLoc, room);
				cv::Mat roomedt;
				cv::threshold(roomObjLoc, roomedt, 127, 255, 0);
				cv::distanceTransform(255 - roomedt, roomedt, cv::DIST_L2, cv::DIST_MASK_3);
				roomedt.copyTo(edt, room);
			}
		} 
		cv::imwrite(semMapDir + "sem_edt.png", edt);
		cv::threshold(edt, edt, maxRange, maxRange, 2); //Threshold Truncated
		o_distMaps[c] = edt;
	}
}


void SemanticLikelihood::ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<SemanticData> data)
{
	const std::vector<Eigen::Vector2f>& poses = data->Pos();
	const std::vector<int>& labels = data->Label();

	Eigen::Vector2f br = o_gmap->BottomRight();


	Eigen::Vector2f semPos = poses[0];
	int semClass = labels[0];
	Eigen::Vector3f scan(semPos(0), -semPos(1), 1);


	for(long unsigned int i = 0; i < particles.size(); ++i)
	{
		double w = 0;

		Eigen::Vector2f mp =scan2Map(particles[i].pose, scan);
		//std::cout << mp(0) << ", " << mp(1) << std::endl;


		if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
		{
				w = getLikelihood(o_maxRange);
		}
		else if (isOccluded(particles[i].pose, mp))
		{
			w = getLikelihood(o_maxRange);
			//std::cout << "occluded" << std::endl;
		}
		else
		{
			float dist = o_distMaps[semClass].at<float>(mp(1) ,mp(0));
			w = getLikelihood(dist);
			//std::cout << mp(0) << ", " << mp(1) << std::endl;
			//std::cout << dist << std::endl;
		}

		particles[i].weight = w;
		//std::cout << w << std::endl;
	}
}

bool SemanticLikelihood::isOccluded(Eigen::Vector3f pose, Eigen::Vector2f sUV)
{
	//bool occluded = false;

	Eigen::Vector2f pUV = o_gmap->World2Map(Eigen::Vector2f(pose(0), pose(1)));
	Eigen::Vector2f dir = sUV - pUV;
	int numSteps = 10;
		
	for(int i = 1; i < numSteps; ++i)
	{
		Eigen::Vector2f pnt = pUV + (1.0/float(numSteps)) * i * dir;
		int val = o_gmap->Map().at<uchar>(int(pnt(1)) ,int(pnt(0)));
	  	if (val > 1) return true;	
	}
/*
	int x1 = pUV(0);
	int y1 = pUV(1);
	int x2 = sUV(0);
	int y2 = sUV(1);

	int m_new = 2 * (y2 - y1);
	int slope_error_new = m_new - (x2 - x1);
	for (int x = x1, y = y1; x <= x2; x++)
	{
	  //cout << "(" << x << "," << y << ")\n";

	  // Add slope to increment angle formed
	  slope_error_new += m_new;

	  // Slope error reached limit, time to
	  // increment y and update slope error.
	  if (slope_error_new >= 0)
	  {
	     y++;
	     slope_error_new  -= 2 * (x2 - x1);
	  }
	  //line.append(Eigen::Vector2f(x, y));
	  Eigen::Vector2f pnt(x, y);
	  int val = o_gmap->Map().at<uchar>(pnt(1) ,pnt(0));
	  if (val > 1) return true;
	}
*/
	return false;
}




float SemanticLikelihood::getLikelihood(float distance)
{
	float l = o_coeff * exp(-0.5 * pow(distance / o_sigma, 2));
	return l;
}


Eigen::Vector2f SemanticLikelihood::scan2Map(Eigen::Vector3f pose, Eigen::Vector3f scan)
{
	Eigen::Matrix3f trans = Vec2Trans(pose);

	Eigen::Vector3f ts = trans * scan;
	Eigen::Vector2f mp = o_gmap->World2Map(Eigen::Vector2f(ts(0), ts(1)));

	return mp;
}
