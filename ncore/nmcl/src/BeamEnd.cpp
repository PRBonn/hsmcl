/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: BeamEnd.cpp                                                           #
# ##############################################################################
**/

#include "BeamEnd.h"
#include "Utils.h"

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

BeamEnd::BeamEnd(std::shared_ptr<GMap> Gmap_, float sigma_, float maxRange_, Weighting weighting )
{
	Gmap = Gmap_;
	maxRange = maxRange_;
	sigma = sigma_;   //value of 8 for map resolution 0.05, 40 for 0.01
	o_weighting = weighting;
	cv::threshold(Gmap->Map(), edt, 127, 255, 0);
	edt = 255 - edt;
	cv::distanceTransform(edt, edt, cv::DIST_L2, cv::DIST_MASK_3);
	cv::threshold(edt, edt, maxRange, maxRange, 2); //Threshold Truncated
	o_coeff = 1.0 / sqrt(2 * M_PI * sigma);
}

void BeamEnd::ComputeWeights(std::vector<Particle>& particles, std::shared_ptr<LidarData> data)
{
	const std::vector<Eigen::Vector3f>& scan = data->Scan();
	const std::vector<double>& scanMask = data->Mask();

	double acc_w = 0;

	#pragma omp parallel for 
	for(long unsigned int i = 0; i < particles.size(); ++i)
	{
		//auto t1 = std::chrono::high_resolution_clock::now();	
		double w = 0;
		switch(o_weighting) 
		{
		    case Weighting::NAIVE : 
		    	w = naive(particles[i].pose, scan, scanMask);
		    	break;
		    case Weighting::INTEGRATION : 
		    	w = integration(particles[i].pose, scan, scanMask);
		    	break;
		    case Weighting::LAPLACE : 
		    	w = laplace(particles[i].pose, scan, scanMask);
		    	break;
		    case Weighting::GEOMETRIC : 
		    	w = geometric(particles[i].pose, scan, scanMask);
		    	break;
		    case Weighting::GPOE : 
		    	w = gPoE(particles[i].pose, scan, scanMask);
		    	break;
		    case Weighting::GIORGIO : 
		    	w = giorgio(particles[i].pose, scan, scanMask);
		    	break;
		}
		particles[i].weight = w;
		acc_w += w;

		//auto t2 = std::chrono::high_resolution_clock::now();
   		//auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
   		//std::cout << "BeamEnd::Timing of compute : " << ns.count() << std::endl;
	}

	//std::cout << acc_w/particles.size() << std::endl;
}

double BeamEnd::giorgio(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{ 

	int rows = edt.rows;
	int cols = edt.cols;
	float radius = 3;

	Eigen::Vector2f pose2d = Gmap->World2Map(Eigen::Vector2f(particle(0), particle(1)));
	if((pose2d(0) < 0) || (pose2d(0) > cols - 1)) return 0;
	if((pose2d(1) < 0) || (pose2d(1) > rows - 1)) return 0;
	float d = std::abs(edt.at<float>(pose2d(1), pose2d(0)));
	if (d < radius) return 0;


	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);

	double cummulative_distance = 0.0;
	int valid = 0;
	double min_weight = 0.001;

	Eigen::Vector2f br = Gmap->BottomRight();

	for (long unsigned int i = 0; i < mapPoints.size(); ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if((mp(0) < 0) || (mp(0) > cols - 1)) continue;
		if((mp(1) < 0) || (mp(1) > rows - 1)) continue;

		float dist = std::abs(edt.at<float>(mp(1), mp(0)));
		if(dist!=dist){
			throw std::runtime_error("BeamEnd::giorgio| nan detected");
		}

		if (dist >= maxRange)
		{
			 continue;
		}
		cummulative_distance += dist;
		++valid;
	}

	if (valid < 30)
	{
		return min_weight;
	}

	double log_likelihood =  0.05 * sigma * (cummulative_distance / (double)valid);
	double w = exp(-log_likelihood) + min_weight;

	return w;
}

double BeamEnd::naive(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{
	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);
	//plotScan(particle, mapPoints);

	Eigen::Vector2f br = Gmap->BottomRight();

	double weight = 1.0;
	int nonValid = 0;

	for(long unsigned int i = 0; i < mapPoints.size(); ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if(scanMask[i] > 0.0)
		{
			if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
			{
					++nonValid;
			}
			else
			{
				float dist = edt.at<float>(mp(1) ,mp(0));
				double w = getLikelihood(dist);
				weight *= w;
			}
		}
	}
	//std::cout << nonValid << std::endl;

	float penalty = pow(getLikelihood(maxRange), nonValid);
	weight *= penalty;

	return weight;
}


double BeamEnd::geometric(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{	
	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);
	//plotScan(particle, mapPoints);

	Eigen::Vector2f br = Gmap->BottomRight();
	float geoW = 1.0 / scan.size();

	double weight = 1.0;
	int nonValid = 0;
	int valid = 0;

	double tot_dist = 0;

	for(long unsigned int i = 0; i < mapPoints.size(); ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if(scanMask[i] > 0.0)
		{
			if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
			{
					++nonValid;
			}
			else
			{
				float dist = edt.at<float>(mp(1) ,mp(0));
				//double w = getLikelihood(dist);
				//weight *= pow(w, geoW);
				tot_dist += pow(dist / sigma, 2.0);
				++valid;
				
			}
		}
	}

	geoW = 1.0 /(valid + nonValid);

	tot_dist +=  nonValid * pow(maxRange / sigma, 2.0);
	weight = o_coeff * exp(-0.5 * tot_dist * geoW);

	//float penalty = pow(getLikelihood(maxRange), nonValid * geoW);
	//weight *= penalty;

	return weight;
}


double BeamEnd::gPoE(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{
	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);
	//plotScan(particle, mapPoints);

	Eigen::Vector2f br = Gmap->BottomRight();
	float geoW = 1.0 / scan.size();

	double weight = 1.0;
	int nonValid = 0;
	int valid = 0;

	for(long unsigned int i = 0; i < mapPoints.size(); ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if(scanMask[i] > 0.0)
		{
			if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
			{
					++nonValid;
			}
			else
			{
				float dist = edt.at<float>(mp(1) ,mp(0));
				double w = getLikelihood(dist);
				if (dist < sigma)
				{
					weight *= pow(w, geoW);
					++valid;
				}
				else ++nonValid;
			}
		}
	}

	float penalty = pow(getLikelihood(maxRange), nonValid * geoW);

	if(valid)
	{
		weight *= penalty;
		weight *= valid;
	}
	else
	{
		weight = penalty;
	}

	return weight;
}


double BeamEnd::integration(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{
	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);
	//plotScan(particle, mapPoints);

	Eigen::Vector2f br = Gmap->BottomRight();

	double sumDist = 0;
	long unsigned numPoints =  mapPoints.size();

	for(long unsigned int i = 0; i < numPoints; ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if(scanMask[i] > 0.0)
		{
			if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
			{
					sumDist += maxRange;
			}
			else
			{
				float dist = edt.at<float>(mp(1) ,mp(0));
				sumDist += dist;
			}
		}
	}

	double weight = getLikelihood(sumDist / numPoints);

	return weight;
}


double BeamEnd::laplace(Eigen::Vector3f particle, const std::vector<Eigen::Vector3f>& scan, std::vector<double> scanMask)
{
	std::vector<Eigen::Vector2f> mapPoints = scan2Map(particle, scan);

	Eigen::Vector2f br = Gmap->BottomRight();
	std::vector<float> distances;

	
	int valid = 0;
	double weight = 1.0;
	double sumDist = 0;
	long unsigned numPoints =  mapPoints.size();

	for(long unsigned int i = 0; i < numPoints; ++i)
	{
		Eigen::Vector2f mp = mapPoints[i];

		if(scanMask[i] > 0.0)
		{
			if ((mp(0) < 0) || (mp(1) < 0) || (mp(0) > br(0)) || (mp(1) > br(1)))
			{
		
					sumDist += maxRange;
			}
			else
			{
				float dist = edt.at<float>(mp(1) ,mp(0));
				sumDist += dist;
				// change sigma to better name when I have nothing to do with my life
				if (dist < sigma)
				{
					++valid;
				}
			}
		}
	}


	double avgDist = sumDist / valid;

	if (valid == 0)
	{
		weight = exp(-maxRange);
		return weight;
	}

	weight = exp(-avgDist);

	return weight;
}




float BeamEnd::getLikelihood(float distance)
{
	float l = o_coeff * exp(-0.5 * pow(distance / sigma, 2));
	return l;
}



void BeamEnd::plotParticles(std::vector<Particle>& particles, std::string title, bool show)
{
	cv::Mat img; 
	cv::cvtColor(Gmap->Map(), img, cv::COLOR_GRAY2BGR);

	cv::namedWindow("Particles", cv::WINDOW_NORMAL);

	for(long unsigned int i = 0; i < particles.size(); ++i)
	{
		Eigen::Vector3f p = particles[i].pose;
		Eigen::Vector2f uv = Gmap->World2Map(Eigen::Vector2f(p(0), p(1)));

		cv::circle(img, cv::Point(uv(0), uv(1)), 5,  cv::Scalar(0, 0, 255), -1);
	}
	if(show)
	{
		cv::imshow("Particles", img);
		cv::waitKey(0);
	}
	cv::imwrite("/home/nickybones/Code/YouBotMCL/nmcl/" + title + ".png", img);

	cv::destroyAllWindows();
}


void BeamEnd::plotScan(Eigen::Vector3f laser, std::vector<Eigen::Vector2f>& zMap)
{
	cv::Mat img; 
	cv::cvtColor(Gmap->Map(), img, cv::COLOR_GRAY2BGR);
	
	cv::namedWindow("Scan", cv::WINDOW_NORMAL);
	Eigen::Vector2f p = Gmap->World2Map(Eigen::Vector2f(laser(0), laser(1)));

	cv::circle(img, cv::Point(p(0), p(1)), 5,  cv::Scalar(255, 0, 0), -1);

	for(long unsigned int i = 0; i < zMap.size(); ++i)
	{
		Eigen::Vector2f p = zMap[i];
		cv::circle(img, cv::Point(p(0), p(1)), 1,  cv::Scalar(0, 0, 255), -1);
	}

	cv::Rect myROI(475, 475, 400, 600);
	// Crop the full image to that image contained by the rectangle myROI
	// Note that this doesn't copy the data
	cv::Mat img_(img, myROI);

	cv::imwrite("scan.png", img_);
	cv::imshow("Scan", img_);
	cv::waitKey(0);


	cv::destroyAllWindows();
}


std::vector<Eigen::Vector2f> BeamEnd::scan2Map(Eigen::Vector3f pose, const std::vector<Eigen::Vector3f>& scan)
{
	Eigen::Matrix3f trans = Vec2Trans(pose);
	std::vector<Eigen::Vector2f> mapPoints(scan.size());

	for(long unsigned int i = 0; i < scan.size(); ++i)
	{
		Eigen::Vector3f ts = trans * scan[i];
		Eigen::Vector2f mp = Gmap->World2Map(Eigen::Vector2f(ts(0), ts(1)));
		//transPoints.push_back(ts);
		mapPoints[i] = mp;
	}

	return mapPoints;
}