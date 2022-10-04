
/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLEngine.cpp    	            		                               #
# ##############################################################################
**/

#include "NMCLEngine.h"
#include "NMCLFactory.h"
#include "LidarData.h"
#include "Utils.h"
#include <numeric>
#include "SemanticData.h"


NMCLEngine::NMCLEngine(const std::string& nmclConfigPath, const std::string& sensorConfigFolder, const std::string& textMapDir)
{

	std::vector<double> mask(1041 * 2, 1.0);
	o_scanMask = Downsample(mask, o_dsFactor);   

	o_renmcl = NMCLFactory::Create(nmclConfigPath);
	std::vector<std::string> dict = o_renmcl->GetFloorMap()->GetRoomNames();
	o_placeRec = std::make_shared<PlaceRecognition>(PlaceRecognition(dict, textMapDir));

	o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam0.config")));
	o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam1.config")));
	o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam2.config")));
	o_cameras.push_back(std::make_shared<Camera>(Camera(sensorConfigFolder + "cam3.config")));

    o_occludedAngles = std::vector<std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>>(4, std::vector<std::pair <Eigen::Vector2f, Eigen::Vector2f>>());

	std::cout << "NMCLEngine::Created Successfully!" << std::endl;
}


void NMCLEngine::Predict(Eigen::Vector3f wheelCurrPose)
{

	if(o_initPose)
	{
		o_wheelPrevPose = wheelCurrPose;
		o_initPose = false;	
	}
	else
	{
		Eigen::Vector3f delta = wheelCurrPose - o_wheelPrevPose;

		if((((sqrt(delta(0) * delta(0) + delta(1) * delta(1))) > o_triggerDist) || (fabs(delta(2)) > o_triggerAngle)) || o_first)
		{
			o_first = false;
			Eigen::Vector3f uWheel = o_renmcl->Backward(o_wheelPrevPose, wheelCurrPose);

			std::vector<Eigen::Vector3f> command{uWheel};
			o_renmcl->Predict(command, o_odomWeights, o_odomNoise);

			o_wheelPrevPose = wheelCurrPose;   
			o_step = true; 
		}
	}
}


int NMCLEngine::Correct(const std::vector<Eigen::Vector3f>& fullScan)
{
	//o_step = true;
	if (o_step && (fullScan.size()))
	{
		//std::vector<Eigen::Vector3f> scan = fullScan;
		int scanSize = o_scanMask.size();
	
		std::vector<Eigen::Vector3f> scan(scanSize);
		for(int i = 0; i < scanSize ; ++i)
		{
			scan[i] = fullScan[i * o_dsFactor];
		}

		 updateScanMask(scan);

		double sum = std::accumulate(o_scanMask.begin(), o_scanMask.end(), 0.0);
		double scanRatio = sum / o_scanMask.size();
		//std::cout << "scanRatio: " << scanRatio << std::endl;
		o_step = false;
		//if (scanRatio > 0.5)
		{
			LidarData data = LidarData(scan, o_scanMask);
			o_renmcl->Correct(std::make_shared<LidarData>(data)); 
		
			SetStatistics stas = o_renmcl->Stats();
			Eigen::Matrix3d cov = stas.Cov();
			Eigen::Vector3d pred = stas.Mean(); 
			if(pred.array().isNaN().any() || cov.array().isNaN().any() || cov.array().isInf().any())
			{ 
				std::cerr << "fails to Localize!" << std::endl;
				o_renmcl->Recover();
			}
			return 1;
		}
		//else
		{
			//std::cout << "ignore observation due to full scan mask" << std::endl;
		}
	}
	return 0;
}


Eigen::Vector2f NMCLEngine::bb2pnt(float u, float v, int camID)
{
	float camAngle = 0;
	if (camID == 1) camAngle = -0.5 * M_PI;
	else if (camID == 2) camAngle = M_PI;
	else if (camID == 3) camAngle = 0.5 * M_PI;

	Eigen::Vector3d xyz =  o_cameras[camID]->UV2CameraFrame(Eigen::Vector2f(u, v));

	float x = xyz(0);
	float y = xyz(1);		
	float x_ = cos(-camAngle) * x + sin(-camAngle) * y;
    float y_ = -sin(-camAngle) * x + cos(-camAngle) * y;

	return Eigen::Vector2f(x_, y_);
}

void NMCLEngine::UpdateConsistency(const Particle& particle, const std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>& combinedScan)
{
	std::vector<int> labels;
	std::vector<Eigen::Vector2f> poses;
	std::vector<float> confidences;

	for(int camID = 0; camID < combinedScan.size(); ++camID)
	{
		std::vector<Eigen::Matrix<float, 6, 1>> semScan = combinedScan[camID];

		for(int i = 0; i < semScan.size(); ++i)
		{
			Eigen::Matrix<float, 6, 1> scan = semScan[i];
			int semclass = int(scan(0));
			float u1 = scan(1);
			float v1 = scan(2);
			float u2 = scan(3);
			float v2 = scan(4);
			float confidence = scan(5);

			Eigen::Vector2f xy1 = bb2pnt(u1, v1, camID);
			Eigen::Vector2f xy2 = bb2pnt(u2, v2, camID);
			float x = 0.5 * (xy1(0) + xy2(0));
			float y = 0.5 * (xy1(1) + xy2(1));

			xy1 = xy1.normalized();
			xy2 = xy2.normalized();

            poses.push_back(Eigen::Vector2f(x, y));
            labels.push_back(semclass);
			confidences.push_back(confidence);
		}
	}

	if (labels.size())
	{
		SemanticData data = SemanticData(labels, poses, confidences);
		o_renmcl->UpdateConsistency(particle, std::make_shared<SemanticData>(data));
	}
}


int NMCLEngine::CorrectSemantic(const std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>& combinedScan)
{

	std::vector<int> labels;
	std::vector<Eigen::Vector2f> poses;
	std::vector<float> confidences;

	o_occludedFlat.clear();

	for(int camID = 0; camID < combinedScan.size(); ++camID)
	{
		o_occludedAngles[camID].clear();
		std::vector<Eigen::Matrix<float, 6, 1>> semScan = combinedScan[camID];

		for(int i = 0; i < semScan.size(); ++i)
		{
			Eigen::Matrix<float, 6, 1> scan = semScan[i];
			int semclass = int(scan(0));
			float u1 = scan(1);
			float v1 = scan(2);
			float u2 = scan(3);
			float v2 = scan(4);
			float confidence = scan(5);

			Eigen::Vector2f xy1 = bb2pnt(u1, v1, camID);
			Eigen::Vector2f xy2 = bb2pnt(u2, v2, camID);
			float x = 0.5 * (xy1(0) + xy2(0));
			float y = 0.5 * (xy1(1) + xy2(1));

			xy1 = xy1.normalized();
			xy2 = xy2.normalized();

			//if ((semclass == 5) || (semclass == 7) ||  (semclass == 8) ||  (semclass == 10) || (semclass == 12) || (semclass == 13))
			if ((semclass == 5) ||  (semclass == 10) || (semclass == 12))
			{
				o_occludedAngles[camID].push_back(std::pair<Eigen::Vector2f, Eigen::Vector2f>(xy1, xy2));
			}	

			if ((semclass == 5) || (semclass == 10) || (semclass == 12)) continue;
            poses.push_back(Eigen::Vector2f(x, y));
            labels.push_back(semclass);
			confidences.push_back(confidence);
		}
		o_occludedFlat.insert(std::end(o_occludedFlat), std::begin(o_occludedAngles[camID]), std::end(o_occludedAngles[camID]));
	}

	if (labels.size())
	{
		SemanticData data = SemanticData(labels, poses, confidences);
		o_renmcl->CorrectSemantic(std::make_shared<SemanticData>(data));

		SetStatistics stas = o_renmcl->Stats();
		Eigen::Matrix3d cov = stas.Cov();
		Eigen::Vector3d pred = stas.Mean(); 
		if(pred.array().isNaN().any() || cov.array().isNaN().any() || cov.array().isInf().any())
		{ 
			std::cerr << "fails to Localize!" << std::endl;
			o_renmcl->Recover();
		}
		//o_step = false;
		return 1;
	}

	return 0;
}


void NMCLEngine::updateScanMask(const std::vector<Eigen::Vector3f>& points_3d)
{
	std::vector<double> tempMask(o_scanMask.size(), 1.0);
		
	int scanSize = o_scanMask.size();

	for(int i = 0; i < scanSize ; ++i)
	{
		 Eigen::Vector3f p = points_3d[i];
		 Eigen::Vector2f xy_v(p(0), p(1));
		 xy_v = xy_v.normalized();

	 	for(int d = 0; d < o_occludedFlat.size(); ++d)
	 	{
	 		std::pair<Eigen::Vector2f, Eigen::Vector2f> occ_ang = o_occludedFlat[d];
	 		Eigen::Vector2f xy_l = occ_ang.first;
	 		Eigen::Vector2f xy_r = occ_ang.second;

	 		float norm_gap = xy_l.dot(xy_r);
	 		float norm1 = xy_l.dot(xy_v);
	 		float norm2 = xy_v.dot(xy_r);
	 		if ((norm_gap < norm1) && (norm_gap < norm2))
	 		{
	 			tempMask[i] = 0.0;
	 			break;
	 		}
	 	}
	}

	o_scanMask = tempMask;
}



void NMCLEngine::TextMask(const cv::Mat& img, int camID)
{
	std::vector<std::string> places = o_textSpotter->Infer(img);

	TextMask(places, camID);
}

void NMCLEngine::TextMask(const std::vector<std::string>& places, int camID)
{
	std::vector<int> matches = o_placeRec->Match(places);   
	int numMatches = matches.size();

	if (numMatches)
	{	
		std::vector<int> validMatches;

		float camAngle = 0;
		if (camID == 1) camAngle = -0.5 * M_PI;
		else if (camID == 2) camAngle = M_PI;
		else if (camID == 3) camAngle = 0.5 * M_PI;

		//std::cout << "detection!" << std::endl;
		for(int i = 0; i < matches.size(); ++i)
		{
			int id = matches[i];

			if ((camID != o_lastDetectedCamera) || (id != o_lastDetectedMatch))				
			{	
				o_lastDetectedCamera = camID;
				o_lastDetectedMatch = id;
				validMatches.push_back(id);	
				std::cout << "id: " << id << "cam: " << camID << std::endl;
			}
		}

		if (validMatches.size())
		{
			std::vector<std::string> confirmedMatches;
			TextData textData = o_placeRec->TextBoundingBoxes(validMatches, confirmedMatches);
			o_renmcl->Relocalize(textData.BottomRight(), textData.TopLeft(), textData.Orientation(), camAngle);
		}
	} 
}







