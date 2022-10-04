/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NMCLFactory.cpp          				                               #
# ##############################################################################
**/


#include <fstream> 

#include "NMCLFactory.h"

#include "Resampling.h"
#include "SetStatistics.h"
#include "GMap.h"
#include "BeamEnd.h"
#include "MixedFSR.h"
#include "FloorMap.h"
#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>
#include "SemanticVisibility.h"

using json = nlohmann::json;

std::shared_ptr<ReNMCL> NMCLFactory::Create(const std::string& configPath)
{
	std::ifstream file(configPath);
	json config;
	file >> config;
	std::string folderPath = boost::filesystem::path(configPath).parent_path().string() + "/";

	std::string sensorModel = config["sensorModel"]["type"];
	std::string motionModel = config["motionModel"];
	bool tracking = config["tracking"]["mode"];
	std::string predictStrategy = config["predictStrategy"];
	bool semantic = config["semantic"]["mode"];


	std::shared_ptr<BeamEnd> sm;
	std::shared_ptr<MixedFSR> mm;
	std::shared_ptr<Resampling> rs;
	std::shared_ptr<FloorMap> fp;
	std::shared_ptr<ReNMCL> renmcl;
	std::shared_ptr<SemanticVisibility> semanticModel;

	int numParticles = config["numParticles"];
	float injRatio = config["injRatio"];

   // std::string jsonPath = folderPath + std::string(config["floorMapPath"]);
   // FloorMap floormap = FloorMap(jsonPath);

    std::string jsonPath = folderPath + std::string(config["floorMapPath"]);
    std::ifstream floorfile(jsonPath);
    json floorconfig;
    floorfile >> floorconfig;
    FloorMap floormap = FloorMap(floorconfig, folderPath);


  	fp = std::make_shared<FloorMap>(floormap);

	if(sensorModel == "BeamEnd")
	{
		float likelihoodSigma = config["sensorModel"]["likelihoodSigma"];
		float maxRange = config["sensorModel"]["maxRange"];
		int wScheme = config["sensorModel"]["weightingScheme"];
		sm = std::make_shared<BeamEnd>(BeamEnd(fp->Map(), likelihoodSigma, maxRange, BeamEnd::Weighting(wScheme)));
	}

	if(semantic)
	{
		int beams = config["semantic"]["beams"];
		std::vector<std::string> classes = config["semantic"]["classes"];
		std::vector<float> confidences = config["semantic"]["confidence"];
		semanticModel = std::make_shared<SemanticVisibility>(SemanticVisibility(fp->Map(), beams, folderPath + std::string("SemMaps/"), classes, confidences));
	}


	if(motionModel == "MixedFSR")
	{
		mm = std::make_shared<MixedFSR>(MixedFSR());
	}
	

	float th = config["resampling"]["lowVarianceTH"];
	rs = std::make_shared<Resampling>(Resampling());
	rs->SetTH(th);
	
	if(tracking)
	{
		//TODO implement these
		std::cout << "tracking" << std::endl;
		Eigen::Vector3f guess(config["tracking"]["x"], config["tracking"]["y"], config["tracking"]["yaw"]);
		Eigen::Vector3f covVec(config["tracking"]["cov_x"], config["tracking"]["cov_y"], config["tracking"]["cov_yaw"]);
		Eigen::Matrix3d cov;
		cov << covVec(0), 0, 0, 0, covVec(1), 0, 0, 0, covVec(2); 
		std::vector<Eigen::Matrix3d> covariances = {cov};
		std::vector<Eigen::Vector3f> initGuesses = {guess};
		renmcl = std::make_shared<ReNMCL>(ReNMCL(fp, mm, sm, rs, semanticModel, numParticles, initGuesses, covariances, injRatio));
	}
	else
	{
		renmcl = std::make_shared<ReNMCL>(ReNMCL(fp, mm, sm, rs, semanticModel, numParticles, injRatio));
	}
	if (predictStrategy == "Uniform")
	{
		renmcl->SetPredictStrategy(ReNMCL::Strategy::UNIFORM);
	}
	else if (predictStrategy == "Gaussian")
	{
		renmcl->SetPredictStrategy(ReNMCL::Strategy::GAUSSIAN);
	}
	else if (predictStrategy == "Giorgio")
	{
		renmcl->SetPredictStrategy(ReNMCL::Strategy::GIORGIO);
	}

	std::cout << "NMCLFactory::Created Successfully!" << std::endl;

	return renmcl;
}


void NMCLFactory::Dump(const std::string& configPath)
{
	json config;
	config["sensorModel"]["type"] = "BeamEnd";
	config["sensorModel"]["likelihoodSigma"] = 8;
	config["sensorModel"]["maxRange"] = 15;
	config["sensorModel"]["weightingScheme"] = 2;
	config["motionModel"] = "MixedFSR";
	config["resampling"]["lowVarianceTH"] = 0.5;
	config["tracking"]["mode"] =  false;
	config["semantic"]["mode"] =  false;
	config["predictStrategy"] = "Uniform";
	config["floorMapPath"] = "floor.config";
	config["numParticles"] = 10000;
	config["injRatio"] = 0.5;

	std::ofstream file(configPath);
	file << std::setw(4) << config << std::endl;
}