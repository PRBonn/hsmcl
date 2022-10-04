/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: TextRecoFromFolder.cpp                                                #
# ##############################################################################
**/



#include "TextSpotting.h"
#include <boost/filesystem.hpp>
#include <fstream>
#include <algorithm>

int main(int argc, char** argv)
{
	std::string jsonPath = "/home/nickybones/Code/YouBotMCL/ncore/data/text/textspotting.config";

	for(int c = 0; c < 4; ++c)
	{
		std::string ImgDirPath = "/home/nickybones/data/MCL/Dump/camera" + std::to_string(c) + "/";
		std::string outputPath = "/home/nickybones/data/MCL/Dump/predictions" + std::to_string(c) + ".txt";
		TextSpotting ts = TextSpotting(jsonPath);
		std::vector<std::string> filenames;

		std::ofstream output;
		output.open(outputPath);

		for (const auto & entry : boost::filesystem::directory_iterator(ImgDirPath))
		{
			std::string imgPath = entry.path().string();
			std::string extension = boost::filesystem::extension(imgPath);
			if(extension == ".png") filenames.push_back(imgPath);
			else std::cout << extension << std::endl;
		}

		std::sort(filenames.begin(), filenames.end());

		for (int f = 0; f < filenames.size(); ++f)
		{
			std::string imgPath = filenames[f];
			std::cout << imgPath << std::endl;
			cv::Mat img = cv::imread(imgPath);
			try 
			{
	   			std::vector<std::string> recRes = ts.Infer(img);
	   			for (int i = 0; i < recRes.size(); ++i)
		        {
		        	output << recRes[i] + ", ";
		        }
			} 
			catch(...) 
			{
	  			std::cerr << "failed to infer" << std::endl;
				
			}
	        output << "\n";

		}

	  output.close();
	}

	return 0;
}

