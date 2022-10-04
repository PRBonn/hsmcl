/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: PlaceRecognition.cpp              		                               #
# ##############################################################################
**/

#include "PlaceRecognition.h"
#include <sstream>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <boost/filesystem.hpp>

PlaceRecognition::PlaceRecognition(const std::vector<std::string>& dict, const std::string& textMapDir)
{
	o_dict = dict;

	int numWords = dict.size();
	o_extDict = std::vector<std::vector<std::string>>(numWords);
	const char delim = ' ';
	for (int w = 0; w < numWords; ++w)
	{
		o_extDict[w] = divideWord(o_dict[w], delim);
	}	

	o_textBBs.push_back(cv::Rect(0, 0, 0, 0));
	o_textOrientations.push_back(0.0);

	for(int i = 1; i < numWords; ++i)
	{
		std::string imgPath = textMapDir + dict[i] + ".png";
		if(boost::filesystem::exists(imgPath))
		{
#ifdef DEBUG
			std::cout << dict[i] << std::endl;
#endif
			cv::Mat img = cv::imread(textMapDir + dict[i] + ".png");
			cv::Mat bgr[3];   
			cv::split(img, bgr);
	 
			cv::Mat heatmap = bgr[0];
			cv::Mat yawmap = bgr[1];
			cv::Mat textmap = bgr[2];

			std::vector<cv::Point2i> locations; 
			cv::findNonZero(heatmap, locations);
			float avgAngle = yawmap.at<uchar>(locations[0]);
			avgAngle = 2 * M_PI* (avgAngle / 255) - M_PI;

			cv::Rect bb = boundingRect(locations);
			o_textBBs.push_back(bb);
			o_textOrientations.push_back(avgAngle);
		}
		else
		{

#ifdef DEBUG
			std::cout << "PlaceRecognition::No textmap for " << dict[i] << std::endl;
#endif
			o_textBBs.push_back(cv::Rect(0, 0, 0, 0));
			o_textOrientations.push_back(0.0);
		}
	}
}


TextData PlaceRecognition::TextBoundingBoxes(const std::vector<int> matches, std::vector<std::string>& confirmedMatches)
{
	int numMatches = matches.size();
	std::vector<cv::Rect> boundingBoxes;

	std::vector<Eigen::Vector2f> tl;
	std::vector<Eigen::Vector2f> br;
	std::vector<float> orientation;

	for(int i = 0; i < matches.size(); ++i)
	{
		int id = matches[i];
		cv::Rect r = o_textBBs[id];
		if(r.area())
		{
			tl.push_back(Eigen::Vector2f(r.x, r.y));
			br.push_back(Eigen::Vector2f(r.x + r.width, r.y + r.height));
			orientation.push_back(o_textOrientations[id]);
			//std::cout << "Found " << o_dict[id] << " " << std::endl;
			confirmedMatches.push_back(o_dict[id]);
		}
	}

	TextData textData(tl, br, orientation);

	return textData;

}


std::vector<int> PlaceRecognition::Match(const std::vector<std::string>& places)
{
	std::vector<int> detections;
	const char delim = ' ';


	for (int p = 0; p < places.size(); ++p)
	{
		for(int d = 0; d < o_dict.size(); ++d)
		{
			std::size_t found = places[p].find(o_dict[d]);
			/*
				will find match if detection is "Room 1$" and room name is "Room 1" since it is a substring of the detection
			*/
			if (found != std::string::npos)
			{
				detections.push_back(d);
			}
			/*
				will find match if detection is "Room1" and room name is "Room 1" by dividing the room name to "Room", "1"
			*/
			else
			{
				bool add = true;
				for (int w = 0; w < o_extDict[d].size(); ++w)
				{
					std::size_t found = places[p].find(o_extDict[d][w]);

					add *= (found != std::string::npos);
				}
				if (add) detections.push_back(d);
			}
		}
	}


	std::vector<std::string> detectedPlaces(detections.size());
	for(int i = 0; i < detections.size(); ++i)
	{
		detectedPlaces[i] = o_dict[detections[i]];
	}

	for(int i = 0; i < detectedPlaces.size(); ++i)
	{
		for(int j = i + 1; j < detectedPlaces.size(); ++j)
		{
			std::size_t found = detectedPlaces[i].find(detectedPlaces[j]);
			if (found != std::string::npos)
			{
				detectedPlaces.erase(detectedPlaces.begin() + j);
				detections.erase(detections.begin() + j);
			}
			else
			{
				found = detectedPlaces[j].find(detectedPlaces[i]);
				if (found != std::string::npos)
				{
					detectedPlaces.erase(detectedPlaces.begin() + i);
					detections.erase(detections.begin() + i);
				}
			}
		}
	}

	return detections;
}


std::vector<std::string> PlaceRecognition::divideWord(const std::string& word, const char delim)
{
	std::vector<std::string> tokens;
	std::stringstream ss(word);
	std::string temp;

	while(std::getline(ss, temp, delim)) 
	{
        tokens.push_back( temp );
    }

    std::sort(tokens.begin(), tokens.end());

    return tokens;
}