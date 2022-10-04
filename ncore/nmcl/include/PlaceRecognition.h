/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: PlaceRecognition.h               		                               #
# ##############################################################################
**/


#ifndef PLACERECOGNITION_H
#define PLACERECOGNITION_H

#include <memory>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Dense>

class TextData
{
public:

	TextData(const std::vector<Eigen::Vector2f>& tl, const std::vector<Eigen::Vector2f>& br, const std::vector<float>& orientation)
	{
		o_tl = tl;
		o_br = br;
		o_orientation = orientation;
	}

	std::vector<Eigen::Vector2f> TopLeft()
	{
		return o_tl;
	}

	std::vector<Eigen::Vector2f> BottomRight()
	{
		return o_br;
	}

	std::vector<float> Orientation()
	{
		return o_orientation;
	}


private:

	std::vector<Eigen::Vector2f> o_tl;
	std::vector<Eigen::Vector2f> o_br;
	std::vector<float> o_orientation;

};



class PlaceRecognition
{
public:
	PlaceRecognition(const std::vector<std::string>& dict, const std::string& textMapDir);

	std::vector<int> Match(const std::vector<std::string>& places);

	TextData TextBoundingBoxes(const std::vector<int> matches, std::vector<std::string>& confirmedMatches);

private:

	std::vector<std::string> divideWord(const std::string& word, const char delim = ' ');

	std::vector<std::string> o_dict;
	std::vector<std::vector<std::string>> o_extDict;
	std::vector<cv::Mat> o_textMaps;
	std::vector<cv::Rect> o_textBBs;
	std::vector<float> o_textOrientations;

};




#endif //PLACERECOGNITION