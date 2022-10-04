/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ImageUtils.cpp           			                                   #
# ##############################################################################
**/

#include "ImageUtils.h"


cv::Mat ThresholdSingleValue(const cv::Mat& img, int val)
{
	cv::Mat mask;
    cv::threshold( img, mask, val, 255, 4 );
    cv::threshold( mask, mask, val - 1, 255, 0 );

    return mask;
}

/*
std::vector<cv::Rect> BoundingBox(const cv::Mat& mask)
{
	std::vector<cv::Rect> bboxes;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    for( size_t i = 0; i < contours.size(); i++ )
    {
    	cv::Rect bb = cv::boundingRect(contours[i]);
    	bboxes.push_back(bb);
    }

    return bboxes;
}*/


cv::Rect BoundingBox(const cv::Mat& mask)
{
	std::vector<cv::Point> locations;
	cv::findNonZero(mask, locations);
	cv::Rect bb = cv::boundingRect(locations);

	return bb;
}



cv::Mat DrawRects(const cv::Mat& mask, std::vector<cv::Rect> bboxes)
{
	cv::RNG rng(12345);
	cv::Mat drawing = mask.clone();
    cv::cvtColor(drawing,drawing,cv::COLOR_GRAY2BGR);

    for( size_t i = 0; i < bboxes.size(); i++ )
    {
		cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
		rectangle(drawing, bboxes[i], color);
	}

	return drawing;
}
