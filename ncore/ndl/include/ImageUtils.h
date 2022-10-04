/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: ImageUtils.h            			                                   #
# ##############################################################################
**/

#ifndef IMAGEUTILS
#define IMAGEUTILS

#include "opencv2/opencv.hpp"
#include <vector>

template<typename type>
std::vector<type> Unique(cv::Mat img) 
{
    assert(img.channels() == 1 && "This implementation is only for single-channel images");
    cv::Mat in = img.clone(); 
    auto begin = in.begin<type>(), end = in.end<type>();
    auto last = std::unique(begin, end);    // remove adjacent duplicates to reduce size
    std::sort(begin, last);                 // sort remaining elements
    last = std::unique(begin, last);        // remove duplicates
    return std::vector<type>(begin, last);
}

cv::Mat ThresholdSingleValue(const cv::Mat& img, int val);


//std::vector<cv::Rect> BoundingBox(const cv::Mat& mask);

cv::Rect BoundingBox(const cv::Mat& mask);

cv::Mat DrawRects(const cv::Mat& mask, std::vector<cv::Rect> bboxes);



#endif