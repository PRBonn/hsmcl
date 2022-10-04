/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: TextSpotting.h                                                        #
# ##############################################################################
**/

#ifndef TEXTSPOTTING
#define TEXTSPOTTING

#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/dnn/dnn.hpp>




class TextSpotting
{
public:

    TextSpotting(std::string detModelPath, std::string recModelPath, std::string vocPath,
      int height = 480, int width = 640, float binThresh = 0.3, float polyThresh = 0.5,
      uint maxCandidates = 50, double unclipRatio = 2.0);

     TextSpotting(std::string jsonPath);

    std::vector<std::string> Infer(const cv::Mat& img);
    std::vector<std::string> InferDebug(const cv::Mat& frame, std::vector< std::vector<cv::Point>>& contours);


private:

    void fourPointsTransform(const cv::Mat& frame, const cv::Point2f vertices[], cv::Mat& result);
    bool sortPts(const cv::Point& p1, const cv::Point& p2);


    std::shared_ptr<cv::dnn::TextDetectionModel_DB> o_detector;
    std::shared_ptr<cv::dnn::TextRecognitionModel> o_recognizer;


};

#endif


