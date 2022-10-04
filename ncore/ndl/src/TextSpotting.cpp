/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: TextSpotting.cpp                                                      #
# ##############################################################################
**/


#include "TextSpotting.h"
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <boost/filesystem.hpp>



 TextSpotting::TextSpotting(std::string detModelPath, std::string recModelPath, std::string vocPath,
      int height, int width, float binThresh, float polyThresh, uint maxCandidates, double unclipRatio)
 {
   // Load networks
    CV_Assert(!detModelPath.empty());
    o_detector = std::make_shared<cv::dnn::TextDetectionModel_DB>(cv::dnn::TextDetectionModel_DB(detModelPath));

    o_detector->setBinaryThreshold(binThresh);
    o_detector->setPolygonThreshold(polyThresh);
    o_detector->setUnclipRatio(unclipRatio);
    o_detector->setMaxCandidates(maxCandidates);


    CV_Assert(!recModelPath.empty());
    o_recognizer = std::make_shared<cv::dnn::TextRecognitionModel>(cv::dnn::TextRecognitionModel(recModelPath));

    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(cv::samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    std::string vocLine;
    std::vector<std::string> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    o_recognizer->setVocabulary(vocabulary);
    o_recognizer->setDecodeType("CTC-greedy");

    // Parameters for Detection
    double detScale = 1.0 / 255.0;
    cv::Size detInputSize = cv::Size(width, height);
    cv::Scalar detMean = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
    o_detector->setInputParams(detScale, detInputSize, detMean);

    // Parameters for Recognition
    double recScale = 1.0 / 127.5;
    cv::Scalar recMean = cv::Scalar(127.5);
    cv::Size recInputSize = cv::Size(100, 32);
    o_recognizer->setInputParams(recScale, recInputSize, recMean);

 }

  TextSpotting::TextSpotting(std::string jsonPath)
  {
    using json = nlohmann::json;

    //std::string folderPath = boost::filesystem::path(jsonPath).parent_path().string() + "/";

    std::ifstream file(jsonPath);
    json config;
    file >> config;

    std::string detModelPath = config["detModelPath"];
    std::string recModelPath = config["recModelPath"];
    std::string vocPath = config["vocPath"];
    int height = config["height"];
    int width = config["width"]; 
    float binThresh = config["binThresh"]; 
    float polyThresh = config["polyThresh"]; 
    uint maxCandidates = config["maxCandidates"]; 
    double unclipRatio = config["unclipRatio"];


    // Load networks
    CV_Assert(!detModelPath.empty());
    o_detector = std::make_shared<cv::dnn::TextDetectionModel_DB>(cv::dnn::TextDetectionModel_DB(detModelPath));

    o_detector->setBinaryThreshold(binThresh);
    o_detector->setPolygonThreshold(polyThresh);
    o_detector->setUnclipRatio(unclipRatio);
    o_detector->setMaxCandidates(maxCandidates);


    CV_Assert(!recModelPath.empty());
    o_recognizer = std::make_shared<cv::dnn::TextRecognitionModel>(cv::dnn::TextRecognitionModel(recModelPath));

    CV_Assert(!vocPath.empty());
    std::ifstream vocFile;
    vocFile.open(cv::samples::findFile(vocPath));
    CV_Assert(vocFile.is_open());
    std::string vocLine;
    std::vector<std::string> vocabulary;
    while (std::getline(vocFile, vocLine)) {
        vocabulary.push_back(vocLine);
    }
    o_recognizer->setVocabulary(vocabulary);
    o_recognizer->setDecodeType("CTC-greedy");

    // Parameters for Detection
    double detScale = 1.0 / 255.0;
    cv::Size detInputSize = cv::Size(width, height);
    cv::Scalar detMean = cv::Scalar(122.67891434, 116.66876762, 104.00698793);
    o_detector->setInputParams(detScale, detInputSize, detMean);

    // Parameters for Recognition
    double recScale = 1.0 / 127.5;
    cv::Scalar recMean = cv::Scalar(127.5);
    cv::Size recInputSize = cv::Size(100, 32);
    o_recognizer->setInputParams(recScale, recInputSize, recMean);



  }

 std::vector<std::string> TextSpotting::Infer(const cv::Mat& frame)
 {
    std::vector<std::string> places;

    std::vector< std::vector<cv::Point> > detResults;
    o_detector->detect(frame, detResults);


    if (detResults.size() > 0) 
    {
        // Text Recognition
        cv::Mat recInput;
        recInput = frame;

        /*
        if (frame.channels() > 1) 
        {
            cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
        } 
        else recInput = frame;*/
        
        std::vector< std::vector<cv::Point> > contours;
        for (uint i = 0; i < detResults.size(); i++)
        {
            const auto& quadrangle = detResults[i];
            CV_CheckEQ(quadrangle.size(), (size_t)4, "");

            contours.emplace_back(quadrangle);

            std::vector<cv::Point2f> quadrangle_2f;
            for (int j = 0; j < 4; j++)
                quadrangle_2f.emplace_back(quadrangle[j]);

            // Transform and Crop
            cv::Mat cropped;
            fourPointsTransform(recInput, &quadrangle_2f[0], cropped);

            std::string recoRes = o_recognizer->recognize(cropped);
            places.push_back(recoRes);
          
            //std::cout << i << ": '" << recoRes << "'" << std::endl;

            //cv::putText(frame, recoRes, quadrangle[3], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        //cv::polylines(frame, contours, true, cv::Scalar(0, 255, 0), 2);
    } 
    
    //cv::imshow("text", frame);
    //cv::waitKey();

    return places;
 }

 std::vector<std::string> TextSpotting::InferDebug(const cv::Mat& frame, std::vector< std::vector<cv::Point>>& contours)
 {
    std::vector<std::string> places;

    std::vector< std::vector<cv::Point> > detResults;
    o_detector->detect(frame, detResults);


    if (detResults.size() > 0) 
    {
        // Text Recognition
        cv::Mat recInput;
        recInput = frame;

        /*
        if (frame.channels() > 1) 
        {
            cvtColor(frame, recInput, cv::COLOR_BGR2GRAY);
        } 
        else recInput = frame;*/
        
        //std::vector< std::vector<cv::Point> > contours;
        for (uint i = 0; i < detResults.size(); i++)
        {
            const auto& quadrangle = detResults[i];
            CV_CheckEQ(quadrangle.size(), (size_t)4, "");

            contours.emplace_back(quadrangle);

            std::vector<cv::Point2f> quadrangle_2f;
            for (int j = 0; j < 4; j++)
                quadrangle_2f.emplace_back(quadrangle[j]);

            // Transform and Crop
            cv::Mat cropped;
            fourPointsTransform(recInput, &quadrangle_2f[0], cropped);

            std::string recoRes = o_recognizer->recognize(cropped);
            places.push_back(recoRes);
          
            //std::cout << i << ": '" << recoRes << "'" << std::endl;

            //cv::putText(frame, recoRes, quadrangle[3], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        //cv::polylines(frame, contours, true, cv::Scalar(0, 255, 0), 2);
    } 
    
    //cv::imshow("text", frame);
    //cv::waitKey();

    return places;
 }


 void TextSpotting::fourPointsTransform(const cv::Mat& frame, const cv::Point2f vertices[], cv::Mat& result)
{
    const cv::Size outputSize = cv::Size(100, 32);

    cv::Point2f targetVertices[4] = {
        cv::Point(0, outputSize.height - 1),
        cv::Point(0, 0),
        cv::Point(outputSize.width - 1, 0),
        cv::Point(outputSize.width - 1, outputSize.height - 1)
    };
    cv::Mat rotationMatrix = cv::getPerspectiveTransform(vertices, targetVertices);

    cv::warpPerspective(frame, result, rotationMatrix, outputSize);

#if 0
    imshow("roi", result);
    waitKey();
#endif
}


bool TextSpotting::sortPts(const cv::Point& p1, const cv::Point& p2)
{
    return p1.x < p2.x;
}



