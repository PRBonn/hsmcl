/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: NDLUnitTests.cpp                                                     #
# ##############################################################################
**/

#include "gtest/gtest.h"
#include "TextSpotting.h"
//#include "PanopticSegmentation.h"
//#include "ImageUtils.h"
#include <algorithm>

std::string textDataPath = PROJECT_TEST_DATA_DIR + std::string("/test/text/");


TEST(TextSpotting, test1) {

    std::string imgPath = textDataPath + "traffic.png";
    float binThresh = 0.3;
    float polyThresh = 0.5;
    uint maxCandidates = 50;
    std::string detModelPath = textDataPath + "DB_IC15_resnet18.onnx";
    std::string recModelPath = textDataPath + "crnn_cs.onnx";
    std::string vocPath = textDataPath + "alphabet_94.txt";
    double unclipRatio = 2.0;
    int height = 480;
    int width = 640;

    cv::Mat frame = cv::imread(imgPath);
    TextSpotting ts = TextSpotting(detModelPath,recModelPath, vocPath, height, width,
    binThresh, polyThresh, maxCandidates, unclipRatio);
    std::vector<std::string> text = ts.Infer(frame);
    ASSERT_EQ(text[11], "Bonn");

}

TEST(TextSpotting, test2) {

    std::string imgPath = textDataPath + "traffic.png";
    cv::Mat frame = cv::imread(imgPath);
    TextSpotting ts = TextSpotting(textDataPath + "textspotting.config");
    std::vector<std::string> text = ts.Infer(frame);
    ASSERT_EQ(text[11], "Bonn");

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


