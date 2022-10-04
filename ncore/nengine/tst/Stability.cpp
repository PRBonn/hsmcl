#include <iostream>
#include "Python.h"
#include <string>
#include <vector>
#include <fstream>

#include "DataFrameLoader.h"
#include "NMCLEngine.h"
#include "Utils.h"
#include <math.h>
#include <boost/filesystem.hpp>


int main(int argc, char** argv)
{
	std::string mapType = "SMap";
    int particles = 10000;

    std::string moduleFolder = "/home/nickybones/Code/OmniNMCL/ncore/nengine/src";
    std::string moduleName = "df_loader";
    std::string sensorConfigFolder = "/home/nickybones/Code/OmniNMCL/ros1_ws/src/nmcl_ros/config/480x640/";
    std::string nmclConfigFile = "/home/nickybones/Code/OmniNMCL/ncore/data/floor/"+ mapType + "/nmcl" + std::to_string(particles) + ".config";
    std::string textMapDir = "/home/nickybones/Code/OmniNMCL/ncore/data/floor/"+ mapType + "/TextMaps/";

    std::string sequences[10] = {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"};
    
    NMCLEngine engine(nmclConfigFile, sensorConfigFolder, textMapDir);
    DataFrameLoader df(moduleFolder, moduleName);

    for(int sequenceID = 0; sequenceID < 10; ++sequenceID)
    {
        std::cout << "sequenceID " << sequenceID  << std::endl;
        std::string picklePath = "/home/nickybones/data/MCL/icra2023/" + sequences[sequenceID] + "/icra2023_" + sequences[sequenceID]+ ".pickle";

        df.Load(picklePath);
        int numFrames = df.GetNumFrames();
        int startFrame = 0;
        Eigen::Vector3f gt;
        bool firstgt = false;
        bool cams[] = {false, false, false, false};
        std::vector<std::vector<Eigen::Matrix<float, 6, 1>>> combinedSemScan = std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>(4, std::vector<Eigen::Matrix<float, 6, 1>>());

        for(int i = startFrame; i < numFrames; ++i)
        {
            FrameData fd = df.GetData(i);
            char frame[8];
            sprintf(frame, "%07d", i);

            switch (fd.type)
            {
                case FrameTypes::SEM0: case FrameTypes::SEM1:  case FrameTypes::SEM2:  case FrameTypes::SEM3:
                {
                	int camID = int(fd.type) % 4;
                    cams[camID] = true;
                    combinedSemScan[camID] = fd.boxes;
                    if(cams[0] * cams[1] * cams[2] * cams[3])
                    {
                      	if(combinedSemScan.size() && firstgt)
                        {
                        	Particle particle(gt);
                            engine.UpdateConsistency(particle, combinedSemScan);
                        }

                        for(int c = 0; c < 4; ++c)
                        {
                            combinedSemScan[c].clear();
                            cams[c] = false;
                        }
                    }
                }
                case FrameTypes::GT:
                {
                    gt = fd.gt; 
                    firstgt = true;              
                    break;
                }
                default:
                    //std::cerr << "wrong input type!" << std::endl;
                    break;
            }
        }
    }

    std::vector<std::string> classNames = {"sink", "door", "oven", "whiteboard", "table", "cardboard", "plant", "drawers", "sofa", "storage", "chair", "extinguisher", "people", "desk"};
    std::vector<Eigen::Vector2f> classConsistency = engine.ClassConsistency();

    for(int c = 0; c < classConsistency.size(); ++c)
    {
        if (classConsistency[c](1) > 0)
        {
        	float cons = classConsistency[c](0) / classConsistency[c](1);
        	std::cout << classNames[c] << " consistency: " << 100 * cons << "% , "  << classConsistency[c](0) << "\\" << classConsistency[c](1) << std::endl;
        }
        else
        {
            std::cout << "class " << c << " not included" << std::endl;
        }
    }

}