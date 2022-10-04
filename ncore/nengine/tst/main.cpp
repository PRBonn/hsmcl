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



Eigen::Vector3f cam2BaselinkTF(Eigen::Vector3f pose)
{
    Eigen::Vector3f camTF(0.08, 0, 1);
    Eigen::Vector3f v(0, 0, pose(2) + M_PI / 2);

    Eigen::Matrix3f trans = Vec2Trans(v);

    Eigen::Vector3f gt = trans  * camTF;
    gt(0) += pose(0);
    gt(1) += pose(1);
    gt(2) = pose(2) + M_PI / 2;

    return gt;
}


void dumpParticles(const std::vector<Particle>& particles, std::string path)
{
    std::ofstream particleFile;
    particleFile.open(path, std::ofstream::out);
    particleFile << "x" << "," << "y" << "," << "yaw" << "," << "w" << std::endl;
    for(int p = 0; p < particles.size(); ++p)
    {
        Eigen::Vector3f pose = particles[p].pose;
        float w = particles[p].weight;
        particleFile << pose(0) << "," << pose(1) << "," << pose(2) << "," << w << std::endl;
    }
    particleFile.close();
}

void dumpScanMask(const std::vector<Eigen::Vector3f>& scan, const std::vector<double>& scanMask, std::string path)
{
    if (scan.empty()) return;

    std::ofstream scanFile;
    scanFile.open(path, std::ofstream::out);
    scanFile << "x" << "," << "y" << "," << "mask" << std::endl;
    int ds_factor = scan.size() / scanMask.size();

    for(int p = 0; p < scan.size(); ++p)
    {
        Eigen::Vector3f pnt = scan[p];
        double val = scanMask[int(p / ds_factor)];
        scanFile << pnt(0) << "," << pnt(1) << "," << val << std::endl;
    }
    scanFile.close();
}

void computeStartSecond(DataFrameLoader& df, int startFrame)
{
    FrameData fd = df.GetData(0);
    FrameData fd2 = df.GetData(startFrame);
    unsigned long diff = fd2.stamp - fd.stamp;
    std::cout << float(diff) / 1000000000 << std::endl;
    std::cout << fd2.stamp << std::endl;
}


std::vector<float>  computeStartSecond(DataFrameLoader& df, const std::vector<int>& startFrames)
{
    std::vector<float> startSeconds;
    FrameData fd = df.GetData(0);

    for(int i = 0; i < startFrames.size(); ++i)
    {    
        FrameData fd2 = df.GetData(startFrames[i]);
        unsigned long diff = fd2.stamp - fd.stamp;
        startSeconds.push_back(float(diff) / 1000000000);
        std::cout << float(diff) / 1000000000 <<  "\t";
    }
     std::cout << std::endl;

    return startSeconds;
}


std::vector<int> computeStartFrame(DataFrameLoader& df, const std::vector<float>& startSeconds)
{
    std::vector<int> startFrames;
    FrameData fd = df.GetData(0);
    
    int k = 0;
    for(int i = 0; i < startSeconds.size(); ++i)
    {    
        float time = startSeconds[i];

        for(int j = k; j < df.GetNumFrames(); ++j, ++k)
        {
            FrameData fd2 = df.GetData(j);
            unsigned long diff = fd2.stamp - fd.stamp;
            //std::cout << float(diff) / 1000000000 <<  "\t";
            if (abs(float(diff) / 1000000000  - time) < 0.01)
            {
                startFrames.push_back(j);
                std::cout << j <<  "\t";
                break;
            }
        }
       
    }
    std::cout << std::endl;

    return startFrames;
}


#include <chrono>

void singleRun(const std::string& nmclConfigFile, const std::string& sensorConfigFolder, const std::string& textMapDir,
 const std::string& resultsDir,  DataFrameLoader& df, int runID, int startFrame, int endFrame, bool text, bool sem)
{

    NMCLEngine engine(nmclConfigFile, sensorConfigFolder, textMapDir);

    if(!( boost::filesystem::exists(resultsDir)))
    {
        boost::filesystem::create_directory(resultsDir);
    }
    if(!( boost::filesystem::exists(resultsDir + "Run" + std::to_string(runID))))
    {
        boost::filesystem::create_directory(resultsDir + "Run" + std::to_string(runID));
    }

    std::string resultsFolder = resultsDir + "Run" + std::to_string(runID) + "/";


    std::string csvFilePath = resultsFolder + "poseestimation.csv";
    std::ofstream csvFile;
    csvFile.open(csvFilePath, std::ofstream::out);
    csvFile << "t" << "," << "pose_x" << "," << "pose_y" << "," << "pose_yaw" << "," << "cov_x" << "," << "cov_y" << "," << "cov_yaw"<< "," << "gt_x" << "," << "gt_y" << "," << "gt_yaw" << std::endl;

    int semcount = 0;
    int odomcount = 0;

    int numFrames = endFrame;
    Eigen::Vector3f gt;
    std::vector<Eigen::Vector3f> lidar;

    std::vector<std::vector<Eigen::Matrix<float, 6, 1>>> combinedSemScan = std::vector<std::vector<Eigen::Matrix<float, 6, 1>>>(4, std::vector<Eigen::Matrix<float, 6, 1>>());

    bool cams[] = {false, false, false, false};

     for(int i = startFrame; i < numFrames; ++i)
    {
        //std::cout << "Run " << runID << ", frame " << i << "/" << numFrames << std::endl;
        FrameData fd = df.GetData(i);
        char frame[8];
        sprintf(frame, "%07d", i);

        switch (fd.type)
        {
            case FrameTypes::SEM0: case FrameTypes::SEM1:  case FrameTypes::SEM2:  case FrameTypes::SEM3:
            {
                if(sem)
                {  
                    int camID = int(fd.type) % 4;
                    cams[camID] = true;
                    combinedSemScan[camID] = fd.boxes;
                    if(cams[0] * cams[1] * cams[2] * cams[3])
                    {
                         ++semcount;
                        //if (semcount % 20 == 0)
                        {                        
                            int ret = 0;
                            if(combinedSemScan.size())
                            {
                               ret = engine.CorrectSemantic(combinedSemScan);
                            }
                            if(ret)
                            {
                                SetStatistics stats = engine.PoseEstimation();
                                Eigen::Vector3d state = stats.Mean(); 
                                Eigen::Matrix3d cov = stats.Cov();

                                std::cout << "Semantic" << std::endl;
                               std::cout << "Run " << runID << ", frame " << i << "/" << numFrames << std::endl;
                               std::cout << "state:" << state(0) << ", " << state(1) << ", " << state(2) << std::endl;
                               std::cout << "gt   :" << gt(0) << ", " << gt(1) << ", " << gt(2) << std::endl;
                            }
                        }
                        for(int c = 0; c < 4; ++c)
                        {
                            combinedSemScan[c].clear();
                            cams[c] = false;
                        }
                    }
                }
                break;

            }
            case FrameTypes::TEXT0: case FrameTypes::TEXT1: case FrameTypes::TEXT2: case FrameTypes::TEXT3: 
            {
                if(text)
                {
                    //std::cout << "Run " << runID << ", frame " << i << "/" << numFrames << std::endl;
                    std::vector<std::string> places = fd.places;
                    engine.TextMask(places, int(fd.type) % 4);
                }
                break;
            }
            case FrameTypes::LIDAR:
             {
                std::vector<Eigen::Vector3f> scan = fd.scan;
                lidar = scan;
		        int ret = 0;
                ret = engine.Correct(scan);
                if(ret)
                {
                    SetStatistics stats = engine.PoseEstimation();
                    Eigen::Vector3d state = stats.Mean(); 
                    Eigen::Matrix3d cov = stats.Cov();

                    std::cout << "Lidar" << std::endl;
                   std::cout << "Run " << runID << ", frame " << i << "/" << numFrames << std::endl;
                   std::cout << "state:" << state(0) << ", " << state(1) << ", " << state(2) << std::endl;
                   std::cout << "gt   :" << gt(0) << ", " << gt(1) << ", " << gt(2) << std::endl;

                    csvFile << fd.stamp << "," << state(0) << "," << state(1) << "," << state(2) << "," << cov(0,0) << "," << cov(1,1) << "," << cov(2, 2) << "," << gt(0) << "," << gt(1) << "," << gt(2) << std::endl;
                    std::vector<Particle> particles = engine.Particles();
                    std::string particlesPath = resultsFolder + std::string(frame) + ".csv";
                    dumpParticles(particles, particlesPath);
                  
                }
                break;
            }
            case FrameTypes::ODOM:
            {
                Eigen::Vector3f odom = fd.odom;
                engine.Predict(odom);
                break;
            }
            case FrameTypes::GT:
            {
                gt = fd.gt;               
                break;
            }
            default:
                //std::cerr << "wrong input type!" << std::endl;
                break;
        }
    }

    csvFile.close();

}


int main(int argc, char** argv)
{
    std::string moduleFolder = "/home/nickybones/Code/OmniNMCL/ncore/nengine/src";
    std::string moduleName = "df_loader";
    std::string sensorConfigFolder = "/home/nickybones/Code/OmniNMCL/ros1_ws/src/nmcl_ros/config/480x640/";

    std::string mapType = "SFMap";
    int particles = 10000;
    std::string model = "sem";
    int sequenceID = 0;

    std::string sequences[4] = {"R1", "R2", "R3", "R4"};

    std::string nmclConfigFile = "/home/nickybones/Code/OmniNMCL/ncore/data/floor/"+ mapType + "/nmcl" + std::to_string(particles) + ".config";
    std::string textMapDir = "/home/nickybones/Code/OmniNMCL/ncore/data/floor/"+ mapType + "/TextMaps/";
    std::string nmclConfigFolder =  "/home/nickybones/Code/OmniNMCL/ncore/data/floor/"+ mapType + "/";


    bool text = false;
    bool sem = false;
    std::string expName = "sem_" + std::to_string(sem);
    std::string picklePath = "/home/nickybones/data/MCL/2022_05_09/" + sequences[sequenceID] + "/2022_05_09_" + sequences[sequenceID]+ ".pickle";
    std::string resultsDir = "/home/nickybones/data/MCL/2022_05_09/" + sequences[sequenceID] + "/" + mapType + "/" + expName + "/";

    DataFrameLoader df(moduleFolder, moduleName);
    df.Load(picklePath);
    int numFrames = df.GetNumFrames();
    // std::vector<int> startFrames = {5094,  7552, 11844, 12283, 14006, 15413, 22851, 26379, 29219, 32338};
    // int endFrame = 62705;
    //std::vector<int> startFrames = {7854,    11733,   18378,   19059,   21717,   23926,   35594,   40922,   44991,   49865};
    //int endFrame = 97260;
    //int endFrame = 95000;

    std::vector<int> startFrames = {0};
    int endFrame = numFrames;

    //TestBoxes(nmclConfigFolder, sensorConfigFolder, textMapDir, df);

   for(int i = 0; i < startFrames.size(); ++i)
    {
        srand48(21);
        singleRun(nmclConfigFile, sensorConfigFolder, textMapDir, resultsDir, df, i, startFrames[i], endFrame, text, sem);
    }

    return 0;
}

