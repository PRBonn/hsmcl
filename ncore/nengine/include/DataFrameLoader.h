/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: DataFrameLoader.h                		                               #
# ##############################################################################
**/

#ifndef DATAFRAMELOADER_H
#define DATAFRAMELOADER_H


#include <iostream>
#include "Python.h"
#include <string>
#include <vector>
#include <utility>

#include <eigen3/Eigen/Dense>
//#include "CustomEigenVec.h"



enum FrameTypes
{
	CAMERA0 = 0,
	CAMERA1 = 1,
	CAMERA2 = 2,
	CAMERA3 = 3,
	SEM0 = 4,
	SEM1 = 5,
	SEM2 = 6,
	SEM3 = 7,
	TEXT0 = 8,
	TEXT1 = 9,
	TEXT2 = 10,
	TEXT3 = 11,
	LIDAR = 12,
	ODOM = 13,
	GT = 14
};



struct FrameData
{
	FrameTypes type;
	unsigned long stamp;
	Eigen::Vector3f gt;
    std::string path;
    Eigen::Vector3f odom;
    std::vector<Eigen::Vector3f> scan;
    std::vector<std::string> places;
    std::vector<Eigen::Matrix<float, 6, 1>> boxes;
    std::vector<Eigen::Vector4f> semScan;
};

// This requires that you install pandas, for the python module to work

class DataFrameLoader
{
	public:

		//! A constructor
	    /*!
	     \param moduleFolder is a string describing the folder in which the python module for loading the .pickle is contained
	     \param moduleName is a string describing the name of the python module for loading the .pickle is contained
	    */

		DataFrameLoader(const std::string& moduleFolder, const std::string& moduleName);


		//! Loads a .pickle file with the expected data format
		/*!
		  \param picklePath is a string and it's obvious what it means
		*/
		void Load(const std::string& picklePath);

		//! Retrieves a single dataframe row from the .pickle according to its index
		/*!
		  \param index is the index of the desired row in the dataframe 
		*/
		FrameData GetData(unsigned long index);

		//! Return the total number of rows in the dataframe
		int GetNumFrames();


		~DataFrameLoader();


	private:

		const char * get_type(unsigned long ind);
		unsigned long get_stamp(unsigned long ind);
		PyObject * get_data(unsigned long ind);
		static std::vector<float> listTupleToVector_Float(PyObject* incoming); 


		PyObject *pDict;
		PyObject *dfObject;

};

#endif