/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: DataFrameLoader.cpp                                                    #
# ##############################################################################
**/

#include "DataFrameLoader.h"
#include "Utils.h"


DataFrameLoader::DataFrameLoader(const std::string& moduleFolder, const std::string& moduleName)
{
	Py_Initialize();

	PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString(moduleFolder.c_str()));

    // Load the module
    PyObject *pName = PyUnicode_FromString(moduleName.c_str());
    PyObject *pModule = PyImport_Import(pName);

    if (pModule != NULL) 
    {
        std::cout << "Python module found\n";

        pDict = PyModule_GetDict(pModule);
    }
    else
    {
    	std::cerr << "Python module not found!\n";
    }
}


DataFrameLoader::~DataFrameLoader()
{
	Py_Finalize();
}

void DataFrameLoader::Load(const std::string& picklePath)
{
	PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("load_pickle"));
    if(pFunc != NULL)
    {
        PyObject * arglist = Py_BuildValue("(s)", picklePath.c_str());
        PyObject *pValue = PyObject_CallObject(pFunc, arglist);
        if (pValue != NULL) 
        {
            dfObject = pValue;
        }
        else
        {
            std::cout << "df is null\n";
           // return NULL;
        }
    }
    else 
    {
        std::cout << "Couldn't find func\n";
       // return NULL;
    }
}

int DataFrameLoader::GetNumFrames()
{
    PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("num_frames"));
    if(pFunc != NULL)
    {
        PyObject * arglist = Py_BuildValue("(O)", dfObject);
        PyObject *pValue = PyObject_CallObject(pFunc, arglist);
        if (pValue != NULL) 
        {
            unsigned long num = PyLong_AsLong(pValue);
            return num;
        }
    
    }
    else 
    {
        std::cout << "Couldn't find func\n";
    }
    return -1;
}

FrameData DataFrameLoader::GetData(unsigned long index)
{
	std::string type = std::string(get_type(index));
    unsigned long t = get_stamp(index);

    FrameData fd;
    //fd.type = std::string(type);
    fd.stamp = t;

    PyObject * data = get_data(index);
    std::size_t foundCam = type.find("camera");
    std::size_t foundSem = type.find("sem");
    std::size_t foundText = type.find("text");

    if(foundCam != std::string::npos)
    {
        Py_ssize_t size;
        const char *str = PyUnicode_AsUTF8AndSize(data, &size);
        fd.path = str;
        if (type == "camera0") fd.type = FrameTypes::CAMERA0;
        else if (type == "camera1") fd.type = FrameTypes::CAMERA1;
        else if (type == "camera2") fd.type = FrameTypes::CAMERA2;
        else if (type == "camera3") fd.type = FrameTypes::CAMERA3;

    }
    else if(foundSem != std::string::npos)
    {
        std::vector<float> semVec = listTupleToVector_Float(data);

        // int boxNum = int(semVec.size() / 4);
        // std::vector<Eigen::Vector4f> boxes(boxNum);
        // for(int i = 0; i < boxNum; ++i)
        // {
        //     boxes[i] = Eigen::Vector4f(semVec[4 * i], semVec[4 * i + 1], semVec[4 * i + 2], semVec[4 * i + 3]);
        // }

        // fd.semScan = boxes;

        int vecSize = 6;
         int boxNum = int(semVec.size() / vecSize);
        std::vector<Eigen::Matrix<float, 6, 1>> boxes(boxNum);
        for(int i = 0; i < boxNum; ++i)
        {
            Eigen::Matrix<float, 6, 1> b = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(&semVec[vecSize * i], vecSize);
            //Vector6f b = {semVec[6 * i], semVec[6 * i + 1], semVec[6 * i + 2], semVec[6 * i + 3], semVec[6 * i + 4], semVec[6 * i + 5]};
            boxes[i] = b;
        }

        fd.boxes = boxes;



        if (type == "sem0") fd.type = FrameTypes::SEM0;
        else if (type == "sem1") fd.type = FrameTypes::SEM1;
        else if (type == "sem2") fd.type = FrameTypes::SEM2;
        else if (type == "sem3") fd.type = FrameTypes::SEM3;

    }
    else if(foundText != std::string::npos)
    {
        Py_ssize_t size;
        const char *str = PyUnicode_AsUTF8AndSize(data, &size);
        std::string s(str);
        std::string delimiter = ", ";

        size_t pos = 0;
        std::string token;
        std::vector<std::string> places;
        while ((pos = s.find(delimiter)) != std::string::npos) 
        {
            token = s.substr(0, pos);
            places.push_back(token);
            s.erase(0, pos + delimiter.length());
        }

        fd.places = places;
        if (type == "text0") fd.type = FrameTypes::TEXT0;
        else if (type == "text1") fd.type = FrameTypes::TEXT1;
        else if (type == "text2") fd.type = FrameTypes::TEXT2;
        else if (type == "text3") fd.type = FrameTypes::TEXT3;

    }
    else if(type == "odom")
    {
        std::vector<float> odomVec = listTupleToVector_Float(data); 
        float yaw = GetYaw(odomVec[5], odomVec[6]);
        Eigen::Vector3f odom(odomVec[0], odomVec[1], yaw);
        fd.odom = odom;
        fd.type = FrameTypes::ODOM;
    }
    else if(type == "gt")
    {
        std::vector<float> gtVec = listTupleToVector_Float(data); 
        Eigen::Vector3f gt(gtVec[0], gtVec[1], gtVec[2]);
        fd.gt = gt;
        fd.type = FrameTypes::GT;
    }
    else if(type == "lidar")
    {
        std::vector<float> scanVec = listTupleToVector_Float(data); 
        std::vector<Eigen::Vector3f> scan(scanVec.size() / 2);
        for(int i = 0; i < scanVec.size() / 2; ++i)
        {
            Eigen::Vector3f p(scanVec[2 * i], scanVec[2 * i + 1], 1.0);
            scan[i] = p;
        }
        fd.scan = scan;
        fd.type = FrameTypes::LIDAR;
    }
    else
    {
        //std::cerr << "no type detected! " << type << std::endl; 
    }


    return fd;

}


const char * DataFrameLoader::get_type(unsigned long ind)
{
    PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("get_type"));
    if(pFunc != NULL)
    {
        PyObject * arglist = Py_BuildValue("(O, l)", dfObject, ind); 
        PyObject *pValue = PyObject_CallObject(pFunc, arglist);  
        if (pValue != NULL) 
        {
             Py_ssize_t size;
            const char *str = PyUnicode_AsUTF8AndSize(pValue, &size);
            return str;
        }      
    }
    else 
    {
        std::cout << "Couldn't find func\n";
    }
    return NULL;
}

unsigned long DataFrameLoader::get_stamp(unsigned long ind)
{
    PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("get_stamp"));
    if(pFunc != NULL)
    {
        PyObject * arglist = Py_BuildValue("(O, l)", dfObject, ind); 
        PyObject *pValue = PyObject_CallObject(pFunc, arglist);  
        if (pValue != NULL) 
        {
            unsigned long t = PyLong_AsLong(pValue);
            return t;
        }      
    }
    else 
    {
        std::cout << "Couldn't find func\n";
    }
    return NULL;
}



std::vector<float> DataFrameLoader::listTupleToVector_Float(PyObject* incoming) 
{
    std::vector<float> data;
    if (PyTuple_Check(incoming)) 
    {
        for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) 
        {
            PyObject *value = PyTuple_GetItem(incoming, i);
            data.push_back( PyFloat_AsDouble(value) );
        }
    } 
    else 
    {
        if (PyList_Check(incoming)) 
        {
            for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) 
            {
                PyObject *value = PyList_GetItem(incoming, i);
                data.push_back( PyFloat_AsDouble(value) );
            }
        } 
        else 
        {
            throw std::logic_error("Passed PyObject pointer was not a list or tuple!");
        }
    }
    return data;
}


PyObject * DataFrameLoader::get_data(unsigned long ind)
{
    PyObject *pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("get_data"));
    if(pFunc != NULL)
    {
        PyObject * arglist = Py_BuildValue("(O, l)", dfObject, ind); 
        PyObject *pValue = PyObject_CallObject(pFunc, arglist);  
        if (pValue != NULL) 
        {
            return pValue;
        }      
    }
    else 
    {
        std::cout << "Couldn't find func\n";
    }
    return NULL;
}
