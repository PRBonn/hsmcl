The reason things don't compile might be that vision_opencv still compiles with OpenCV 4.2.
To force compilation with OpenCV 4.5, you can edit two CMakeLists.txt files.
For [cv_bridge](https://github.com/ros-perception/vision_opencv/blob/noetic/cv_bridge/CMakeLists.txt), edit lines 20-25 as follows

```bash
set(_opencv_version 4.5)
find_package(OpenCV 4.5 QUIET)
if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4.5")
endif()
```

For [image_geometry](https://github.com/ros-perception/vision_opencv/blob/noetic/image_geometry/CMakeLists.txt), edit line 5

```bash
find_package(OpenCV 4.5 REQUIRED)
```
