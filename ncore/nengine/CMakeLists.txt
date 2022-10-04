cmake_minimum_required(VERSION 3.16.0)

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nsensors/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nmap/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann_json/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nmcl/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../ndl/include/)


set(PYTHON_INCLUDE_DIR "/usr/include/python3.8")

find_package (Python COMPONENTS Interpreter Development)
if(Python_FOUND)
  message(STATUS "Found Python version ${Python_VERSION}")
  include_directories(${PYTHON_INCLUDE_DIR})
endif(Python_FOUND)

set(RESULTS_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RESULTS_OUTPUT_DIRECTORY}/bin)




add_subdirectory("src")
#add_library(NSENSORS STATIC IMPORTED)
set_target_properties(NSENSORS PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libNSENSORS.a)
set_target_properties(NMAP PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libNMAP.a)
set_target_properties(NMAP PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libNMCL.a)
set_target_properties(NMAP PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/libNDL.a)


# After all are setup is done, we can go to our src/ directory to build our
# files
if(BUILD_TESTING)
  add_subdirectory("tst")
endif()
