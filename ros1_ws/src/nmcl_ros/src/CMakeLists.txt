





add_executable(ScanMergeNode ScanMergeNode.cpp RosUtils.cpp)
target_link_libraries(ScanMergeNode ${OpenCV_LIBS} ${catkin_LIBRARIES} NSENSORS)


add_executable(TextRecoNode TextRecoNode.cpp)
target_link_libraries(TextRecoNode ${OpenCV_LIBS} ${catkin_LIBRARIES} NDL ${TORCH_LIBRARIES})


add_executable(ConfigNMCLNode ConfigNMCLNode.cpp RosUtils.cpp)
target_link_libraries(ConfigNMCLNode ${OpenCV_LIBS} ${catkin_LIBRARIES} NMCL NSENSORS NMAP ${Boost_LIBRARIES} nlohmann_json::nlohmann_json)

