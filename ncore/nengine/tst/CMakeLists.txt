add_executable(NEngineTests main.cpp)
target_link_libraries(NEngineTests NEGNINE ${Python_LIBRARIES} ${OpenCV_LIBS} NSENSORS NMAP NMCL NDL nlohmann_json::nlohmann_json  ${Boost_LIBRARIES})

add_executable(Stability Stability.cpp)
target_link_libraries(Stability NEGNINE ${Python_LIBRARIES} ${OpenCV_LIBS} NSENSORS NMAP NMCL NDL nlohmann_json::nlohmann_json  ${Boost_LIBRARIES})

#add_executable(EngineBenchmark EngineBenchmark.cpp)
#target_link_libraries(EngineBenchmark NEGNINE ${Python_LIBRARIES} ${OpenCV_LIBS} NSENSORS NMAP NMCL NDL nlohmann_json::nlohmann_json  ${Boost_LIBRARIES})

