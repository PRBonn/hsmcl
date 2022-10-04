
add_executable(TextRecoFromFolder TextRecoFromFolder.cpp TextSpotting.cpp)
target_link_libraries(TextRecoFromFolder ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Boost_LIBRARIES} )

add_executable(TextRecoFromFolder2 TextRecoFromFolder2.cpp TextSpotting.cpp)
target_link_libraries(TextRecoFromFolder2 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${Boost_LIBRARIES} )


add_library(NDL TextSpotting.cpp )
#target_link_libraries(NDL PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(NDL ${OpenCV_LIBS})




