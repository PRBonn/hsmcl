


add_executable(NDLUnitTests NDLUnitTests.cpp)
target_link_libraries(NDLUnitTests ${OpenCV_LIBS} NDL GTest::GTest gtest_main )
target_compile_definitions(NDLUnitTests PRIVATE PROJECT_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
add_test(AllTestsInTests NDLUnitTests)









