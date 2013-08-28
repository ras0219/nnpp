cmake_minimum_required (VERSION 2.6)
project (nnpp)

enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo/build_aux/cmake/Modules/")

add_subdirectory(lib/armadillo)

include_directories(lib/armadillo/include)
include_directories(include)

set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Build the library
add_library(nnpp src/sigmoid.cpp src/table.cpp src/fanout.cpp)

# Test executables
add_executable(sigmoid-test test/sigmoid-test.cpp test/test-harness.cpp)
target_link_libraries(sigmoid-test nnpp)

add_executable(table-test test/table-test.cpp test/test-harness.cpp)
target_link_libraries(table-test nnpp)

add_executable(multilayer-test test/multilayer-test.cpp test/test-harness.cpp)
target_link_libraries(multilayer-test nnpp)

add_executable(multilayer-table-test test/multilayer-table-test.cpp test/test-harness.cpp)
target_link_libraries(multilayer-table-test nnpp)

add_executable(fanout-test test/fanout-test.cpp test/test-harness.cpp)
target_link_libraries(fanout-test nnpp)

add_executable(multilayer-fanout-test test/multilayer-fanout-test.cpp test/test-harness.cpp)
target_link_libraries(multilayer-fanout-test nnpp)

# Tests
add_test(sigmoid-test    sigmoid-test)
add_test(table-test      table-test)
add_test(multilayer-test multilayer-test)
add_test(multilayer-table-test multilayer-table-test)
add_test(fanout-test fanout-test)
add_test(multilayer-fanout-test multilayer-fanout-test)

# Samples
add_executable(germanic samples/germanic.cpp)
target_link_libraries(germanic nnpp)
file(COPY samples/data DESTINATION .)
