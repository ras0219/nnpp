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

add_library(nnpp sigmoid.cpp table.cpp)

add_executable(sigmoid-test test/sigmoid-test.cpp test/test-harness.cpp)
target_link_libraries(sigmoid-test nnpp)

add_executable(table-test test/table-test.cpp test/test-harness.cpp)
target_link_libraries(table-test nnpp)

add_executable(multilayer-test test/multilayer-test.cpp test/test-harness.cpp)
target_link_libraries(multilayer-test nnpp)

add_test(sigmoid-test    sigmoid-test)
add_test(table-test      table-test)
add_test(multilayer-test multilayer-test)
