cmake_minimum_required(VERSION 3.15...3.30)

project(
	FiguresHomework
	VERSION 1.0
	LANGUAGES CXX)

# Prepare doctest for other targets to use

SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_COMPILER clang)

file(GLOB_RECURSE FIGURES_SOURCES
	./src/*.cpp
)
# Make main application
add_executable(main main.cpp ${FIGURES_SOURCES})
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
#target_compile_features(main PRIVATE cxx_std_23)
target_compile_options(main PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_link_options(main PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_include_directories(main PRIVATE src/)
#target_include_directories(main PUBLIC ../lib/)

# Make encode application
add_executable(encode encode.cpp ${FIGURES_SOURCES})
set_target_properties(encode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
#target_compile_features(encode PRIVATE cxx_std_23)
target_compile_options(encode PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_link_options(encode PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_include_directories(encode PRIVATE src/)
#target_include_directories(encode PUBLIC ../lib/)

# Make decode application
add_executable(decode decode.cpp ${FIGURES_SOURCES})
set_target_properties(decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
#target_compile_features(decode PRIVATE cxx_std_23)
target_compile_options(decode PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_link_options(decode PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_include_directories(decode PRIVATE src/)
#target_include_directories(decode PUBLIC ../lib/)


# Make noisy application
add_executable(noisy noisy.cpp ${FIGURES_SOURCES})
set_target_properties(noisy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)
#target_compile_features(noisy PRIVATE cxx_std_23)
target_compile_options(noisy PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_link_options(noisy PRIVATE -fsanitize=address -std=c++23 -g -O0 -fno-inline -Wall -Wextra)
target_include_directories(noisy PRIVATE src/)
#target_include_directories(noisy PUBLIC ../lib/)

#SET(COVERAGE_FLAGS 
#	--coverage
#	-fprofile-arcs
#	-ftest-coverage
#)
#
## Make test executable
#add_executable(tests test.cpp ${FIGURES_SOURCES})
#target_compile_definitions(tests PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
#set_target_properties(tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./)
#target_compile_features(tests PRIVATE cxx_std_20)
#target_compile_options(tests PRIVATE ${COVERAGE_FLAGS} -g)
#target_link_options(tests PRIVATE ${COVERAGE_FLAGS})
#target_include_directories(tests PUBLIC ../lib/)

set(CMAKE_BUILD_TYPE Debug)


#list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
#
#include(CTest)
#enable_testing()
#add_test(NAME allTests 
#	COMMAND
#	"./tests"
#	DEPENDS tests
#)
