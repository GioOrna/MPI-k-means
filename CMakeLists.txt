cmake_minimum_required(VERSION 3.15)

project(ParallelKMeans VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)

set(SOURCES
	src/kmeans.cpp
	src/MPI_utils.cpp
	src/csv_utils.cpp
	src/KDTree.cpp
)

# Main executable
add_executable(main ${SOURCES} src/main.cpp)

target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE MPI::MPI_CXX)

# Test parallel clustering
add_executable(test_parallel_clustering ${SOURCES}
			   test/test_parallel_clustering.cpp)

target_include_directories(test_parallel_clustering PRIVATE include)
target_link_libraries(test_parallel_clustering PRIVATE MPI::MPI_CXX)

# Test KDTree
#add_executable(test_kdtree
#    src/KDTree.cpp        # Only KDTree implementation
#    test/test_kdtree.cpp  # test file
#)

#target_include_directories(test_kdtree PRIVATE include)
