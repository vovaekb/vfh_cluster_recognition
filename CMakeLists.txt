cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_FLAGS "-g")

set (CMAKE_CXX_STANDARD 17)

# we need FindFLANN.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

project(vfh_cluster_classifier)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


#file(GLOB vfh_cluster_classifier_SRC
#    "src/*.h"
#    "src/recognizer.cpp"
#    "src/recognize_testing.cpp"
#)

add_library(feature_cloud src/feature_cloud.cpp)
add_library(recognizer src/recognizer.cpp)
add_library(test_runner src/test_runner.cpp)

find_package(HDF5)
if(HDF5_FOUND)
  find_package(FLANN)
  include_directories(${FLANN_INCLUDE_DIRS})
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

  include_directories(${HDF5_INCLUDE_DIR})

  add_executable(build_tree src/build_tree.cpp)
  add_executable(nearest_neighbors src/nearest_neighbors.cpp)
  add_executable(recognize_testing src/recognize_testing.cpp)
  add_executable(view_preprocess src/view_preprocess.cpp)


  target_link_libraries(build_tree 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${FLANN_LIBRARIES} 
    ${HDF5_hdf5_LIBRARY}
  )

  target_link_libraries(nearest_neighbors
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES} 
    ${FLANN_LIBRARIES} 
    ${HDF5_hdf5_LIBRARY}
  )

  target_link_libraries(feature_cloud
    ${catkin_LIBRARIES}
  )

  target_link_libraries(recognizer
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${FLANN_LIBRARIES} 
    ${HDF5_hdf5_LIBRARY} 
    feature_cloud
  )

  target_link_libraries(test_runner 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    recognizer
  )

  target_link_libraries(recognize_testing 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    ${FLANN_LIBRARIES} 
    ${HDF5_hdf5_LIBRARY} 
    recognizer test_runner
  )

  target_link_libraries(
    view_preprocess 
    ${PCL_LIBRARIES}
  )
endif(HDF5_FOUND)
