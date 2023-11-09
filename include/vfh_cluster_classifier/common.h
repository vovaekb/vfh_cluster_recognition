#ifndef COMMON_H
#define COMMON_H

#include <list>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <flann/flann.h>

#include "typedefs.h"

using namespace std;

extern string base_descr_dir;

// Algorithm parameters
extern float distance_thresh;
extern bool perform_crh;
extern bool apply_thresh;
extern int nn_k;
extern int thresh; // inlier distance threshold

extern string training_data_path;
extern string gt_files_dir;
extern string gt_file_path;
extern string test_scenes_dir;
extern string test_scene;
extern string scene_name;

extern std::list<PointCloudPtr> cluster_clouds;
extern std::list<std::string> recognized_objects;
extern string found_model;

extern std::list<vfh_model> models;
extern std::list<std::string> training_objects_ids;
extern flann::Matrix<float> data;

#endif // COMMON_H
