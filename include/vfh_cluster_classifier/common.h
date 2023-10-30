#ifndef COMMON_H
#define COMMON_H

#include <list>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <flann/flann.h>

using namespace std;

// types definition
typedef pcl::PointXYZRGB PointType;
typedef pcl::PointXYZ DepthPointType;
typedef pcl::Normal NormalType;
typedef pcl::PointCloud<PointType> PointCloudType;
typedef pcl::PointCloud<DepthPointType> DepthPointCloudType;
typedef pcl::PointCloud<NormalType> NormalCloudType;
typedef PointCloudType::Ptr PointCloudTypePtr;
typedef DepthPointCloudType::Ptr DepthPointCloudTypePtr;
typedef PointCloudType::ConstPtr PointTConstPtr;
typedef pcl::VFHSignature308 FeatureType;
typedef pcl::PointCloud<FeatureType> FeatureCloudType;
typedef FeatureCloudType::Ptr FeatureCloudTypePtr;
typedef NormalCloudType::Ptr NormalCloudTypePtr;
typedef std::pair<std::string, std::vector<float>> vfh_model;

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

extern std::list<PointCloudTypePtr> cluster_clouds;
extern std::list<std::string> recognized_objects;
extern string found_model;

extern std::list<vfh_model> models;
extern std::list<std::string> training_objects_ids;
extern flann::Matrix<float> data;

#endif // COMMON_H
