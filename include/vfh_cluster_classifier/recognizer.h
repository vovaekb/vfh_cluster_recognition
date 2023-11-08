#define REFACTOR_DEBUG
// #define SEGMENTATION_DEBUG

#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <iostream>
#include <vector>
#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/vfh.h>
#include <pcl/features/crh.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <boost/filesystem.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "typedefs.h"

using namespace std;

// Required for saving CRH histogram to PCD file
POINT_CLOUD_REGISTER_POINT_STRUCT(CRH90,
                                  (float[90], histogram, histogram90))

// TODO: Replace index_score with ObjectHypothesis
struct index_score
{
    string model_id;
    double score;
};

// A struct for storing alignment results
struct ObjectHypothesis
{
    std::string model_id;
    PointCloudTypePtr model_template;
    float icp_score;
    Eigen::Matrix4f transformation;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// vector<string> best_candidate_names;
// vector<float> best_candidate_scores;

/** \brief Load the list of file model names from an ASCII file
 * \param models the resultant list of model name
 * \param filename the input file name
 */
bool loadFileList(vector<vfh_model> &models, const string &filename);

/** \brief Load FLANN search index
 */
void loadIndex();

/** \brief Search for the closest k neighbors
 * \param index the tree
 * \param model the query model
 * \param k the number of neighbors to search for
 * \param indices the resultant neighbor indices
 * \param distances the resultant neighbor distances
 */
inline void
nearestKSearch(flann::Index<flann::ChiSquareDistance<float>> &index, const vfh_model &model,
               int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

/** \brief Loads an n-D histogram file as a VFH signature
 * \param index the index of input cluster cloud
 * \param vfh the resultant VFH model
 */
bool loadHist(const int &index, vfh_model &vfh);

void createHist(PointCloudTypePtr &cloud, FeatureCloudType::Ptr &descriptor, CRHCloudTypePtr &crh_histogram, Eigen::Vector4f &centroid);

void preprocessCloud(PointCloudTypePtr &input, PointCloudTypePtr &output);

void segmentScene(PointCloudTypePtr &cloud);

void classifyCluster(const int &ind, PointCloudTypePtr &cloud);

void recognize(PointCloudTypePtr &cloud, PointCloudTypePtr &cloud_filtered);

void clearData();

#endif // RECOGNIZER_H
