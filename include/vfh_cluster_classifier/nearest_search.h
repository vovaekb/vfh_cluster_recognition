#include <iostream>
#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "vfh_cluster_classifier/typedefs.h"

using namespace pcl::console;
using namespace pcl::io;

/**
 * \brief Loads an n-D histogram file as a VFH signature.
 *
 * \param path - The input file name.
 * \param vfh - The resultant VFH signature.
 * \return - True if the loading is successful, false otherwise.
 */
bool loadHist(const boost::filesystem::path &path, vfh_model &vfh)
{
    try
    {
        // Read the header of the PCD file
        pcl::PCLPointCloud2 cloud;
        pcl::PCDReader reader;
        Eigen::Vector4f origin;
        Eigen::Quaternionf orientation;
        int version;
        int type;
        unsigned int idx;
        reader.readHeader(path.string(), cloud, origin, orientation, version, type, idx);

        // Check if the "vfh" field exists and if the point cloud has only one point
        auto vfh_idx = pcl::getFieldIndex(cloud, "vfh");
        if (vfh_idx == -1 || static_cast<int>(cloud.width) * cloud.height != 1)
        {
            return false;
        }
    }
    catch (const pcl::InvalidConversionException &)
    {
        return false;
    }

    // Load the PCD file into a point cloud
    FeatureCloudType point;
    pcl::io::loadPCDFile(path.string(), point);
    vfh.second.resize(308);

    std::vector<pcl::PCLPointField> fields;
    pcl::getFieldIndex(point, "vfh", fields);

    // Copy the histogram values from the loaded point cloud
    for (size_t i = 0; i < fields[vfh_idx].count; ++i)
    {
        vfh.second[i] = point.points[0].histogram[i];
    }

    // Set the file name as the first element of the VFH signature
    vfh.first = path.string();
    return true;
}

/** \brief Load FLANN search index
 */
void loadIndex()
{
    std::cout << "Loading search index ...\n";
    string kdtree_idx_file_name = training_data_path + "/kdtree.idx";

    std::cout << "FLANN index file: " << kdtree_idx_file_name << "\n";

    // Check if the tree index has already been saved to disk
    if (!boost::filesystem::exists(kdtree_idx_file_name))
    {
        print_error("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str());
        return;
    }
    else
    {
        flann_index = new flann_distance_metric(data, flann::SavedIndexParams(kdtree_idx_file_name.c_str()));
        flann_index->buildIndex();
    }
}

/** \brief Search for the closest k neighbors
 * \param index the tree
 * \param model the query model
 * \param k the number of neighbors to search for
 * \param indices the resultant neighbor indices
 * \param distances the resultant neighbor distances
 */
inline void
nearestKSearch(flann_distance_metric &index, const vfh_model &model,
               int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
    // Query point
    flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size()], 1, model.second.size());
    memcpy(&p.ptr()[0], &model.second[0], p.cols * p.rows * sizeof(float));

    indices = flann::Matrix<int>(new int[k], 1, k);
    distances = flann::Matrix<float>(new float[k], 1, k);
    index.knnSearch(p, indices, distances, k, flann::SearchParams(512));
    delete[] p.ptr();
}
