// #define VFH_COMPUTE_DEBUG
// #define DISABLE_COMPUTING_CRH

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/vfh.h>
#include <pcl/features/crh.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <fstream>

#include "vfh_cluster_classifier/persistence_utils.h"
#include "vfh_cluster_classifier/nearest_search.h"

using namespace std;
using namespace pcl::console;
using namespace pcl::io;
namespace fs = boost::filesystem;

bool calculate_crh(false);
bool calculate_vfh(false);

#ifndef DISABLE_COMPUTING_CRH
// Required for saving CRH histogram to PCD file
POINT_CLOUD_REGISTER_POINT_STRUCT(CRH90,
                                  (float[90], histogram, histogram90))
#endif

string training_dir;

auto voxel_leaf_size{0.005};
auto normal_radius{0.03};

// /**
//  * \brief Loads an n-D histogram file as a VFH signature.
//  *
//  * \param path - The input file name.
//  * \param vfh - The resultant VFH signature.
//  * \return - True if the loading is successful, false otherwise.
//  */
// bool loadHist(const fs::path &path, vfh_model &vfh)
// {
//     try
//     {
//         // Read the header of the PCD file
//         pcl::PCLPointCloud2 cloud;
//         pcl::PCDReader reader;
//         Eigen::Vector4f origin;
//         Eigen::Quaternionf orientation;
//         int version;
//         int type;
//         unsigned int idx;
//         reader.readHeader(path.string(), cloud, origin, orientation, version, type, idx);

//         // Check if the "vfh" field exists and if the point cloud has only one point
//         int vfh_idx = pcl::getFieldIndex(cloud, "vfh");
//         if (vfh_idx == -1 || static_cast<int>(cloud.width) * cloud.height != 1)
//         {
//             return false;
//         }
//     }
//     catch (const pcl::InvalidConversionException &)
//     {
//         return false;
//     }

//     // Load the PCD file into a point cloud
//     FeatureCloudType point;
//     loadPCDFile(path.string(), point);
//     vfh.second.resize(308);

//     std::vector<pcl::PCLPointField> fields;
//     pcl::getFieldIndex(point, "vfh", fields);

//     // Copy the histogram values from the loaded point cloud
//     for (size_t i = 0; i < fields[vfh_idx].count; ++i)
//     {
//         vfh.second[i] = point.points[0].histogram[i];
//     }

//     // Set the file name as the first element of the VFH signature
//     vfh.first = path.string();
//     return true;
// }

void loadFeatureModels(const fs::path &base_dir, const std::string &extension,
                       std::vector<vfh_model> &models)
{
    cout << "[loadFeatureModels]\n";
    if (!fs::exists(base_dir) && !fs::is_directory(base_dir))
        return;

    for (fs::directory_iterator it(base_dir); it != fs::directory_iterator(); ++it)
    {
        if (fs::is_directory(it->status()))
        {
            std::stringstream ss;
            ss << it->path();
            print_highlight("Loading %s (%lu models loaded so far).\n", ss.str().c_str(), (unsigned long)models.size());
            loadFeatureModels(it->path(), extension, models);
        }
        if (fs::is_regular_file(it->status()) && fs::extension(it->path()) == extension)
        {
            vfh_model m;
            if (loadHist(base_dir / it->path().filename(), m))
                models.push_back(m);
            m.second.clear();
        }
    }
}

void processCloud(PointCloudPtr &in, PointCloudPtr &out)
{
    vector<int> mapping;
    pcl::removeNaNFromPointCloud(*in, *in, mapping);

    // Downsampling
    pcl::VoxelGrid<PointType> vox_grid;
    vox_grid.setInputCloud(in);
    vox_grid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    // TODO: use std::shared_ptr
    std::shared_ptr<PointCloudType>... = std::make_shared<PointCloudType>();
    PointCloudPtr temp_cloud(new PointCloudType());
    vox_grid.filter(*temp_cloud);

    out = temp_cloud;
}

void createFeatureModels(const fs::path &base_dir, const std::string &extension)
{
    cout << "[createFeatureModels] Loading files in directory: " << base_dir << "\n";

    if (!fs::exists(base_dir) && !fs::is_directory(base_dir))
        return;

    for (fs::directory_iterator it(base_dir); it != fs::directory_iterator(); ++it)
    {
        cout << "Process " << it->path().filename() << "...\n";
        if (fs::is_directory(it->status()))
        {
            std::stringstream ss;
            ss << it->path();
            //        print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
            createFeatureModels(it->path(), extension);
        }

        string file_name = (it->path().filename()).string();

        if (fs::is_regular_file(it->status()) && fs::extension(it->path()) == extension && !strstr(file_name.c_str(), "vfh") && !strstr(file_name.c_str(), "crh"))
        {
            std::vector<string> strs;
            boost::split(strs, file_name, boost::is_any_of("."));
            string view_id = strs[0];
            strs.clear();


            string descr_file = PersistenceUtils::getModelDescriptorFileName(base_dir, view_id);

            if (!fs::exists(descr_file))
            {
                PointCloudPtr view(new PointCloudType());

                string full_file_name = it->path().string(); // (base_dir / it->path ().filename ()).string();
                //          string file_name = (it->path ().filename ()).string();

                cout << "Compute VFH for " << full_file_name << "\n";

                loadPCDFile(full_file_name.c_str(), *view);

                cout << "Cloud has " << view->points.size() << " points\n";

                // Preprocess view cloud
                processCloud(view, view);

                cout << "Cloud has " << view->points.size() << " points after processing\n";

                NormalCloudTypePtr normals(new NormalCloudType());
                FeatureCloudTypePtr descriptor(new FeatureCloudType);

                // Estimate the normals.
                pcl::NormalEstimation<PointType, NormalType> normalEstimation;
                normalEstimation.setInputCloud(view);

                normalEstimation.setRadiusSearch(normal_radius);
                pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
                normalEstimation.setSearchMethod(kdtree);

                // Alternative from local pipeline
                //              int norm_k = 10;
                //              normalEstimation.setKSearch(norm_k);
                normalEstimation.compute(*normals);

                // VFH estimation object.
                pcl::VFHEstimation<PointType, NormalType, FeatureType> vfh;
                vfh.setInputCloud(view);
                vfh.setInputNormals(normals);
                vfh.setSearchMethod(kdtree);
                // Optionally, we can normalize the bins of the resulting histogram,
                // using the total number of points.
                vfh.setNormalizeBins(true);
                // Also, we can normalize the SDC with the maximum size found between
                // the centroid and any of the cluster's points.
                vfh.setNormalizeDistance(false);

                vfh.compute(*descriptor);

                cout << "VFH descriptor has size: " << descriptor->points.size() << "\n";

                savePCDFileBinary(descr_file.c_str(), *descriptor);
                cout << descr_file << " was saved\n";

#ifndef DISABLE_COMPUTING_CRH
                if (calculate_crh)
                {
                    std::cout << "Compute CRH features ...\n";
                    // Compute the CRH histogram
                    CRHCloudType::Ptr histogram(new CRHCloudType);

                    // CRH estimation object
                    CRHEstimationPtr crh;
                    crh.setInputCloud(view);
                    crh.setInputNormals(normals);
                    Eigen::Vector4f centroid4f;
                    pcl::compute3DCentroid(*view, centroid4f);
                    crh.setCentroid(centroid4f);

                    crh.compute(*histogram);

                    // Save centroid to file
                    auto centroid_file = PersistenceUtils::getCentroidFileName(base_dir, view_id);
                    Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);

                    // TODO: Move to the PersistenceUtils class
                    std::ofstream out(centroid_file.c_str());
                    if (!out)
                    {
                        std::cout << "Failed to open file " << centroid_file << " for saving centroid\n";
                    }

                    out << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
                    out.close();

                    std::cout << centroid_file << " was saved\n";

                    auto roll_file = PersistenceUtils::getCRHDescriptorFileName(base_dir, view_id);

                    savePCDFileBinary(roll_file.c_str(), *histogram);
                    cout << roll_file << " was saved\n";
                }
#endif
            }
            else
            {
                print_highlight("Descriptor file %s already exists\n", descr_file.c_str());
            }
        }
    }
}

void showHelp(char *filename)
{
    std::cout << "*****************************************************************\n"
              << "*                                                               *\n"
              << "*           VFH Cluster classifier: Build tree                  *\n"
              << "*                                                               *\n"
              << "*****************************************************************\n"
              << "Usage: " << filename << " [model_directory] [options]\n"
              << "*options:\n"
              << "-vfh         - calculate VFH features\n"
              << "-crh         - calculate CRH features\n\n";
}

void parseCommandLine(int argc, char **argv)
{
    if (argc < 2)
    {
        PCL_ERROR("Need at least two parameters!\n");
        showHelp(argv[0]);
        exit(-1);
    }

    training_dir = string(argv[1]);

    if (find_switch(argc, argv, "-vfh"))
    {
        calculate_vfh = true;
    }

    if (find_switch(argc, argv, "-crh"))
    {
        calculate_crh = true;
    }

    if (find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }
}

int main(int argc, char **argv)
{
    parseCommandLine(argc, argv);

    cout << "Calculate VFH: " << calculate_vfh << "\n";
    cout << "Calculate CRH: " << calculate_crh << "\n";

    cout << "Training dir: " << training_dir << "\n";

    std::string extension(".pcd");
    transform(extension.begin(), extension.end(), extension.begin(), (int (*)(int))tolower);

    std::string kdtree_idx_file_name = training_dir + "/kdtree.idx";
    std::string training_data_h5_file_name = training_dir + "/training_data.h5";
    std::string training_data_list_file_name = training_dir + "/training_data.list";

    std::vector<vfh_model> models;

    // Remove previously saved flann index and data files
    if (fs::exists(kdtree_idx_file_name))
    {
        if (remove(kdtree_idx_file_name.c_str()) != 0)
            perror("Error deleting old flann index file");
        else
            cout << "Old flann index file was successfully deleted\n";
    }

    if (fs::exists(training_data_h5_file_name))
    {
        if (remove(training_data_h5_file_name.c_str()) != 0)
            perror("Error deleting old training data file");
        else
            cout << "Old training data file was successfully deleted\n";
    }

    if (calculate_vfh)
        createFeatureModels(argv[1], extension);

#ifndef VFH_COMPUTE_DEBUG
    // Load the model histograms
    loadFeatureModels(argv[1], extension, models);
    print_highlight("Loaded %d VFH models. Creating training data %s/%s.\n",
                                  static_cast<int>(models.size()), training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());

    // Convert data into FLANN format
    flann::Matrix<float> data(new float[models.size() * models[0].second.size()], models.size(), models[0].second.size());

    for (size_t i = 0; i < data.rows; ++i)
    {
        for (size_t j = 0; j < data.cols; ++j)
        {
            data[i][j] = models[i].second[j];
        }
    }

    // Save data to disk (list of models)
    flann::save_to_file(data, training_data_h5_file_name, "training_data");
    std::ofstream fs;
    fs.open(training_data_list_file_name.c_str());
    for (auto &model : models)
    {
        fs << model.first << "\n";
    }
    fs.close();

    models.clear();

    // Build the tree index and save it to disk
    print_error("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str(), static_cast<int>(data.rows));
    flann_distance_metric index(data, flann::LinearIndexParams());
    // flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
    index.buildIndex();
    index.save(kdtree_idx_file_name);
    delete[] data.ptr();
#endif

    return (0);
}
