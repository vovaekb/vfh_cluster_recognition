
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
#include <memory>

#include "vfh_cluster_classifier/persistence_utils.h"
#include "vfh_cluster_classifier/nearest_search.h"

using namespace std;
using namespace pcl::console;
using namespace pcl::io;
namespace fs = boost::filesystem;

using utils = PersistenceUtils;

bool calculate_crh(false);
bool calculate_vfh(false);

// Required for saving CRH histogram to PCD file
POINT_CLOUD_REGISTER_POINT_STRUCT(CRH90,
                                  (float[90], histogram, histogram90))

string training_dir;

auto voxel_leaf_size{0.005};
auto normal_radius{0.03};


void loadFeatureModels(const fs::path &base_dir, const std::string &extension,
                       std::vector<vfh_model> &models)
{
    cout << "[loadFeatureModels]\n";
    if (!fs::exists(base_dir) && !fs::is_directory(base_dir))
        return;

    fs::directory_iterator end_it();
    for (fs::directory_iterator it(base_dir); it != end_it; ++it)
    {
        if (fs::is_directory(it->status()))
        {
            std::stringstream ss;
            ss << it->path();
            print_highlight("Loading %s (%lu models loaded so far).\n", ss.str().c_str(), (unsigned long)models.size());
            loadFeatureModels(it->path(), extension, models);
        }
        if (fs::is_regular_file(it->status()) &&
         fs::extension(it->path()) == extension)
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
    VoxelGrid vox_grid;
    vox_grid.setInputCloud(in);
    vox_grid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    PointCloudPtr temp_cloud(new PointCloudType());
    vox_grid.filter(*temp_cloud);

    out = temp_cloud;
}

void createFeatureModels(const fs::path &base_dir, const std::string &extension)
{
    if (!fs::exists(base_dir) && !fs::is_directory(base_dir))
        return;

    fs::directory_iterator end_it();
    for (fs::directory_iterator it(base_dir); it != end_it; ++it)
    {
        auto file_name = (it->path().filename()).string();
        PCL_INFO("Process %s ...\n", file_name);
        if (fs::is_directory(it->status()))
        {
            std::stringstream ss;
            ss << it->path();
            //        print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
            createFeatureModels(it->path(), extension);
        }

        if (fs::is_regular_file(it->status()) && fs::extension(it->path()) == extension && !strstr(file_name.c_str(), "vfh") && !strstr(file_name.c_str(), "crh"))
        {
            std::vector<string> strs;
            boost::split(strs, file_name, boost::is_any_of("."));
            string view_id = strs[0];
            strs.clear();

            string descr_file = utils::getModelDescriptorFileName(base_dir, view_id);

            if (!fs::exists(descr_file))
            {
                PointCloudPtr view(new PointCloudType());

                string full_file_name = it->path().string();

                PCL_INFO("Compute VFH for %s\n", full_file_name);

                loadPCDFile(full_file_name.c_str(), *view);

                PCL_INFO("Point cloud has %d points\n", static_cast<int>(view->points.size()));

                // Preprocess view cloud
                processCloud(view, view);

                PCL_INFO("Point cloud has %d points after processing\n", static_cast<int>(view->points.size()));

                NormalCloudTypePtr normals(new NormalCloudType());
                FeatureCloudTypePtr descriptor(new FeatureCloudType);

                // Estimate the normals.
                std::shared_ptr<NormalEstimation> normal_estimator;
                normal_estimator->setInputCloud(view);

                normal_estimator->setRadiusSearch(normal_radius);
                SearchTreePtr kdtree(new SearchTree);
                normal_estimator->setSearchMethod(kdtree);

                // Alternative from local pipeline
                //              int norm_k = 10;
                //              normal_estimator->setKSearch(norm_k);
                normal_estimator->compute(*normals);

                // VFH estimation object.
                std::shared_ptr<FeatureEstimation> vfh;
                vfh->setInputCloud(view->makeShared());
                vfh->setInputNormals(normals);
                vfh->setSearchMethod(kdtree);
                // Optionally, we can normalize the bins of the resulting histogram,
                // using the total number of points.
                vfh->setNormalizeBins(true);
                // Also, we can normalize the SDC with the maximum size found between
                // the centroid and any of the cluster's points.
                vfh->setNormalizeDistance(false);

                vfh->compute(*descriptor);

                PCL_INFO("VFH descriptor has size: %d\n", static_cast<int>(descriptor->points.size()));

                savePCDFileBinary(descr_file.c_str(), *descriptor);
                PCL_INFO("%s was saved\n", descr_file);

                if (calculate_crh)
                {
                    PCL_INFO("Compute CRH features ...\n");
                    // Compute the CRH histogram
                    CRHCloudTypePtr histogram(new CRHCloudType);

                    // CRH estimation object
                    CRHEstimationPtr crh;
                    crh.setInputCloud(view);
                    crh.setInputNormals(normals);
                    
                    Eigen::Vector4f centroid4f;
                    pcl::compute3DCentroid(*view, centroid4f);
                    crh.setCentroid(centroid4f);

                    crh.compute(*histogram);

                    // Save centroid to file
                    auto centroid_file = utils::getCentroidFileName(base_dir, view_id);
                    Eigen::Vector3f centroid(centroid4f[0], centroid4f[1], centroid4f[2]);

                    // TODO: Move to the PersistenceUtils class
                    std::ofstream out(centroid_file.c_str());
                    if (!out)
                    {
                        std::cout << "Failed to open file " << centroid_file << " for saving centroid\n";
                    }

                    out << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
                    out.close();

                    PCL_INFO("%s was saved\n", centroid_file);

                    auto roll_file = utils::getCRHDescriptorFileName(base_dir, view_id);

                    savePCDFileBinary(roll_file.c_str(), *histogram);
                    PCL_INFO("%s was saved\n", roll_file);
                }
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
            PCL_ERROR("Error deleting old flann index file");
        else
            cout << "Old flann index file was successfully deleted\n";
    }

    if (fs::exists(training_data_h5_file_name))
    {
        if (remove(training_data_h5_file_name.c_str()) != 0)
            PCL_ERROR("Error deleting old training data file");
        else
            cout << "Old training data file was successfully deleted\n";
    }

    if (calculate_vfh)
        createFeatureModels(argv[1], extension);

    // Load the model histograms
    loadFeatureModels(argv[1], extension, models);
    print_highlight("Loaded %d VFH models. Creating training data %s/%s.\n",
                                  static_cast<int>(models.size()), training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());

    // Convert data into FLANN format
    FLANNMatrix data(new float[models.size() * models[0].second.size()], models.size(), models[0].second.size());

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
    PCL_ERROR("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str(), static_cast<int>(data.rows));
    FLANNIndex index(data, flann::LinearIndexParams());
    index.buildIndex();
    index.save(kdtree_idx_file_name);
    delete[] data.ptr();

    return (0);
}
