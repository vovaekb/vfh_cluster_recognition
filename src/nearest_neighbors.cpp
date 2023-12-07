// #define VISUALIZE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/filesystem.hpp>

#include "vfh_cluster_classifier/persistence_utils.h"
#include "vfh_cluster_classifier/nearest_search.h"

using namespace pcl::console;
using namespace pcl::io;
namespace fs = boost::filesystem;

int main(int argc, char **argv)
{
    int k = 6;

    double thresh = DBL_MAX; // No threshold, disabled by default
    bool visualize(false);

    if (argc < 2)
    {
        print_error("Need at least three parameters! Syntax is: %s <query_vfh_model.pcd> [options] {kdtree.idx} {training_data.h5} {training_data.list}\n", argv[0]);
        print_info("    where [options] are:  -k      = number of nearest neighbors to search for in the tree (default: ");
        print_value("%d", k);
        print_info(")\n");
        print_info("                          -thresh = maximum distance threshold for a model to be considered VALID (default: ");
        print_info("                          -vis = visualize\n");
        print_value("%f", thresh);
        print_info(")\n\n");
        return (-1);
    }

    std::string extension(".pcd");
    transform(extension.begin(), extension.end(), extension.begin(), (int (*)(int))tolower);

    // Load the test histogram
    std::vector<int> pcd_indices = parse_file_extension_argument(argc, argv, ".pcd");
    vfh_model histogram;
    if (!loadHist(argv[pcd_indices.at(0)], histogram))
    {
        print_error("Cannot load test file %s\n", argv[pcd_indices.at(0)]);
        return (-1);
    }

    parse_argument(argc, argv, "-thresh", thresh);
    // Search for the k closest matches
    parse_argument(argc, argv, "-k", k);
    parse_argument(argc, argv, "-vis", visualize);
    print_highlight("Using ");
    print_value("%d", k);
    print_info(" nearest neighbors.\n");

    std::string kdtree_idx_file_name = "kdtree.idx";
    std::string training_data_h5_file_name = "training_data.h5";
    std::string training_data_list_file_name = "training_data.list";

    std::vector<vfh_model> models;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    flann::Matrix<float> data;
    // Check if the data has already been saved to disk
    if (!fs::exists("training_data.h5") || !fs::exists("training_data.list"))
    {
        print_error("Could not find training data models files %s and %s!\n",
                                  training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());
        return (-1);
    }
    else
    {
        PersistenceUtils::loadFileList(models, training_data_list_file_name);
        flann::load_from_file(data, training_data_h5_file_name, "training_data");
        print_highlight("Training data found. Loaded %d VFH models from %s/%s.\n",
                                      static_cast<int>(data.rows), training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());
    }

    // Check if the tree index has already been saved to disk
    if (!fs::exists(kdtree_idx_file_name))
    {
        print_error("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str());
        return (-1);
    }
    else
    {
        flann_distance_metric index(data, flann::SavedIndexParams("kdtree.idx"));
        index.buildIndex();
        nearestKSearch(index, histogram, k, k_indices, k_distances);
    }

    // Output the results on screen
    print_highlight("The closest %d neighbors for %s are:\n", k, argv[pcd_indices[0]]);
    for (int i = 0; i < k; ++i)
        print_info("    %d - %s (%d) with a distance of: %f\n",
                                 i, models.at(k_indices[0][i]).first.c_str(), k_indices[0][i], k_distances[0][i]);

    auto best_dist = std::numeric_limits<float>::infinity();
    auto best_index = -1;
    for (auto i = 0; i < k; ++i)
    {
        if (k_distances[0][i] < best_dist)
        {
            best_dist = k_distances[0][i];
            best_index = i;
        }
    }

    vfh_model best_model = models.at(k_indices[0][best_index]);

    cout << "The best model index: " << best_index << "\n";

    cout << "The best model " << best_model.first.c_str() << " has distance: " << best_dist << "\n";

    if (visualize)
    {
        // Load the results
        pcl::visualization::PCLVisualizer p(argc, argv, "VFH Cluster Classifier");
        int y_s = static_cast<int>(
                std::floor(std::sqrt(static_cast<double>(k))));
        int x_s = y_s + static_cast<int>(
                std::ceil((k / static_cast<double>(y_s)) - y_s));
        double x_step = 1 / static_cast<double>(x_s);
        double x_step = 1 / static_cast<double>(y_s);
        print_highlight("Preparing to load ");
        print_value("%d", k);
        print_info(" files (");
        print_value("%d", x_s);
        print_info("x");
        print_value("%d", y_s);
        print_info(" / ");
        print_value("%f", x_step);
        print_info("x");
        print_value("%f", y_step);
        print_info(")\n");

        int viewport = 0, l = 0, m = 0;
        for (int i = 0; i < k; ++i)
        {
            std::string cloud_name = models.at(k_indices[0][i]).first;
            boost::replace_last(cloud_name, "_vfh", "");

            p.createViewPort(l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
            l++;
            if (l >= x_s)
            {
                l = 0;
                m++;
            }

            pcl::PCLPointCloud2 cloud;
            print_highlight(stderr, "Loading ");
            print_value(stderr, "%s ", cloud_name.c_str());
            if (loadPCDFile(cloud_name, cloud) == -1)
                break;

            // Convert from blob to PointCloud
            DepthPointCloudType cloud_xyz;
            pcl::fromPCLPointCloud2(cloud, cloud_xyz);

            if (cloud_xyz.points.size() == 0)
                break;

            print_info("[done, ");
            print_value("%d", static_cast<int>(cloud_xyz.points.size());
            print_info(" points]\n");
            print_info("Available dimensions: ");
            print_value("%s\n", pcl::getFieldsList(cloud).c_str());

            // Demean the cloud
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(cloud_xyz, centroid);
            DepthPointCloudTypePtr cloud_xyz_demean(new DepthPointCloudType);
            pcl::demeanPointCloud<DepthPointType>(cloud_xyz, centroid, *cloud_xyz_demean);
            // Add to renderer*
            //    p.addPointCloud (cloud_xyz_demean, cloud_name, viewport);

            // Check if the model found is within our inlier tolerance
            std::stringstream ss;
            ss << k_distances[0][i];
            if (k_distances[0][i] > thresh)
            {
                p.addText(ss.str(), 20, 30, 1, 0, 0, ss.str(), viewport); // display the text with red

                // Create a red line
                DepthPointType min_p, max_p;
                pcl::getMinMax3D(*cloud_xyz_demean, min_p, max_p);
                std::stringstream line_name;
                line_name << "line_" << i;
                p.addLine(min_p, max_p, 1, 0, 0, line_name.str(), viewport);
                p.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, line_name.str(), viewport);
            }
            else
                p.addText(ss.str(), 20, 30, 0, 1, 0, ss.str(), viewport);

            // Increase the font size for the score*
            p.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, ss.str(), viewport);

            // Add the cluster name
            p.addText(cloud_name, 20, 10, cloud_name, viewport);
            p.addCoordinateSystem(0.1);
        }
        // Add coordianate systems to all viewports
        //  p.addCoordinateSystem (0.1, "global", 0);

        p.spin();
    }

    return (0);
}
