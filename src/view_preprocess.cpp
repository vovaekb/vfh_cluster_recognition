#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

string samples_path;
string output_path;

bool perform_scaling(false);

// Static parameters
int clouds_number(42);
float voxel_leaf_size_(0.001);

// Containers for objects
vector<PointCloudTypePtr> clouds;

vector<int> mapping;

void process_cloud(PointCloudTypePtr &cloud, int index)
{
    if (perform_scaling)
    {
        // Scale cloud, ...
        cout << "Scale cloud\n";
        Eigen::Matrix4f cloud_transform = Eigen::Matrix4f::Identity();

        cloud_transform(0, 0) = 0.001;
        cloud_transform(1, 1) = 0.001;
        cloud_transform(2, 2) = 0.001;

        pcl::transformPointCloud(*cloud, *cloud, cloud_transform);
    }

    // ... downsample, ...
    cout << "Downsample cloud\n";
    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    PointCloudTypePtr temp_cloud(new PointCloudType());
    voxel_grid.filter(*temp_cloud);
    cloud = temp_cloud;

    // ... and remove NaNs
    cout << "Remove NaN\n";
    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);

    // Save point cloud
    cout << "Save cloud\n";
    string output_pcd = output_path + "/" + boost::to_string(index) + ".pcd";
    pcl::io::savePCDFileASCII(output_pcd, *cloud);

    cout << "Point cloud was saved as " + output_pcd + "\n";
}

void process()
{
    if (!boost::filesystem::exists(output_path))
        boost::filesystem::create_directory(output_path);

    // Load point clouds
    for (int i = 1; i <= clouds_number; i++)
    {
        string pcd_path = samples_path + "/" + boost::to_string(i) + ".pcd";

        if (boost::filesystem::exists(pcd_path))
        {
            PointCloudTypePtr cloud(new PointCloudType());
            if (pcl::io::loadPCDFile(pcd_path, *cloud) != 0)
            {
                return;
            }

            cout << "Point cloud " << pcd_path << " has " << cloud->points.size() << " points\n";

            process_cloud(cloud, i);
        }
    }
}

void showHelp(char *filename)
{
    cout << "\t\t ** process_cloud package **\n";
    cout << "Usage: " << filename << " --samples_path <samples_path> [options]\n";
    cout << "* where --samples_path       path to the directory of clouds to process\n";
    cout << "* where options are:\n";
    cout << "--output_path <output_dir>   path to saved output clouds\n";
    cout << "--clouds_n <clouds_n>        number of files to process\n";
    cout << "-scale                       perform scaling\n";
}

void parseCommandLine(int argc, char **argv)
{
    if (pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }

    pcl::console::parse_argument(argc, argv, "--samples_path", samples_path);

    if (samples_path == "")
    {
        cout << "Samples directory missing!\n";
        showHelp(argv[0]);
        exit(-1);
    }

    pcl::console::parse_argument(argc, argv, "--output_path", output_path);

    if (output_path == "")
    {
        output_path = output_path + "_scaled";
    }

    if (pcl::console::find_switch(argc, argv, "-scale"))
    {
        perform_scaling = true;
    }

    pcl::console::parse_argument(argc, argv, "--clouds_n", clouds_number);
}

int main(int argc, char **argv)
{
    parseCommandLine(argc, argv);

    //    std::stringstream output_ss;
    //    output_ss << output_path << "/" << samples_path << "_output";

    cout << "Samples dir: " << samples_path << "\n";
    cout << "Output path: " << output_path << "\n";
    cout << "Clouds number: " << clouds_number << "\n";
    cout << "Perform scaling: " << perform_scaling << "\n";

    process();
    return 0;
}
