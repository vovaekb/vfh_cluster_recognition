// #define LOADING_MODELS_DEBUG
// #define INCLUDE_RECOGNIZER_DEBUG
#define REFACTOR_TESTING_DEBUG
// #define DEBUG_SPECIFING_GT_FILE

/*
 * Script for performing experimental estimation of VFH recognizer
 * Input: testing data set (PCD scenes), training data set (PCD + VFH models, Kd-tree)
 * Output: recognition results written in log files
 * ************************ Basic test **********************************
 * Test cases: distance, view angle, occluded
 */

#include <iostream>
#include <vector>
#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
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

#include "vfh_cluster_classifier/common.h"
#include "vfh_cluster_classifier/recognizer.h"
#include "vfh_cluster_classifier/persistence_utils.h"
#include "vfh_cluster_classifier/test_runner.h"

using namespace std;

// types definition
typedef pcl::PointXYZRGB PointT;
typedef pcl::Normal NormalT;
typedef pcl::VFHSignature308 FeatureT;
typedef std::pair<std::string, std::vector<float>> vfh_model;
typedef pcl::Histogram<90> CRH90;

/***** Shared parameters ******/
string base_descr_dir = "clusters_vfh";
string training_data_path = "training_models";
string gt_files_dir = "gt_files";
string gt_file_path;
string scene_name;
string test_scene;

// Main algorithm parameters
bool run_tests(false);
bool perform_crh(false);
bool apply_thresh(false);
int nn_k = 3;     // 6
int thresh = 195; // 180; // 60
float distance_thresh;

std::vector<pcl::PointCloud<PointT>::Ptr> cluster_clouds;
std::vector<std::string> recognized_objects;
string found_model("");

// Main file paths
string test_scenes_dir;
string experiments_dir;

ofstream output_stream;

// Data structures for VFH recognition
vector<vfh_model> models;
std::vector<std::string> training_objects_ids;
flann::Matrix<float> data;
vfh_model histogram;

/************************** Command line procedures **********************************/
void showHelp(char *filename)
{
    cout << "Usage: " << filename << " [options]\n"
         << "*options: \n"
         << "--test_scenes_dir <dir>        - directory with test scenes for experiments\n"
         << "--test_scene <scene_path>    - path to scene to recognize\n"
         << "--train_dir <train_dir>      - path to directory with training data\n"
         << "--gt_files <gt_files>         - path to ground truth files (for evaluation case)\n"
         << "--exper_dir <dir>            - experiments directory\n"
         << "--dist_thresh <dist_thresh>  - distance threshold\n"
         << "--th <thresh>                - match threshold\n"
         << "--k <k>                      - number of NNs in knn-search\n"
         << "-thresh                      - apply threshold\n"
         << "-test                        - run tests\n"
         << "-crh                         - perform CRH\n"
         << "-h                           - show help\n\n";
}

void parseCommandLine(int argc, char **argv)
{
    if (pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }

    pcl::console::parse_argument(argc, argv, "--test_scenes_dir", test_scenes_dir);

    pcl::console::parse_argument(argc, argv, "--test_scene", test_scene);

    if (test_scenes_dir == "" && test_scene == "")
    {
        pcl::console::print_error("Test data directory or test scene should be specified!\n");
        showHelp(argv[0]);
        exit(-1);
    }

    pcl::console::parse_argument(argc, argv, "--train_dir", training_data_path);

    pcl::console::parse_argument(argc, argv, "--gt_files", gt_files_dir);

    pcl::console::parse_argument(argc, argv, "--exper_dir", experiments_dir);

    pcl::console::parse_argument(argc, argv, "--dist_thresh", distance_thresh);

    pcl::console::parse_argument(argc, argv, "--th", thresh);

    pcl::console::parse_argument(argc, argv, "--k", nn_k);

    if (pcl::console::find_switch(argc, argv, "-thresh"))
    {
        apply_thresh = true;
    }

    if (pcl::console::find_switch(argc, argv, "-test"))
    {
        run_tests = true;
    }

    if (pcl::console::find_switch(argc, argv, "-crh"))
    {
        perform_crh = true;
    }

    if (run_tests)
    {
        cout << "test_scenes_dir: " << test_scenes_dir << "\n";
        cout << "experiments_dir: " << experiments_dir << "\n";
    }

    cout << "test_scene: " << test_scene << "\n";
    cout << "training_data_path: " << training_data_path << "\n";
    cout << "gt_files_dir: " << gt_files_dir << "\n";
    cout << "distance_thresh: " << distance_thresh << "\n";
    cout << "knn thresh: " << thresh << "\n";
    cout << "nn_k: " << nn_k << "\n";
    cout << "apply thresh: " << apply_thresh << "\n";
    cout << "run tests: " << run_tests << "\n";
    cout << "perform crh: " << perform_crh << "\n";
}

void getTrainingObjectsIds(const boost::filesystem::path &base_dir)
{
    std::cout << "Loading training objects names\n";

    if (!boost::filesystem::exists(base_dir) && !boost::filesystem::is_directory(base_dir))
        return;

    for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator(); ++it)
    {
        if (boost::filesystem::is_directory(it->status()))
        {
            string object_name = (it->path().filename()).string();

            std::cout << "Storing object " << object_name << "\n";
            training_objects_ids.push_back(object_name);
        }
    }
}

/** \brief Runs the testing procedure
 */
void runTests()
{
    cout << "*********************\n";
    cout << "*** Run test mode ***\n";
    cout << "*********************\n";

    string test_name = "descr_runtime_test"; // "standard_test";
    TestRunner test_runner(experiments_dir, test_name);
    test_runner.initTests();

#ifndef REFACTOR_TESTING_DEBUG

    string exper_setup_name = "basic";

    cout << "[runTests]\n";

    if (!boost::filesystem::exists(test_scenes_dir) && !boost::filesystem::is_directory(test_scenes_dir))
        return;

    // TODO: Iterate through all experiment cases
    // 1. Read every case_n.txt file
    // 2. Use case_n as the name of the file for experimental results
    // 3. Use each line test_scene_k.pcd in the case_n.txt file to open the test_dir/case_n/test_scene_k.pcd
    // 4. Run recognizer on test_scene_k.pcd

    for (boost::filesystem::directory_iterator it(test_scenes_dir); it != boost::filesystem::directory_iterator(); ++it)
    {
        if (boost::filesystem::is_regular_file(it->status()) && boost::filesystem::extension(it->path()) == ".txt")
        {
            string case_file = (it->path().filename()).string();
            string case_file_path = (it->path()).string();
            string case_name = case_file.substr(0, case_file.find_last_of("."));

            cout << "[runTests] Case name: " << case_name << "\n";

            stringstream exper_file_ss;
            exper_file_ss << experiments_dir << "/" << case_name << ".txt";
            string exper_file_path = exper_file_ss.str();

            cout << "Experiment file: " << exper_file_path << "\n";

            ifstream input_stream(case_file_path.c_str());

            string scene_name;
            while (input_stream.is_open())
            {
                while (getline(input_stream, scene_name))
                {
                    if (scene_name.empty() || scene_name.at(0) == '#') // || scene_name.substr(0, 10) != "whiteboard")
                        continue;

                    //                    cout << "[runTests] Reading line: " << scene_name << "\n";

                    stringstream scene_ss;
                    scene_ss << test_scenes_dir << "/" << case_name << "/" << scene_name;
                    string scene_pcd_path = scene_ss.str();

                    cout << "[runTests] Scene path: " << scene_pcd_path << "\n";

                    pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>()), scene_cloud_filtered(new pcl::PointCloud<PointT>());

                    pcl::io::loadPCDFile(scene_pcd_path.c_str(), *scene_cloud);

                    //                    pcl::console::print_info("Scene cloud has size: %d\n", (int)scene_cloud->points.size());

                    recognize(scene_cloud, scene_cloud_filtered);

                    //                    output_stream.open(exper_file_path.c_str(), ios::app);
                    //                    output_stream << scene_name << " " << found_model << "\n";
                    //                    output_stream.close();

                    //                    found_model = "";

                    cout << "\n\n";
                }
                input_stream.close();
            }
        }
    }
#endif
}

/** \brief Runs simple scene recognition procedure
 */
void recognizeScene()
{
    pcl::PointCloud<PointT>::Ptr scene_cloud(new pcl::PointCloud<PointT>()), scene_cloud_filtered(new pcl::PointCloud<PointT>());

    pcl::io::loadPCDFile(test_scene.c_str(), *scene_cloud);

    pcl::console::print_info("Scene cloud has size: %d\n", (int)scene_cloud->points.size());

    recognize(scene_cloud, scene_cloud_filtered);

    if (recognized_objects.size())
    {
        std::cout << "Estimate accuracy of recognition\n";

        if (!boost::filesystem::exists(gt_files_dir))
        {
            pcl::console::print_error("Ground truth path %s doesn't exist\n", gt_files_dir.c_str());
            exit(-1);
        }

        int positives_n = 0;
        int negatives_n = 0;
        int tp_n = 0;
        int fp_n = 0;
        int tn_n = 0;
        int fn_n = 0;

        for (auto &train_object_id : training_objects_ids)
        {
            bool is_present = PersistenceUtils::modelPresents(gt_file_path, train_object_id);
            bool is_found = false;

            for (auto &object_id : recognized_objects)
            {
                if (object_id == train_object_id)
                    is_found = true;
            }

            cout << "Model " << train_object_id << "\n";
            cout << "\t- is present: " << is_present << "\n";
            cout << "\t- is found: " << is_found << "\n\n";

            if (is_present)
            {
                if (is_found)
                {
                    tp_n++;
                    positives_n++;
                }
                else
                {
                    fn_n++;
                    negatives_n++;
                }
            }
            else
            {
                if (is_found)
                {
                    fp_n++;
                    positives_n++;
                }
                else
                {
                    tn_n++;
                    negatives_n++;
                }
            }
        }

        cout << "Positives: " << positives_n << "\n";
        cout << "Negatives: " << negatives_n << "\n";
        cout << "tp: " << tp_n << ", fp: " << fp_n << "\n";
        cout << "fn: " << fn_n << ", tn: " << tn_n << "\n";
    }
    else
    {
        pcl::console::print_highlight("No objects found\n");
    }
}

int main(int argc, char **argv)
{
    cout << "recognize_training script\n";
    parseCommandLine(argc, argv);

    if (test_scene != "")
    {
        std::size_t pos = test_scene.find_last_of("/") + 1;
        scene_name = test_scene.substr(pos);

        scene_name = scene_name.substr(0, scene_name.find(".pcd"));

        std::cout << "scene_name: " << scene_name << "\n";

        string gt_file = scene_name + ".txt";

        stringstream gt_path_ss;
        gt_path_ss << gt_files_dir << "/" << gt_file;

        gt_file_path = gt_path_ss.str();

        cout << "gt file path: " << gt_file_path << "\n";
    }

#ifndef DEBUG_SPECIFING_GT_FILE
    //    string kdtree_idx_file_name = training_data_path + "/kdtree.idx";
    string training_data_h5_file_name = training_data_path + "/training_data.h5";
    string training_data_list_file_name = training_data_path + "/training_data.list";

    getTrainingObjectsIds(training_data_path);

    // Check if the data has already been saved to disk
    if (!boost::filesystem::exists(training_data_h5_file_name) || !boost::filesystem::exists(training_data_list_file_name))
    {
        pcl::console::print_error("Could not find training data models files %s and %s!\n",
                                  training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());
        return -1;
    }
    else
    {
        loadFileList(models, training_data_list_file_name);
        flann::load_from_file(data, training_data_h5_file_name, "training_data");
        pcl::console::print_highlight("Training data found. Loaded %d VFH models from %s and %s.\n",
                                      (int)data.rows, training_data_h5_file_name.c_str(), training_data_list_file_name.c_str());
    }

    loadIndex();

#ifndef LOADING_MODELS_DEBUG
    if (run_tests)
        runTests();
    else
        recognizeScene();
#endif

#endif
}
