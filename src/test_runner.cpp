//#define DEBUG_GT_FILE_PATH_EXISTENCE
//#define DEBUG_ITERATING_THROUGH_THRESH_RANGE
//#define DEBUG_RUN_TIME_TEST

#include "vfh_cluster_classifier/common.h"
#include "vfh_cluster_classifier/persistence_utils.h"
#include "vfh_cluster_classifier/recognizer.h"
#include "vfh_cluster_classifier/test_runner.h"

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType>::Ptr PointCloudTypePtr;

TestRunner::TestRunner(const string &tests_base_path, string test_setup_name): tests_base_path_(tests_base_path), test_setup_name_(test_setup_name)
{
    tests_base_path_ = tests_base_path_ + "/" + test_setup_name_;

    std::cout << "tests_base_path: " << tests_base_path_ << "\n";

    start_thresh_ = 190;
    end_thresh_ = 200;
}

void TestRunner::initTests()
{
    std::cout << "[TestRunner::initTests]\n";

    if(!boost::filesystem::exists(tests_base_path_))
        boost::filesystem::create_directories(tests_base_path_);

    stringstream test_path_ss;
    test_path_ss << tests_base_path_ << "/setup.txt";

    string setup_file = test_path_ss.str();

    std::cout << "setup file: " << setup_file << "\n";

    // Write test case params setup to file
    output_.open(setup_file.c_str());
    output_ << "Experiment: " << test_setup_name_ << "\n\n";
    output_ << "start inlier thresh: " << start_thresh_ << "\n";
    output_ << "end inlier thresh: " << end_thresh_ << "\n";
    output_ << "nn_k: " << nn_k << "\n";
    output_.close();

    // Specify path to results file
//    stringstream results_path_ss;
//    results_path_ss << tests_base_path_ << "/results.txt";

//    test_output_file_ = results_path_ss.str();

//    pcl::console::print_debug("Results file: %s\n", test_output_file_.c_str());

//    iterateTestScenes();

    // Iterate over all the threshold values in the specified range
    for(int th = start_thresh_; th <= end_thresh_; th += 1)
    {
        thresh = th;

        pcl::console::print_info("inlier threshold: %d\n", thresh);

        stringstream results_path_ss;
        results_path_ss << tests_base_path_ << "/th_" << thresh << ".txt";

        test_output_file_ = results_path_ss.str();

        results_path_ss.str("");

        pcl::console::print_debug("Results file: %s\n", test_output_file_.c_str());

        iterateTestScenes();

#ifdef DEBUG_RUN_TIME_TEST
        break;
#endif
    }

    clearData();
}

void TestRunner::iterateTestScenes()
{
    pcl::console::print_info("Iterate through the test scenes ...\n");

    boost::filesystem::path test_scenes_path = test_scenes_dir;
    boost::filesystem::directory_iterator end_itr;

    for(boost::filesystem::directory_iterator iter (test_scenes_path); iter != end_itr; ++iter)
    {
        if(boost::filesystem::extension(iter->path()) == ".pcd")
        {
            test_scene = (iter->path()).string();

            pcl::console::print_debug("Load the test scene: %s\n", test_scene.c_str());

            std::size_t pos = test_scene.find_last_of("/") + 1;
            scene_name = test_scene.substr( pos );

            scene_name = scene_name.substr(0, scene_name.find(".pcd"));

            std::cout << "scene_name: " << scene_name << "\n";


            // Specify path to ground truth file
            string gt_file = scene_name + ".txt";

            stringstream gt_path_ss;
            gt_path_ss << gt_files_dir << "/" << gt_file;
            gt_file_path = gt_path_ss.str();

            pcl::console::print_debug("gt file path: %s\n", gt_file_path.c_str());

            if(!boost::filesystem::exists(gt_file_path))
            {
                pcl::console::print_error("Ground truth path %s doesn't exist\n", gt_file_path.c_str());
            }

            runDetector();
        }
    }

}

void TestRunner::runDetector()
{
    std::cout << "Run detector\n";

    PointCloudTypePtr scene_cloud (new PointCloudType ()), scene_cloud_filtered (new PointCloudType ());

    //
    // Load scene
    //
//    cout << "\n\n ---------------- Loading scene ---------------------- \n\n";
//    cout << "\n\nScene: " << scene_pcd_file << "\n";
    pcl::io::loadPCDFile(test_scene, *scene_cloud);

    pcl::console::print_debug("Scene cloud has %d points\n", (int)scene_cloud->points.size());

#ifndef DEBUG_ITERATING_THROUGH_THRESH_RANGE
    recognize(scene_cloud, scene_cloud_filtered);
#endif

    // Calculate recognition performance

    int positives_n = 0;
    int negatives_n = 0;
    int tp_n = 0;
    int fp_n = 0;
    int tn_n = 0;
    int fn_n = 0;

//    if(recognized_objects.size())
//    {
        for(int i = 0; i < training_objects_ids.size(); i++)
        {
            string train_object_id = training_objects_ids[i];
            bool is_present = PersistenceUtils::modelPresents(gt_file_path, train_object_id);
            bool is_found = false;

            for(int j = 0; j < recognized_objects.size(); j++)
            {
                std::string object_id = recognized_objects[j];
                if(object_id == train_object_id) is_found = true;
            }

            cout << "Model " << train_object_id << "\n";
            cout << "\t- is present: " << is_present << "\n";
            cout << "\t- is found: " << is_found << "\n\n";

            if(is_present)
            {
                if(is_found) { tp_n++; positives_n++; }
                else { fn_n++; negatives_n++; }
            }
            else
            {
                if(is_found) { fp_n++; positives_n++; }
                else { tn_n++; negatives_n++; }
            }

        }

        recognized_objects.clear();
//    }

//    cout << "Positives: " << positives_n << "\n";
//    cout << "Negatives: " << negatives_n << "\n";
//    cout << "tp: " << tp_n << ", fp: " << fp_n << "\n";
//    cout << "fn: " << fn_n << ", tn: " << tn_n << "\n";

    // Write the results to file
    output_.open(test_output_file_.c_str(), std::ios::app);
    output_ << scene_name << "\n";

    output_ << "tp: " << tp_n << ", fp: " << fp_n <<
               ", fn: " << fn_n << ", tn: " << tn_n << "\n\n";

    output_.close();

    std::cout << "\n\n";
}
