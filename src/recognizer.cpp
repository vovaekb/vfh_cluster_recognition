//#define THRESH_DISABLE
//#define ENABLE_DISTANCE_THRESH
#define DISABLE_SINGLE_OBJECT_RECOGNITION
//#define ENABLE_CRH_ALIGNMENT
#define DISABLE_CRH_FEATURES
//#define DEBUG_RUN_TIME_TEST

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

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
#include <boost/algorithm/string.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "vfh_cluster_classifier/common.h"
#include "vfh_cluster_classifier/recognizer.h"
#include "vfh_cluster_classifier/persistence_utils.h"

float voxel_leaf_size (0.001); // (0.005);
float normal_radius (0.03);

std::vector<index_score> models_scores;
vector<ObjectHypothesis, Eigen::aligned_allocator<ObjectHypothesis> > object_hypotheses_;

flann::Index<flann::ChiSquareDistance<float> > * flann_index;
sortIndexScores sortIndexScoresOp;

bool
loadFileList(vector<vfh_model> &models, const string &filename)
{
//    pcl::console::print_info("[loadFileList]\n");

    ifstream fs;
    fs.open(filename.c_str());
    if(!fs.is_open() || fs.fail())
        return (false);

    string line;
    while(!fs.eof())
    {
        getline(fs, line);
        if(line.empty())
            continue;

        vfh_model m;
        m.first = line;
        models.push_back(m);
    }
    fs.close();
    return (true);
}

void loadIndex()
{
    std::cout << "Loading search index ...\n";
    string kdtree_idx_file_name = training_data_path + "/kdtree.idx";

    std::cout << "FLANN index file: " << kdtree_idx_file_name << "\n";

    // Check if the tree index has already been saved to disk
    if(!boost::filesystem::exists(kdtree_idx_file_name))
    {
        pcl::console::print_error("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str());
        return;
    }
    else
    {
        flann_index = new flann::Index<flann::ChiSquareDistance<float> > (data, flann::SavedIndexParams(kdtree_idx_file_name.c_str()));
        flann_index->buildIndex();
    }
}

void createHist(pcl::PointCloud<PointT>::Ptr &cloud, pcl::PointCloud<FeatureT>::Ptr &descriptor, pcl::PointCloud<CRH90>::Ptr &crh_histogram, Eigen::Vector4f &centroid)
{
    cout << "Create VFH histogram...\n";

//    std::cout << "XYZRGB cloud has size: " << cloud->points.size() << "\n";

//    typedef pcl::PointXYZ PointDT;
//    pcl::PointCloud<PointDT>::Ptr depth_cloud (new pcl::PointCloud<PointDT>);

//    for(size_t i = 0; i < cloud->points.size(); i++)
//    {
//        PointDT p;
//        p.x = cloud->points[i].x;
//        p.y = cloud->points[i].y;
//        p.z = cloud->points[i].z;
//        depth_cloud->push_back(p);
//    }

//    std::cout << "XYZ cloud has size: " << depth_cloud->points.size() << "\n";

    // Data structures
    pcl::PointCloud<NormalT>::Ptr normals (new pcl::PointCloud<NormalT>);

    // Algorithm parameters

    // estimate normals
    pcl::NormalEstimation<PointT, NormalT> normalEstimation;
    normalEstimation.setInputCloud(cloud); // depth_cloud

    normalEstimation.setRadiusSearch(normal_radius);
    pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
    normalEstimation.setSearchMethod(kdtree);

    // Alternative from local pipeline
    //              int norm_k = 10;
    //              normalEstimation.setKSearch(norm_k);
    normalEstimation.compute(*normals);

//    cout << "[createHist] Normal cloud has size: " << normals->points.size() << "\n";
//    cout << "[createHist] Point cloud has size: " << cloud->points.size() << "\n";

#ifdef DEBUG_RUN_TIME_TEST
    std::ofstream out;
    out.open("vfh_time.txt", std::ios::app);

    clock_t t_start = clock();
#endif
    // calculate vfh
    pcl::VFHEstimation<PointT, NormalT, FeatureT> vfh;
    vfh.setInputCloud(cloud); // depth_cloud
    vfh.setInputNormals(normals);
    vfh.setSearchMethod(kdtree);
    // Optionally, we can normalize the bins of the resulting histogram,
    // using the total number of points.
    vfh.setNormalizeBins(true);
    // Also, we can normalize the SDC with the maximum size found between
    // the centroid and any of the cluster's points.
    vfh.setNormalizeDistance(false);

    vfh.compute(*descriptor);

#ifdef DEBUG_RUN_TIME_TEST
    double calc_time = (double)(clock() - t_start)/CLOCKS_PER_SEC;
    PCL_INFO("Computing time: %.3fs\n", calc_time);

    out << calc_time << "\n";
    out.close();
#endif

    cout << "VFH descriptor has size: " << descriptor->points.size() << "\n\n";

//    stringstream vfh_file_ss;
//    vfh_file_ss << base_descr_dir << "/cluster_" << index << "_vfh.pcd";

//    string descr_file = vfh_file_ss.str();
//    pcl::io::savePCDFileBinary(descr_file.c_str(), *descriptor);
//    cout << "[createHist] " << descr_file << " was saved\n";

#ifndef DISABLE_CRH_FEATURES
    if(perform_crh)
    {
        std::cout << "Computing CRH histogram...\n";

        pcl::PointCloud<CRH90>::Ptr crh_histogram (new pcl::PointCloud<CRH90>);

        // CRH estimation object
        pcl::CRHEstimation<PointDT, NormalT, CRH90> crh;
        crh.setInputCloud(depth_cloud); // cloud);
        crh.setInputNormals(normals);

        pcl::compute3DCentroid(*depth_cloud, centroid); // cloud
        crh.setCentroid(centroid);

        crh.compute(*crh_histogram);
        crh.
        std::cout << "CRH computing complete\n";

//        stringstream path_ss;
//        path_ss << base_descr_dir << "/cluster_" << ind << "_crh.pcd"; // ind is a parameter of method

//        string roll_file = path_ss.str();

//        pcl::io::savePCDFileBinary(roll_file.c_str(), *crh_histogram);
//        cout << roll_file << " was saved\n";
    }
#endif
}


inline void
nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
                int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
    std::cout << "Nearest K search...\n";
  // Query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
  memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);

  index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
  delete[] p.ptr ();
}


void preprocessCloud(pcl::PointCloud<PointT>::Ptr &input, pcl::PointCloud<PointT>::Ptr &output)
{
    cout << "Preprocess cloud ...\n";
//    cout << "Input cloud has size: " << input->points.size() << "\n";

    vector<int> mapping;
    pcl::removeNaNFromPointCloud(*input, *input, mapping);

    // Downsampling
    pcl::VoxelGrid<PointT> vox_grid;
    vox_grid.setInputCloud(input);
    vox_grid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    pcl::PointCloud<PointT>::Ptr temp_cloud (new pcl::PointCloud<PointT> ());
    vox_grid.filter(*temp_cloud);

    output = temp_cloud;

#ifdef ENABLE_DISTANCE_THRESH
    // Distance thresholding
    const float depth_limit = 0.6;
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(output);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, distance_thresh); // depth_limit);
    pass.filter(*output);
#endif

//    cout << "Output cloud has size: " << output->points.size() << "\n";
}

void segmentScene(pcl::PointCloud<PointT>::Ptr &input)
{
//    cout << "[segmentScene] Input cloud has size: " << input->points.size() << "\n";

    pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_p (new pcl::PointCloud<PointT> ()), cloud_f (new pcl::PointCloud<PointT> ());

    // Segmentation plane
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
    // Create segmentation object
    pcl::SACSegmentation<PointT> seg;
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01); // 0.03 - default

    // Segmentation
    seg.setInputCloud(input);
    seg.segment(*inliers, *coefficients);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_p);

    extract.setNegative(true);
    extract.filter(*cloud_f);
    input.swap(cloud_f);

    vector<pcl::PointIndices> clusters;
    pcl::PointCloud<PointT>::Ptr colored_cloud;

    // RegionGrowingRGB segmentation
    pcl::RegionGrowingRGB<PointT> reg;
    reg.setInputCloud(input);
    reg.setSearchMethod(tree);
    reg.setDistanceThreshold(10);
    reg.setPointColorThreshold(6);
    reg.setRegionColorThreshold(5);
    reg.setMinClusterSize(500); // 300 - previous // 400 // 600 - default

    reg.extract(clusters);

    colored_cloud = reg.getColoredCloud();

//    string colored_cloud_pcd = "clusters_cloud.pcd";
//    pcl::io::savePCDFileASCII(colored_cloud_pcd.c_str(), *colored_cloud);

//    cout << "[segmentScene] " << clusters.size() << " clusters was found\n\n";

    cluster_clouds.clear();
    if(clusters.size() > 0)
    {
        // Creating cluster clouds
        int ind = 0;
        for(std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
        {
            pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            for(std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
                cloud->points.push_back(input->points[*point]);
            cloud->width = cloud->points.size();
            cloud->height = 1;
            cloud->is_dense = true;

//            cout << "[segmentScene] Cluster " << ind << " has size: " << cloud->points.size() << "\n";

            cluster_clouds.push_back(cloud);
            ind++;
        }
    }

    cout << "\n";
}

void classifyCluster(const int &ind, pcl::PointCloud<PointT>::Ptr &cloud)
{
    pcl::console::print_info("[classifyCluster] Cluster cloud %i has size: %d\n", ind, (int)cloud->points.size());

    pcl::PointCloud<FeatureT>::Ptr descriptor (new pcl::PointCloud<FeatureT>);
    pcl::PointCloud<CRH90>::Ptr cluster_crh (new pcl::PointCloud<CRH90>);
    Eigen::Vector4f cluster_centroid;

    createHist(cloud, descriptor, cluster_crh, cluster_centroid);

//    cout << "Cluster CRH has size: " << cluster_crh->points.size() << "\n";

//    if(cluster_crh->points.size())
//        pcl::console::print_highlight("Cluster centroid: %f %f %f\n", cluster_centroid[0], cluster_centroid[1], cluster_centroid[2]);

    pcl::console::print_highlight("Preparing data for K search ...\n");

    float* hist = descriptor->points[0].histogram;
    int size_feat = sizeof(descriptor->points[0].histogram) / sizeof(float);
    std::vector<float> std_hist (hist, hist + size_feat);

    vfh_model histogram;
    histogram.second = std_hist;

//    std_hist.clear();

    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;

    nearestKSearch(*flann_index, histogram, nn_k, k_indices, k_distances);

//    histogram.second.clear();

    // Output the results on the screen
        pcl::console::print_highlight("The closest %d neighbors for cluster %d are:\n", nn_k, ind);

        for(int i = 0; i < nn_k; i++)
        {
#ifndef THRESH_DISABLE
            if(apply_thresh && k_distances[0][i] > thresh) continue;
#endif
            string vfh_model_path = models.at(k_indices[0][i]).first;

            std::vector<std::string> strs;
            boost::split(strs, vfh_model_path, boost::is_any_of("/"));
            std::string model_name = strs[strs.size() - 2];

            pcl::console::print_info("    %d - %s (%d) with distance of: %f\n",
                                     i, model_name.c_str(), k_indices[0][i], k_distances[0][i]);

            float score = k_distances[0][i];
            index_score model_score;
            model_score.model_id = model_name;
            model_score.score = score;
            models_scores.push_back(model_score);

#ifdef ENABLE_CRH_ALIGNMENT
            if(perform_crh)
            {
                std::cout << "-------- Perform pose estimation --------------\n";
                stringstream model_path_ss;
                model_path_ss << model_path << ".pcd";
                string model_cloud_file = model_path_ss.str();

                std::cout << "model_cloud_file: " << model_cloud_file << "\n";

                // Clouds for storing object's cluster and model
                pcl::PointCloud<PointT>::Ptr model_cloud (new pcl::PointCloud<PointT>);

                pcl::io::loadPCDFile<PointT>(model_cloud_file.c_str(), *model_cloud);

                cout << "Model cloud has size: " << model_cloud->points.size() << "\n";

                // Object for storing CRHs of both
                pcl::PointCloud<CRH90>::Ptr model_crh (new pcl::PointCloud<CRH90>);

                // Objects for storing the centroids
                Eigen::Vector3f model_centroid;

                // Performing pose estimation
                // For the cluster and each  of the best k models:
                // 1. Load CRH histograms - y
                // 2. Compute the centroids
                // 3. Align the centroids and compute the roll angles

                // Loading CRHs
                model_path_ss.str("");
                model_path_ss << model_path << "_crh.pcd";
                string model_crh_file = model_path_ss.str();

                cout << "Model CRH histogram file: " << model_crh_file << "\n";
                pcl::io::loadPCDFile(model_crh_file, *model_crh);

                cout << model_crh_file << " histogram has size: " << model_crh->points.size() << "\n";

                // read centroid from file
                model_path_ss.str("");
                model_path_ss << model_path << "_centroid.txt";
                string centroid_file = model_path_ss.str();

                PersistenceUtils::getCentroidFromFile(centroid_file, model_centroid);

                std::cout << "Model centroid is loaded from file " << centroid_file << "\n";
                pcl::console::print_highlight("Model centroid: %f %f %f\n", model_centroid[0], model_centroid[1], model_centroid[2]);

                pcl::CRHAlignment<PointT, 90> alignment;
                alignment.setInputAndTargetView(cloud, model_cloud);
                // CRHAlignment works with Vector3f, not Vector4f.
                Eigen::Vector3f cluster_centroid_3f (cluster_centroid[0], cluster_centroid[1], cluster_centroid[2]);
                alignment.setInputAndTargetCentroids(cluster_centroid_3f, model_centroid);

                // Compute the roll transforms
                alignment.align(*cluster_crh, *model_crh);

                vector< Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > roll_transforms;
                alignment.getTransforms(roll_transforms);

                //create object hypothesis
                for (size_t j = 0; j < roll_transforms.size (); j++)
                {
                    ObjectHypothesis oh;
                    oh.model_id = model_name;
                    oh.model_template = model_cloud;
                    oh.transformation = roll_transforms[j];
                    object_hypotheses_.push_back(oh);
                }

                if(roll_transforms.size() > 0)
                {
                    std::cout << "Number of object hypotheses: " << object_hypotheses_.size() << "\n";

//                    cout << "Resulting roll transforms:\n";

//                    for(int j = 0; j < roll_transforms.size(); j++)
//                    {
//                        Eigen::Matrix3f rotation = roll_transforms[j].block<3,3>(0, 0);
//                        Eigen::Vector3f translation = roll_transforms[j].block<3,1>(0, 3);

//                        printf ("\nRoll transformation: %d\n", j);
//                        printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
//                        printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
//                        printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
//                        printf ("\n");
//                        printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
//                    }

//                    cout << "\n\n";
                }
                else
                {
                    cout << "No transforms found\n";
                }
            }
#endif

        }



    std::cout << "\n";


}


void recognize(pcl::PointCloud<PointT>::Ptr &cloud, pcl::PointCloud<PointT>::Ptr &cloud_filtered)
{
//    cout << "\n\n[recognize] Point cloud has size: " << cloud->points.size() << "\n";

    // Clear descriptors directory
    if(boost::filesystem::exists(base_descr_dir))
    {
        boost::filesystem::remove_all(base_descr_dir);
        boost::filesystem::create_directory(base_descr_dir);
    }

    preprocessCloud(cloud, cloud_filtered);

    segmentScene(cloud_filtered);

    for(int i = 0; i < cluster_clouds.size(); i++)
    {
        pcl::PointCloud<PointT>::Ptr cluster_cloud = cluster_clouds[i];

        stringstream path_ss;
        path_ss << "clusters/cluster_" << i << ".pcd";

        string cluster_file = path_ss.str();

        pcl::io::savePCDFileASCII(cluster_file.c_str(), *cluster_cloud);

        std::cout << cluster_file << " was saved\n";

        classifyCluster(i, cluster_cloud);
    }

    //
    // Choose the best object candidates for all the models in database
    //

    // Sort matches
    std::sort(models_scores.begin(), models_scores.end(), sortIndexScoresOp);

    std::cout << "Best model candidates:\n";
    for(size_t i = 0; i < models_scores.size(); i++)
    {
        index_score model_score = models_scores[i];
        printf("    %s: %f\n", model_score.model_id.c_str(), model_score.score);
    }

    // For every training model select the best match
    for(size_t i = 0; i < training_objects_ids.size(); i++)
    {
        string training_object = training_objects_ids[i];
        std::cout << "Looking for best candidate for object " << training_object << "\n";

        for(size_t j = 0; j < models_scores.size(); j++)
        {
            if(models_scores[j].model_id == training_object)
            {
                recognized_objects.push_back(models_scores[j].model_id);
                break;
            }
        }
    }

    std::cout << "Recognized objects in the scene:\n";

    for(size_t i = 0; i < recognized_objects.size(); i++)
    {
        std::cout << recognized_objects[i] << "\n";
    }


#ifndef DISABLE_SINGLE_OBJECT_RECOGNITION

    // Find the best candidate over all the clusters
    if(models_scores.size() > 0)
    {
        pcl::console::print_highlight("The best candidates over all the clusters\n");

        for(size_t i = 0; i < models_scores.size(); i++)
        {
            index_score model_score = models_scores[i];
            printf("    %s: %f\n", model_score.model_id.c_str(), model_score.score);
        }

        cout << "\n\n";

        // Method 1
//        float best_score = std::numeric_limits<float>::infinity();
//        index_score best_model;
//        for(size_t i = 0; i < models_scores.size(); i++)
//        {
//            index_score model_score = models_scores[i];
//            if(model_score.score < best_score){
//                best_model = model_score;
//                best_score = model_score.score;
//            }
//        }

        // Method 2
        std::sort(models_scores.begin(), models_scores.end(), sortIndexScoresOp);

        for(size_t i = 0; i < models_scores.size(); i++)
        {
            index_score model_score = models_scores[i];
            printf("    %s: %f\n", model_score.model_id.c_str(), model_score.score);
        }

        index_score best_model = models_scores[0];

        pcl::console::print_highlight("The best candidate\n");
        printf("    %s: %f\n\n", best_model.model_id.c_str(), best_model.score);

        cout << "\n\n";

        found_model = best_model.model_id;
    }
#endif

    // Clear data
    models_scores.clear();
}

void clearData()
{
    delete[] data.ptr();
}

