//#define VFH_COMPUTE_DEBUG
#define DISABLE_COMPUTING_CRH

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

using namespace std;

bool calculate_crh (false);
bool calculate_vfh (false);

typedef std::pair<std::string, std::vector<float> > vfh_model;
typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT>::ConstPtr PointTConstPtr;
typedef pcl::VFHSignature308 FeatureT;
typedef pcl::Histogram<90> CRH90;

#ifndef DISABLE_COMPUTING_CRH
// Required for saving CRH histogram to PCD file
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<90>,
    (float[90], histogram, histogram90)
)
#endif

string training_dir;

float voxel_leaf_size (0.005);
float normal_radius (0.03);

/** \brief Loads an n-D histogram file as a VFH signature
  * \param path the input file name
  * \param vfh the resultant VFH model
  */
bool
loadHist (const boost::filesystem::path &path, vfh_model &vfh)
{
  int vfh_idx;
  // Load the file as a PCD
  try
  {
    pcl::PCLPointCloud2 cloud;
    int version;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    pcl::PCDReader r;
    int type; unsigned int idx;
    r.readHeader (path.string (), cloud, origin, orientation, version, type, idx);

    vfh_idx = pcl::getFieldIndex (cloud, "vfh");
    if (vfh_idx == -1)
      return (false);
    if ((int)cloud.width * cloud.height != 1)
      return (false);
  }
  catch (const pcl::InvalidConversionException&)
  {
    return (false);
  }

  // Treat the VFH signature as a single Point Cloud
  pcl::PointCloud <pcl::VFHSignature308> point;
  pcl::io::loadPCDFile (path.string (), point);
  vfh.second.resize (308);

  std::vector <pcl::PCLPointField> fields;
  pcl::getFieldIndex (point, "vfh", fields);

  for (size_t i = 0; i < fields[vfh_idx].count; ++i)
  {
    vfh.second[i] = point.points[0].histogram[i];
  }
//  string file_name = (path.filename()).string();
  vfh.first = path.string (); //  file_name;
  return (true);
}

/** \brief Load a set of VFH features that will act as the model (training data)
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param extension the file extension containing the VFH features
  * \param models the resultant vector of histogram models
  */
void
loadFeatureModels (const boost::filesystem::path &base_dir, const std::string &extension,
                   std::vector<vfh_model> &models)
{
    cout << "[loadFeatureModels]\n";
  if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
    return;

  for (boost::filesystem::directory_iterator it (base_dir); it != boost::filesystem::directory_iterator (); ++it)
  {
    if (boost::filesystem::is_directory (it->status ()))
    {
      std::stringstream ss;
      ss << it->path ();
      pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
      loadFeatureModels (it->path (), extension, models);
    }
    if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension)
    {
      vfh_model m;
      if (loadHist (base_dir / it->path ().filename (), m))
        models.push_back (m);
      m.second.clear();
    }
  }
}

void processCloud(pcl::PointCloud<PointT>::Ptr &in, pcl::PointCloud<PointT>::Ptr &out)
{
    vector<int> mapping;
    pcl::removeNaNFromPointCloud(*in, *in, mapping);

    // Downsampling
    pcl::VoxelGrid<PointT> vox_grid;
    vox_grid.setInputCloud(in);
    vox_grid.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

    pcl::PointCloud<PointT>::Ptr temp_cloud (new pcl::PointCloud<PointT> ());
    vox_grid.filter(*temp_cloud);

    out = temp_cloud;
}

void createFeatureModels (const boost::filesystem::path &base_dir, const std::string &extension)
{
    cout << "[createFeatureModels] Loading files in directory: " << base_dir << "\n";

    if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
      return;

    for (boost::filesystem::directory_iterator it (base_dir); it != boost::filesystem::directory_iterator (); ++it)
    {
        cout << "Process " << it->path().filename() << "...\n";
      if (boost::filesystem::is_directory (it->status ()))
      {
        std::stringstream ss;
        ss << it->path ();
//        pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
        createFeatureModels (it->path (), extension);
      }

      string file_name = (it->path().filename()).string();

      if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension && !strstr(file_name.c_str(), "vfh") && !strstr(file_name.c_str(), "crh"))
      {
          std::vector<string> strs;
          boost::split(strs, file_name, boost::is_any_of("."));
          string view_id = strs[0];
          strs.clear();

          stringstream path_ss;
          path_ss << base_dir.string() << "/" << view_id << "_vfh.pcd";
          string descr_file = path_ss.str();

          if(!boost::filesystem::exists(descr_file))
          {
              pcl::PointCloud<PointT>::Ptr view (new pcl::PointCloud<PointT> ());

              string full_file_name = it->path ().string (); // (base_dir / it->path ().filename ()).string();
              //          string file_name = (it->path ().filename ()).string();

              cout << "Compute VFH for " << full_file_name << "\n";

              pcl::io::loadPCDFile(full_file_name.c_str(), *view);

              cout << "Cloud has " << view->points.size() << " points\n";

              // Preprocess view cloud
              processCloud(view, view);

              cout << "Cloud has " << view->points.size() << " points after processing\n";

              pcl::PointCloud<NormalT>::Ptr normals (new pcl::PointCloud<NormalT> ());
              pcl::PointCloud<FeatureT>::Ptr descriptor (new pcl::PointCloud<FeatureT>);

              // Estimate the normals.
              pcl::NormalEstimation<PointT, NormalT> normalEstimation;
              normalEstimation.setInputCloud(view);

              normalEstimation.setRadiusSearch(normal_radius);
              pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
              normalEstimation.setSearchMethod(kdtree);

              // Alternative from local pipeline
//              int norm_k = 10;
//              normalEstimation.setKSearch(norm_k);
              normalEstimation.compute(*normals);

              // VFH estimation object.
              pcl::VFHEstimation<PointT, NormalT, FeatureT> vfh;
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

              pcl::io::savePCDFileBinary(descr_file.c_str(), *descriptor);
              cout << descr_file << " was saved\n";

#ifndef DISABLE_COMPUTING_CRH
              if(calculate_crh)
              {
                  std::cout << "Compute CRH features ...\n";
                  // Compute the CRH histogram
                  pcl::PointCloud<CRH90>::Ptr histogram (new pcl::PointCloud<CRH90>);

                  // CRH estimation object
                  pcl::CRHEstimation<PointT, NormalT, CRH90> crh;
                  crh.setInputCloud(view);
                  crh.setInputNormals(normals);
                  Eigen::Vector4f centroid4f;
                  pcl::compute3DCentroid(*view, centroid4f);
                  crh.setCentroid(centroid4f);

                  crh.compute(*histogram);

                  // Save centroid to file
                  path_ss.str("");
                  path_ss << base_dir.string() << "/" << view_id << "_centroid.txt";
                  string centroid_file = path_ss.str();
                  Eigen::Vector3f centroid (centroid4f[0], centroid4f[1], centroid4f[2]);

                  // TODO: Move to the PersistenceUtils class
                  std::ofstream out (centroid_file.c_str ());
                  if (!out)
                  {
                    std::cout << "Failed to open file " << centroid_file << " for saving centroid\n";
                  }

                  out << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
                  out.close ();

                  std::cout << centroid_file << " was saved\n";

                  stringstream roll_path;
                  roll_path << base_dir.string() << "/" << view_id << "_crh.pcd";

                  string roll_file = roll_path.str();

                  pcl::io::savePCDFileBinary(roll_file.c_str(), *histogram);
                  cout << roll_file << " was saved\n";
              }
#endif

          }
          else
          {
              pcl::console::print_highlight("Descriptor file %s already exists\n", descr_file.c_str());
          }

      }
    }
}

void showHelp(char* filename)
{
    std::cout << "*****************************************************************\n" <<
                 "*                                                               *\n" <<
                 "*           VFH Cluster classifier: Build tree                  *\n" <<
                 "*                                                               *\n" <<
                 "*****************************************************************\n" <<
                 "Usage: " << filename << " [model_directory] [options]\n" <<
                 "*options:\n" <<
                 "-vfh         - calculate VFH features\n" <<
                 "-crh         - calculate CRH features\n\n";
}

void parseCommandLine(int argc, char** argv)
{
    if (argc < 2)
    {
      PCL_ERROR ("Need at least two parameters!\n");
      showHelp(argv[0]);
      exit(-1);
    }

    training_dir = string(argv[1]);

    if(pcl::console::find_switch(argc, argv, "-vfh"))
    {
        calculate_vfh = true;
    }

    if(pcl::console::find_switch(argc, argv, "-crh"))
    {
        calculate_crh = true;
    }

    if(pcl::console::find_switch(argc, argv, "-h"))
    {
        showHelp(argv[0]);
        exit(0);
    }
}

int
main (int argc, char** argv)
{
    parseCommandLine(argc, argv);

  cout << "Calculate VFH: " << calculate_vfh << "\n";
  cout << "Calculate CRH: " << calculate_crh << "\n";

  cout << "Training dir: " << training_dir << "\n";

  std::string extension (".pcd");
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

  std::string kdtree_idx_file_name = training_dir + "/kdtree.idx";
  std::string training_data_h5_file_name = training_dir + "/training_data.h5";
  std::string training_data_list_file_name = training_dir + "/training_data.list";

  std::vector<vfh_model> models;

  // Remove previously saved flann index and data files
  if(boost::filesystem::exists(kdtree_idx_file_name))
  {
      if(remove(kdtree_idx_file_name.c_str()) != 0)
          perror("Error deleting old flann index file");
      else
          cout << "Old flann index file was successfully deleted\n";
  }

  if(boost::filesystem::exists(training_data_h5_file_name))
  {
      if(remove(training_data_h5_file_name.c_str()) != 0)
          perror("Error deleting old training data file");
      else
          cout << "Old training data file was successfully deleted\n";
  }

  if(calculate_vfh)
    createFeatureModels(argv[1], extension);

#ifndef VFH_COMPUTE_DEBUG
  // Load the model histograms
  loadFeatureModels (argv[1], extension, models);
  pcl::console::print_highlight ("Loaded %d VFH models. Creating training data %s/%s.\n",
      (int)models.size (), training_data_h5_file_name.c_str (), training_data_list_file_name.c_str ());

  // Convert data into FLANN format
  flann::Matrix<float> data (new float[models.size () * models[0].second.size ()], models.size (), models[0].second.size ());

  for (size_t i = 0; i < data.rows; ++i)
    for (size_t j = 0; j < data.cols; ++j)
      data[i][j] = models[i].second[j];

  // Save data to disk (list of models)
  flann::save_to_file (data, training_data_h5_file_name, "training_data");
  std::ofstream fs;
  fs.open (training_data_list_file_name.c_str ());
  for (size_t i = 0; i < models.size (); ++i)
    fs << models[i].first << "\n";
  fs.close ();

  models.clear();

  // Build the tree index and save it to disk
  pcl::console::print_error ("Building the kdtree index (%s) for %d elements...\n", kdtree_idx_file_name.c_str (), (int)data.rows);
  flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
  //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
  index.buildIndex ();
  index.save (kdtree_idx_file_name);
  delete[] data.ptr ();
#endif

  return (0);
}
