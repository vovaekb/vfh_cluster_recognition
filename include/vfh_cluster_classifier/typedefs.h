// types definition
using PointType = pcl::PointXYZRGB;

using DepthPointType = pcl::PointXYZ;
using NormalType = pcl::Normal;
using FeatureType = pcl::VFHSignature308;

using CRH90 = pcl::Histogram<90>; // Camera Roll Histograms
using PointCloudType = pcl::PointCloud<PointType>;
using DepthPointCloudType = pcl::PointCloud<DepthPointType>;
using NormalCloudType = pcl::PointCloud<NormalType>;
using FeatureCloudType = pcl::PointCloud<FeatureType>;
using CRHCloudType = pcl::PointCloud<CRH90>;
using PointCloudPtr = PointCloudType::Ptr;
using DepthPointCloudTypePtr = pcl::PointCloud<DepthPointType>::Ptr;
using NormalCloudTypePtr = pcl::PointCloud<NormalType>::Ptr;
using FeatureCloudTypePtr = pcl::PointCloud<FeatureType>::Ptr;
using CRHCloudTypePtr = pcl::PointCloud<CRH90>::Ptr;
using PointCloudConstPtr = PointCloudType::ConstPtr;
using FeatureType = pcl::VFHSignature308;
using FeatureCloudType = pcl::PointCloud<FeatureType>;
using FeatureCloudTypePtr = FeatureCloudType::Ptr;
using NormalCloudTypePtr = NormalCloudType::Ptr;

using vfh_model = std::pair<std::string, std::vector<float>>;
using flann_distance_metric = flann::Index<flann::ChiSquareDistance<float>>;

using CRHEstimationPtr = pcl::CRHEstimation<PointType, NormalType, CRH90>;