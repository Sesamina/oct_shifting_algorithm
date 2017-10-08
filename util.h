#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <string>
#include <algorithm>

#include "vtk_model_sampling.h"

namespace bf = boost::filesystem;

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', (int)path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}


//get the path to all models in a directory
void getModelsInDirectory(bf::path& dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext) {
	bf::directory_iterator end_itr;
	for (bf::directory_iterator itr(dir); itr != end_itr; ++itr) {
		//check that it is a ply file and then add, otherwise ignore..
		std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
		std::string file = (itr->path().filename()).string();
#else
		std::string file = (itr->path()).filename();
#endif

		boost::split(strs, file, boost::is_any_of("."));
		std::string extension = strs[strs.size() - 1];

		if (extension.compare(ext) == 0)
		{
#if BOOST_FILESYSTEM_VERSION == 3
			std::string path = rel_path_so_far + (itr->path().filename()).string();
#else
			std::string path = rel_path_so_far + (itr->path()).filename();
#endif

			relative_paths.push_back(path);
		}
	}
}

//generate a point cloud from a ply file
void generatePointCloudFromModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& modelCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& model_voxelized, std::string path) {
	//get models in directory
	std::vector < std::string > files;
	std::string start = "";
	std::string ext = std::string("ply");
	bf::path dir = path;
	getModelsInDirectory(dir, start, files, ext);
	std::stringstream model_path;
	model_path << path << "/" << files[0];
	std::string path_model = model_path.str();
	//sample points on surface of model
	uniform_sampling(path_model, 100000, *modelCloud, 1.f);
	//downsample points
	float VOXEL_SIZE_ICP_ = 0.02f;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
	voxel_grid_icp.setInputCloud(modelCloud);
	voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
	voxel_grid_icp.filter(*model_voxelized);

	Eigen::Matrix4f rotationZ;
	rotationZ << 0, 1, 0, 0,
		-1, 0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;
	pcl::transformPointCloud(*modelCloud, *modelCloud, rotationZ);
	pcl::transformPointCloud(*model_voxelized, *model_voxelized, rotationZ);
}

//cut the model in half in given direction, 0 for x-axis, 1 for y-axis, 2 for z-axis
void cutModelinHalf(pcl::PointCloud<pcl::PointXYZ>::Ptr& modelCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& result, int axis) {
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*modelCloud, centroid);
	for (int i = 0; i < modelCloud->points.size(); i++) {
		pcl::PointXYZ point = modelCloud->points.at(i);
		if (axis == 0) {
			if (point.x <= centroid.x()) {
				result->push_back(point);
			}
		}
		else if (axis == 1) {
			if (point.y >= centroid.y()) {
				result->push_back(point);
			}
		}
		else if (axis == 2) {
			if (point.z < centroid.z()) {
				result->push_back(point);
			}
		}
	}
}

//cut a part of the back of the model off, until specified value, keep only part before the value
void cutPartOfModel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cut, float to_cut_until) {
	for (int i = 0; i < cloud->points.size(); i++) {
		if (cloud->points.at(i).z < to_cut_until) {
			cut->push_back(cloud->points.at(i));
		}
	}
}

//get the model size in z direction
float getModelSize(pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud) {
	float min = modelCloud->points.at(0).z;
	float max = modelCloud->points.at(0).z;
	for (int i = 0; i < modelCloud->points.size(); i++) {
		if (modelCloud->points.at(i).z < min) {
			min = modelCloud->points.at(i).z;
		}
		else if (modelCloud->points.at(i).z > max) {
			max = modelCloud->points.at(i).z;
		}
	}
	float modelSize = std::abs(max - min);
	return modelSize;
}

//find the index of the pair with maximal value
auto findMaxIndexOfVectorOfPairs(std::vector<std::pair<float, float>> map) {
	auto max_index = std::distance(map.begin(), std::max_element(map.begin(), map.end(),
		[](const std::pair<float, float>& p1, const std::pair<float, float>& p2) {
		return p1.second < p2.second; }));
	return max_index;
}

//find the index of the tuple with maximal value
auto findMaxIndexOfVectorOfTuples(std::vector<std::tuple<float, float, float>> tuples) {
	auto max_index = std::distance(tuples.begin(), std::max_element(tuples.begin(), tuples.end(),
		[](const std::tuple<float, float, float>& p1, const std::tuple<float, float, float>& p2) {
		return std::get<2>(p1) < std::get<2>(p2); }));
	return max_index;
}

//check for a given index if it is possible to go a given number of steps in both directions or if it would be out of bounds
int checkMinBoundsForVectorIndex(int steps, int index_in_vector) {
	int angle_min = steps;
	if (!(index_in_vector > 1)) {
		angle_min--;
		if (!(index_in_vector > 0)) {
			angle_min--;
		}
	}
	return angle_min;
}
int checkMaxBoundsForVectorIndex(int steps, int index_in_vector, int vector_size) {
	int angle_max = steps;
	if (!(index_in_vector < (vector_size - 2))) {
		angle_max--;
		if (!(index_in_vector < (vector_size - 1))) {
			angle_max--;
		}
	}
	return angle_max;
}

float checkMinBoundsForValue(float value, float start, float step) {
	float val = value - step;
	if (val > start) {
		if (val - step >= start) {
			return val - step;
		}
		return val;
	}
	return start;
}

float checkMaxBoundsForValue(float value, float end, float step) {
	float val = value + step;
	if (val < end) {
		if (val + step <= end) {
			return val + step;
		}
		return val;
	}
	return end;
}

//compute middle of points at a specified z-value
float computeMiddle(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, float z) {
	float x_min = 2.0f;
	float x_max = -2.0f;
	for (int i = 0; i < point_cloud_ptr->points.size(); i++) {
		pcl::PointXYZ point = point_cloud_ptr->at(i);
		if (point.z == z) {
			if (point.x < x_min) {
				x_min = point.x;
			}
			if (point.x > x_max) {
				x_max = point.x;
			}
		}
	}
	if (x_max != x_min) {
		return x_min + ((x_max - x_min) / 2.0f);
	}
	else {
		return x_min;
	}
}

//compute minimum z-value of cloud
float getMinZValue(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	float z = 2.0f;
	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZ point = cloud->at(i);
		if (point.z < z) {
			z = point.z;
		}
	}
	return z;
}

//compute minimum point of cloud
pcl::PointXYZ getMinPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::PointXYZ z(0.0f, 0.0f, 5.0f);
	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZ point = cloud->at(i);
		if (point.z < z.z) {
			z = point;
		}
	}
	return z;
}