
#include "opencv2\opencv.hpp"

#include <pcl/pcl_macros.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/common/time.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/voxel_grid.h>

#include <tuple>

#include "util.h"
#include "regression.h"
#include "oct_processing.h"
#include "graphUtils/GraphUtils.h"
#include "transformations.h"

int global_video_ctr = 0;

//----------------------------------------
//compute needle direction
//----------------------------------------
std::pair<Eigen::Vector3f, Eigen::Vector3f> computeNeedleDirection(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr peak_inliers(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<int> inliers = getInliers(peak_points);
	for (int i = 0; i < inliers.size(); i++) {
		peak_inliers->push_back(peak_points->at(inliers.at(i)));
	}
	std::vector<Eigen::Vector3f> peak_positions;
	for (int i = 0; i < peak_inliers->points.size(); i++) {//for RANSAC use peak_inliers, else peak_points
		pcl::PointXYZ point = peak_inliers->points.at(i); //for RANSAC use peak_inliers, else peak_points
		Eigen::Vector3f eigenPoint(point.x, point.y, point.z);
		peak_positions.push_back(eigenPoint);
	}
	peak_points = peak_inliers; //only when using RANSAC
	return fitLine(peak_positions);
}


//-----------------------------
// compute correspondences
//-----------------------------
float computeCorrespondences(Eigen::Matrix4f& guess, pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointCloud<pcl::PointXYZ>::Ptr target, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& corr_cloud, bool video) {
	// Point cloud containing the correspondences of each point in <input, indices>
	pcl::PointCloud<pcl::PointXYZ>::Ptr input_transformed(new pcl::PointCloud<pcl::PointXYZ>);

	// If the guessed transformation is non identity
	if (guess != Eigen::Matrix4f::Identity())
	{
		input_transformed->resize(input->size());
		// Apply passed transformation
		pcl::transformPointCloud(*input, *input_transformed, guess);
	}
	else
		*input_transformed = *input;

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>::Ptr correspondence_estimation(
		new pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>);
	// Pass in the default target for the Correspondence Estimation code
	correspondence_estimation->setInputTarget(target);

	// Set the source
	correspondence_estimation->setInputSource(input_transformed);
	boost::shared_ptr<pcl::Correspondences> correspondences(new pcl::Correspondences);

	// Estimate correspondences
	correspondence_estimation->determineCorrespondences(*correspondences, 0.02f);
	boost::shared_ptr<pcl::Correspondences> temp_correspondences(new pcl::Correspondences(*correspondences));
	/*pcl::registration::CorrespondenceRejectorMedianDistance::Ptr rejector_median(new pcl::registration::CorrespondenceRejectorMedianDistance);
	rejector_median->setInputCorrespondences(temp_correspondences);
	rejector_median->getCorrespondences(*correspondences);*/

	//VIDEO
	if (video) {
		corr_cloud->clear();
		for (int i = 0; i < correspondences->size(); i++) {
			corr_cloud->push_back(input->at(correspondences->at(i).index_query));
		}
		pcl::transformPointCloud(*corr_cloud, *corr_cloud, guess);
	}

	//get number of correspondences
	size_t cnt = correspondences->size();
	return (float)cnt;
}


//-------------------------------------------------------------------
//shifting/roll in defined intervals without summing up correspondences
//-------------------------------------------------------------------
void shift_and_roll_without_sum(float angle_min, float angle_max, float angle_step,
	float shift_min, float shift_max, float shift_step,
	std::vector<std::tuple<float, float, float>>& count,
	Eigen::Matrix3f rotation, Eigen::Vector3f initialTranslation, Eigen::Vector3f direction,
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& modelTransformed,
	boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewerForVideo,
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler5, bool video, std::string video_path) {
	int num_angle_steps = std::round((angle_max - angle_min) / angle_step) + 1;
	int num_shift_steps = std::round((shift_max - shift_min) / shift_step) + 1;
	for (int i = 0; i < (num_angle_steps) * (num_shift_steps); i++) {
		count.push_back(std::tuple<float, float, float>(0, 0, 0));
	}
	//VIDEO
	pcl::PointCloud<pcl::PointXYZ>::Ptr corr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//don't parallelize for VIDEO
#pragma omp parallel for num_threads(omp_get_num_procs()) if (!video)
	for (int angle = 0; angle < num_angle_steps; angle++) {
		for (int shift = 0; shift < num_shift_steps; shift++) {
			Eigen::Matrix3f rot = rotateByAngle((float)angle_min + angle * angle_step, rotation);
			Eigen::Vector3f trans = shiftByValue((float)shift_min + shift * shift_step, initialTranslation, direction);
			Eigen::Matrix4f transform = buildTransformationMatrix(rot, trans);
			float correspondence_count = computeCorrespondences(transform, model_voxelized, point_cloud_ptr, corr_cloud, video);
			count.at(angle * num_shift_steps + shift) = std::tuple<float, float, float>(angle_min + angle * angle_step, 
				shift_min + shift * shift_step, correspondence_count);
			//VIDEO
			if (video) {
				pcl::transformPointCloud(*model_voxelized, *modelTransformed, transform);
				viewerForVideo->updatePointCloud(modelTransformed, rgb_handler5, "model transformed");
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler6(corr_cloud, 255, 0, 0);
				if (!viewerForVideo->updatePointCloud(corr_cloud, rgb_handler6, "correspondences")) {
					viewerForVideo->addPointCloud(corr_cloud, rgb_handler6, "correspondences");
					viewerForVideo->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "correspondences");
				}
				std::stringstream fileName;
				fileName << video_path << global_video_ctr++ << ".png";
				viewerForVideo->saveScreenshot(fileName.str());
				viewerForVideo->spinOnce(100);
			}
		}
	}
}

float computeTipX(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::pair<Eigen::Vector3f, Eigen::Vector3f> origin_and_direction_needle, float x_middle_OCT) {
	pcl::PointXYZ min(0.0f, 0.0f, 5.0f);
	for (int i = 0; i < cloud->points.size(); i++) {
		pcl::PointXYZ point = cloud->at(i);
		if (point.z < min.z) {
			min = point;
		}
	}
	Eigen::VectorXf line1(6);
	line1 << x_middle_OCT, 0.0f, getMinZValue(cloud), std::get<1>(origin_and_direction_needle)(0), 0.0f, std::get<1>(origin_and_direction_needle)(2);
	Eigen::VectorXf line2(6);
	line2 << min.x, 0.0f, min.z, std::get<1>(origin_and_direction_needle)(2), 0.0f, -std::get<1>(origin_and_direction_needle)(0);
	Eigen::Vector4f point;
	pcl::lineWithLineIntersection(line1, line2, point);
	return point.x();
}

//------------------------------------------------------
// tip approximation
//------------------------------------------------------
Eigen::Matrix4f tipApproximation(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& modelTransformed,
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized, std::pair<Eigen::Vector3f, Eigen::Vector3f> direction, const Eigen::Matrix4f& transformation) {
	Eigen::Matrix4f transform = transformation;
	float x_middle_OCT = computeMiddle(point_cloud_ptr, getMinZValue(point_cloud_ptr));

	float z_min = getMinZValue(modelTransformed);
	float x_middle_model = computeMiddle(modelTransformed, z_min);

	Eigen::Vector3f OCT_point(x_middle_OCT, 0.0f, 0.0f);
	float x_in_direction = computeTipX(modelTransformed, direction, x_middle_OCT);


	float angle_to_rotate = 0.5f;
	float sign = 1.0f;
	{
		pcl::ScopeTime t("Tip Approximation");
		float first = 0.0f;
		float second = 0.0f;
		float r = 0.0f;
		if (x_middle_model < x_in_direction) {
			sign = -1.0f;
			first = x_middle_model;
			second = x_in_direction;
		}
		else if (x_middle_model > x_in_direction) {
			sign = 1.0f;
			first = x_in_direction;
			second = x_middle_model;
		}
		while (r < 360.0f && first < second) {
			transform = buildTransformationMatrix(rotateByAngle(sign * angle_to_rotate, transform.block(0, 0, 3, 3)), transform.block(0, 3, 3, 0));
			pcl::transformPointCloud(*model_voxelized, *modelTransformed, transform);
			if (sign < 0) {
				first = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
			}
			else {
				second = computeMiddle(modelTransformed, getMinZValue(modelTransformed));
			}
			r += angle_to_rotate;
		}
	}
	return transform;
}

float getAngleFromMatrix(const Eigen::Matrix4f& transformation) {
	float angle = 0.0f;
	Eigen::Matrix3f end_rot = transformation.block(0, 0, 3, 3);
	Eigen::Vector3f eulerAngles = end_rot.eulerAngles(0, 1, 2);
	eulerAngles *= 180 / M_PI;
	std::cout << eulerAngles << std::endl;
	if (eulerAngles.z() < 0) {
		angle = -180 - eulerAngles.z();
	}
	else {
		angle = 180 - eulerAngles.z();
	}
	std::cout << "angle: " << angle << std::endl;
	angle *= -1.0f;
	return angle;
}

void printHelp()
{
	pcl::console::print_error("Syntax is: .\oct_shift -models_dir -oct_dir -only_tip -shift -video <-video_dir>\n");
	pcl::console::print_info("  where arguments are:\n");
	pcl::console::print_info("                     -models_dir = directory where CAD model in .ply format is located, \n");
	pcl::console::print_info("                     -oct_dir = directory where OCT images are located, \n");
	pcl::console::print_info("                     -only_tip = 0 if whole OCT cloud should be used, 1 if only tip should be used, \n");
	pcl::console::print_info("                     -video = 1 if screenshots of algorithm for video should be taken, 0 if not, \n");
	pcl::console::print_info("                     -video_dir = necessary if video is set to 1. \n");
}


int main(int argc, char ** argv)
{
	//------------------------------
	//parse command line arguments
	//------------------------------
	if (argc < 9)
	{
		printHelp();
		return (-1);
	}

	std::string path = "models/";
	std::string oct_dir = "oct/";
	std::string video_path = "video/";
	//use only needle tip or not
	bool only_tip = false;
	//assemble screenshots for video
	bool video = false;

	pcl::console::parse_argument(argc, argv, "-models_dir", path);
	pcl::console::parse_argument(argc, argv, "-oct_dir", oct_dir);
	pcl::console::parse_argument(argc, argv, "-only_tip", only_tip);
	if (pcl::console::parse_argument(argc, argv, "-video", video) == 1)
	{
		if (pcl::console::parse_argument(argc, argv, "-video_dir", video_path) == -1)
		{
			PCL_ERROR("Need an output directory for video! Please use -video_dir to continue.\n");
			return (-1);
		}
	}


	//-------------------------------------
	//process OCT images
	//-------------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_not_cut(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr peak_points(new pcl::PointCloud<pcl::PointXYZ>);
	boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width = recognizeOCT(point_cloud_not_cut, peak_points, oct_dir, only_tip);
	//cutPartOfModel(point_cloud_not_cut, point_cloud_ptr, 1.3f);
	//cutModelinHalf(point_cloud_not_cut, point_cloud_ptr, 2);

	//-------------------------------
	//shifting algorithm
	//-------------------------------

	//-------------------------------
	//process the CAD mdoel
	//-------------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr modelCloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_voxelized(new pcl::PointCloud<pcl::PointXYZ>());
	generatePointCloudFromModel(modelCloud, model_voxelized, path);
	//cut oct cloud
	cutPartOfModel(point_cloud_not_cut, point_cloud_ptr, getModelSize(model_voxelized) - 0.1f + getMinZValue(point_cloud_not_cut));

	//--------------------------------------
	//compute initial translation/rotation
	//--------------------------------------
	//compute the 3d direction of the needle
	std::pair<Eigen::Vector3f, Eigen::Vector3f> direction = computeNeedleDirection(peak_points);
	std::cout << "origin: " << std::endl << std::get<0>(direction) << std::endl << "direction: " << std::endl << std::get<1>(direction) << std::endl;

	//compute the 3d rotation of the needle
	Eigen::Matrix3f rotation = computeNeedleRotation(direction);
	std::cout << "rotation matrix: " << std::endl << rotation << std::endl;
	//rotate back to 0 degree on z axis
	Eigen::Vector3f euler = rotation.eulerAngles(0, 1, 2) * 180 / M_PI;
	rotation = rotateByAngle(180 - euler.z(), rotation);
	std::cout << "euler angles: " << std::endl << rotation.eulerAngles(0, 1, 2) * 180 / M_PI << std::endl;

	//compute 3d translation of the needle
	float tangencyPoint = regression(needle_width) / (float)NUM_FRAMES * SCALE_Z; //scaling
	std::cout << "tangency point: " << tangencyPoint << std::endl;
	Eigen::Vector3f initialTranslation = computeNeedleTranslation(tangencyPoint, std::get<0>(direction), 
		std::get<1>(direction), getModelSize(model_voxelized) / 2);
	std::cout << "translation: " << std::endl << initialTranslation << std::endl;

	//build the transformation matrix with currently computed rotation and translation
	Eigen::Matrix4f transformation = buildTransformationMatrix(rotation, initialTranslation);

	//transform point cloud to initial values
	pcl::PointCloud<pcl::PointXYZ>::Ptr modelTransformed(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*model_voxelized, *modelTransformed, transformation);

	//VIDEO
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerForVideo(new pcl::visualization::PCLVisualizer("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler5(modelTransformed, 0, 255, 255);
	if (video) {
		viewerForVideo->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler3(point_cloud_ptr, 0, 255, 0);
		viewerForVideo->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler3, "oct cloud");
		viewerForVideo->addPointCloud<pcl::PointXYZ>(modelTransformed, rgb_handler5, "model transformed");
		viewerForVideo->addCoordinateSystem(2.0);
		viewerForVideo->initCameraParameters();
		viewerForVideo->setCameraPosition(1.45732, 2.56393, -1.49624, -0.127368, 0.760336, 0.63692);
		viewerForVideo->spinOnce();
		std::stringstream fileName;
		fileName << video_path << global_video_ctr++ << ".png";
		viewerForVideo->saveScreenshot(fileName.str());
	}

	//----------------------
	//tip approximation
	//----------------------
	transformation = tipApproximation(point_cloud_ptr, modelTransformed, model_voxelized, direction, transformation);
	float end_angle = getAngleFromMatrix(transformation);

	//VIDEO
	if (video) {
		viewerForVideo->updatePointCloud(modelTransformed, rgb_handler5, "model transformed");
		viewerForVideo->spinOnce();
		std::stringstream filename;
		filename << video_path << global_video_ctr++ << ".png";
		viewerForVideo->saveScreenshot(filename.str());
	}

	//--------------------------------------
	//start of shifting algorithm
	//--------------------------------------
	//angle, shift, count in one vector
	std::vector<std::tuple<float, float, float>> correspondence_count;
	//angle and count
	std::vector<std::pair<float, float>> angle_count;
	//shift and count
	std::vector<std::pair<float, float>> shift_count;

	//initialize interval values
	float angleStart = end_angle - 5.0f;
	float angleEnd = end_angle + 5.0f;
	float angleStep = 1.0f;
	float shiftStart = 0.0f;
	float shiftEnd = 0.5;
	float shiftStep = 0.05f;
	//more initialization
	int max_index_angles = 0;
	int max_index_shift = 0;
	int correspondence_index = 0;
	float max_angle = 0.0f;
	float max_shift = 0.0f;

	{
		pcl::ScopeTime t("Shift and Roll");
		for (int i = 0; i < 4; i++) {
			angle_count.clear();
			shift_count.clear();
			correspondence_count.clear();
			//apply shift and roll in small steps in given intervals and compute correspondences
			shift_and_roll_without_sum(angleStart, angleEnd, angleStep, shiftStart, shiftEnd, shiftStep, correspondence_count, rotation, initialTranslation, 
				std::get<1>(direction), model_voxelized, point_cloud_ptr, modelTransformed, viewerForVideo, rgb_handler5, video, video_path);

			//fill count correspondences for all angles and all shifts					
			for (int i = 0; i < correspondence_count.size(); i++) {
				std::tuple<float, float, float> current = correspondence_count.at(i);
				float angle_tmp = std::get<0>(current);
				float shift_tmp = std::get<1>(current);
				float count_tmp = std::get<2>(current);
				std::vector<std::pair<float, float>>::iterator it;
				it = std::find_if(angle_count.begin(), angle_count.end(), [angle_tmp](const std::pair<float, float>& p1) {
					return p1.first == angle_tmp; });
				if (it != angle_count.end()) {
					angle_count.at(std::distance(angle_count.begin(), it)).second += count_tmp;
				}
				else {
					angle_count.push_back(std::pair<float, float>(angle_tmp, count_tmp));
				}
				it = std::find_if(shift_count.begin(), shift_count.end(), [shift_tmp](const std::pair<float, float>& p1) {
					return p1.first == shift_tmp; });
				if (it != shift_count.end()) {
					shift_count.at(std::distance(shift_count.begin(), it)).second += count_tmp;
				}
				else {
					shift_count.push_back(std::pair<float, float>(shift_tmp, count_tmp));
				}
			}

			//find index of maximum correspondences
			max_index_angles = findMaxIndexOfVectorOfPairs(angle_count);
			max_index_shift = findMaxIndexOfVectorOfPairs(shift_count);
			correspondence_index = findMaxIndexOfVectorOfTuples(correspondence_count);
			max_angle = std::get<0>(angle_count.at(max_index_angles));
			max_shift = std::get<0>(shift_count.at(max_index_shift));

			//check bounds of vectors to make sure that in both directions of max indices you can go as far as specified
			angleStart = checkMinBoundsForValue(max_angle, angleStart, angleStep);
			angleEnd = checkMaxBoundsForValue(max_angle, angleEnd, angleStep);
			shiftStart = checkMinBoundsForValue(max_shift, shiftStart, shiftStep);
			shiftEnd = checkMaxBoundsForValue(max_shift, shiftEnd, shiftStep);

			//assign new interval values
			angleStep /= 5.0f;
			shiftStep /= 5.0f;
			std::cout << "angle: " << max_angle * -1 << std::endl;
			std::cout << "shift: " << max_shift << std::endl;
			std::cout << "end of round: " << i << std::endl;

			//show correspondence count as graph
			/*std::vector<float> angle_corr;
			for (int j = 0; j < angle_count.size(); j++) {
				angle_corr.push_back(angle_count.at(j).second);
			}
			showFloatGraph("Angle Correspondences", &angle_corr[0], angle_corr.size(), 0);
			std::vector<float> shift_corr;
			for (int j = 0; j < shift_count.size(); j++) {
				shift_corr.push_back(shift_count.at(j).second);
			}
			showFloatGraph("Shift Correspondences", &shift_corr[0], shift_corr.size(), 0);*/

			//debugging
			/*for (int i = 0; i < shift_count.size(); i++) {
				std::cout << shift_count.at(i).first << " : " << shift_count.at(i).second << std::endl;
			}*/
			/*for (int i = 0; i < correspondence_count.size(); i++) {
				std::cout << std::get<0>(correspondence_count.at(i)) << " : " << std::get<1>(correspondence_count.at(i)) << " : " 
				<< std::get<2>(correspondence_count.at(i)) << std::endl;
			}*/
		}
	}

	transformation = buildTransformationMatrix(rotateByAngle(max_angle, rotation), shiftByValue(max_shift, initialTranslation, std::get<1>(direction)));
	pcl::transformPointCloud(*model_voxelized, *modelTransformed, transformation);

	//VIDEO
	if (video) {
		viewerForVideo->updatePointCloud(modelTransformed, rgb_handler5, "model transformed");
		viewerForVideo->spinOnce();
	}

	//------------------------------------------------------
	// tip approximation
	//------------------------------------------------------
	transformation = tipApproximation(point_cloud_ptr, modelTransformed, model_voxelized, direction, transformation);
	end_angle = getAngleFromMatrix(transformation);

	//VIDEO
	if (video) {
		viewerForVideo->updatePointCloud(modelTransformed, rgb_handler5, "model transformed");
		viewerForVideo->spinOnce();
		std::stringstream file_name;
		file_name << video_path << global_video_ctr++ << ".png";
		viewerForVideo->saveScreenshot(file_name.str());
		viewerForVideo->spin();
	}

	//get final position
	Eigen::Vector4f centroid_transformed;
	pcl::compute3DCentroid(*modelTransformed, centroid_transformed);
	std::cout << "position: " << centroid_transformed;

	//--------------------------------
	//visualization
	//--------------------------------
	pcl::PointXYZ point2;
	point2.x = std::get<0>(direction).x() + 2 * std::get<1>(direction).x();
	point2.y = std::get<0>(direction).y() + 2 * std::get<1>(direction).y();
	point2.z = std::get<0>(direction).z() + 2 * std::get<1>(direction).z();
	pcl::PointXYZ point3;
	point3.x = std::get<0>(direction).x() - 2 * std::get<1>(direction).x();
	point3.y = std::get<0>(direction).y() - 2 * std::get<1>(direction).y();
	point3.z = std::get<0>(direction).z() - 2 * std::get<1>(direction).z();
	pcl::PointCloud<pcl::PointXYZ>::Ptr debug(new pcl::PointCloud<pcl::PointXYZ>);
	debug->push_back(pcl::PointXYZ(computeMiddle(point_cloud_ptr, getMinZValue(point_cloud_ptr)), getMinPoint(point_cloud_ptr).y, getMinZValue(point_cloud_ptr)));
	debug->push_back(pcl::PointXYZ());
	//show model
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(model_voxelized, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(model_voxelized, rgb_handler, "model");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler6(point_cloud_ptr, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, rgb_handler6, "oct cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler4(peak_points, 255, 10, 10);
	viewer->addPointCloud<pcl::PointXYZ>(peak_points, rgb_handler4, "peak points");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handlert(debug, 255, 10, 10);
	viewer->addPointCloud<pcl::PointXYZ>(debug, rgb_handlert, "debug");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler7(modelTransformed, 0, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(modelTransformed, rgb_handler7, "model transformed");
	viewer->addLine(point2, point3, "line");
	viewer->addCoordinateSystem(2.0);
	viewer->initCameraParameters();
	viewer->spin();
}

