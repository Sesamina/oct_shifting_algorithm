#pragma once

#include "opencv2\opencv.hpp"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>

//------------------------------------------------------------------
//fit a line through the given points
//source: https://gist.github.com/ialhashim/0a2554076a6cf32831ca
//------------------------------------------------------------------
std::pair<Eigen::Vector3f, Eigen::Vector3f> fitLine(std::vector<Eigen::Vector3f> points) {
	// copy coordinates to  matrix in Eigen format
	size_t num_atoms = points.size();
	Eigen::Matrix< Eigen::Vector3f::Scalar, Eigen::Dynamic, Eigen::Dynamic > centers(num_atoms, 3);
	for (size_t i = 0; i < num_atoms; ++i) centers.row(i) = points[i];

	Eigen::Vector3f origin = centers.colwise().mean();
	Eigen::MatrixXf centered = centers.rowwise() - origin.transpose();
	Eigen::MatrixXf cov = centered.adjoint() * centered;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
	Eigen::Vector3f axis = eig.eigenvectors().col(2).normalized();
	//multiply with -1 so that it points towards origin
	return std::make_pair(origin, axis * -1.0f);
}

//---------------------------------
//linear regression for 2D points
//---------------------------------
double linearRegression(std::vector<std::tuple<int, int>> input) {
	std::vector<Eigen::Vector3f> points;
	for (int i = 0; i < input.size(); i++) {
		points.push_back(Eigen::Vector3f(std::get<0>(input.at(i)), std::get<1>(input.at(i)), 0.0f));
	}
	std::pair<Eigen::Vector3f, Eigen::Vector3f> res = fitLine(points);
	double errorSum = 0.0f;
	for (int i = 0; i < points.size(); i++) {
		errorSum += pcl::sqrPointToLineDistance(Eigen::Vector4f(points.at(i).x(), points.at(i).y(), points.at(i).z(), 0.0f),
			Eigen::Vector4f(res.first.x(), res.first.y(), res.first.z(), 0.0f), Eigen::Vector4f(res.second.x(), res.second.y(), res.second.z(), 0.0f));
	}
	return errorSum;
}

//---------------------------------------------
//use RANSAC to get inliers from given points
//---------------------------------------------
std::vector<int> getInliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points) {
	std::vector<int> inliers;
	pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr
		model(new pcl::SampleConsensusModelLine<pcl::PointXYZ>(peak_points));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model);
	ransac.setDistanceThreshold(.1f);
	ransac.computeModel();
	ransac.getInliers(inliers);
	return inliers;
}

// regress the variable t in the equation
// y = m * x + t
// when m is fixed
// for the given input values
double regress_t_with_fixed_m(std::vector<std::tuple<int, int>> pos, double m)
{
	double n = pos.size();

	double accum = 0.0;
	for (int i = 0; i < n; i++)
	{
		accum += std::get<1>(pos[i]) - m * std::get<0>(pos[i]);
	}
	double error = 0.0;
	double t = accum / n;
	for (int j = 0; j < n; j++) {
		double tmp = (std::get<1>(pos[j]) - t) * (std::get<1>(pos[j]) - t);
		error += tmp;
	}

	return error / n;
}

//-----------------------------------------------------------------------
//helper method for summing up errors of different regression methods
//-----------------------------------------------------------------------
double regress_split_at(std::vector<std::tuple<int, int>> part_a, std::vector<std::tuple<int, int>> part_b)
{
	double error_a = linearRegression(part_a);
	double error_b = regress_t_with_fixed_m(part_b, 0.0);
	return error_a + error_b;
}

//---------------------------------------------------------------
//get the index of in z-direction where the needle tip has ended
//---------------------------------------------------------------
int regression(boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width) {
	std::vector<std::tuple<int, int>> widths;
	for (int i = 0; i < needle_width->size(); i++) {
		widths.push_back(std::tuple<int, int>(std::get<0>(needle_width->at(i)), std::get<1>(needle_width->at(i))));
	}
	std::vector<double> errors;
	for (int j = 3; j < widths.size(); j++) {
		errors.push_back(regress_split_at(std::vector<std::tuple<int, int>>(widths.begin(), widths.begin() + j), std::vector<std::tuple<int, int>>(widths.begin() + j, widths.end())));
	}
	int error_min_index = 0;
	for (int k = 0; k < errors.size(); k++) {
		if (errors[k] < errors[error_min_index]) {
			error_min_index = k;
		}
	}
	int index = error_min_index;
	//add number of frames at which oct cloud starts
	return index + std::get<0>(widths.at(0));
}