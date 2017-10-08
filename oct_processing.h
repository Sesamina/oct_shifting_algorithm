
//fixed number of OCT images
#define NUM_FRAMES 128
//scale of OCT cube
#define SCALE_X 2.7
#define SCALE_Y 2.4
#define SCALE_Z 3.0

//-------------------------------------
//helper method to generate a PointXYZ
//-------------------------------------
void generatePoint(pcl::PointXYZ& point, float x, float y, float z, float width, float height) {
	point.x = (float)x / width * SCALE_X;
	point.y = (float)y / height * SCALE_Y;
	point.z = (float)z / NUM_FRAMES * SCALE_Z;
}

//------------------------------------------------------------
//convert labelled image (opencv matrix) to points for cloud
//------------------------------------------------------------
void MatToPointXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, std::vector<cv::Point>& elipsePoints, int z,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width)
{
	//get the infos for the bounding box
	int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
	int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
	int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
	int labelHeight = labelInfo.at<int>(0, cv::CC_STAT_HEIGHT);
	int leftHeight = 0;
	int rightHeight = 0;
	//go through points in bounding box 
	for (int i = x; i < x + labelWidth; i++) {
		//indicate if first point with intensity = 1 in row has been found
		bool firstNotFound = true;
		//position of last point with intensity = 1 in row
		int lastPointPosition = 0;
		for (int j = y; j < y + labelHeight; j++)
		{
			if (OpencVPointCloud.at<unsigned char>(j, i) >= 1.0f) {
				if (firstNotFound) {
					firstNotFound = false;
				}
				lastPointPosition = j;
				if (i == x) {
					leftHeight = j;
				}
				if (i == x + labelWidth - 1) {
					rightHeight = j;
				}
			}
		}
		if (!firstNotFound) {
			//add the last point with intensity = 1 in row to the point cloud
			pcl::PointXYZ point;
			generatePoint(point, i, lastPointPosition, z, width, height);
			point_cloud_ptr->points.push_back(point);
			elipsePoints.push_back(cv::Point(i, lastPointPosition));
		}
	}
}

//----------------------------------------------
//process the OCT frame to get a labelled image
//----------------------------------------------
void processOCTFrame(cv::Mat imageGray, int number, boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>& needle_width) {
	//flip and transpose the image
	cv::Mat transposedOCTimage;
	cv::flip(imageGray, imageGray, 0);

	//set a threshold (0.26)
	cv::Mat thresholdedImage;
	cv::threshold(imageGray, thresholdedImage, 0.26 * 255, 1, 0);

	//use a median blur filter
	cv::Mat filteredImage;
	cv::medianBlur(thresholdedImage, filteredImage, 3);

	//label the image
	cv::Mat labelledImage;
	cv::Mat labelStats;
	cv::Mat labelCentroids;
	int numLabels = cv::connectedComponentsWithStats(filteredImage, labelledImage, labelStats, labelCentroids);

	//for every label with more than 400 points process it further for adding points to the cloud
	for (int i = 1; i < numLabels; i++) {
		//original threshold at 400
		if (labelStats.at<int>(i, cv::CC_STAT_AREA) > 250) {
			cv::Mat labelInfo = labelStats.row(i);
			//save bounding box width for finding the point where needle gets smaller
			needle_width->push_back(std::tuple<int, int, cv::Mat, cv::Mat>(number, labelStats.at<int>(i, cv::CC_STAT_WIDTH), filteredImage, labelInfo));
		}
	}
}

//-----------------------------------
//setup oct point cloud for alignment
//-----------------------------------
boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>>
recognizeOCT(pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr& peak_points, std::string oct_dir, bool only_tip) {

	std::string oct_directory = getDirectoryPath(oct_dir);
	//count oct images
	int minFrameNumber = 0;
	int maxFrameNumber = NUM_FRAMES;

	//tuple with frame number, bounding box width, filteredImage, labelInfo
	boost::shared_ptr<std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>> needle_width(new std::vector<std::tuple<int, int, cv::Mat, cv::Mat>>);
	cv::Mat imageGray;
	{
		pcl::ScopeTime t("Process OCT images");
		//	go through all frames
		for (int number = minFrameNumber; number < maxFrameNumber; number++)
		{
			//get the next frame
			std::stringstream filename;
			if (number < 100) {
				filename << "0";
			}
			if (number < 10) {
				filename << "0";
			}
			filename << number << ".bmp";
			//read the image in grayscale
			imageGray = cv::imread(oct_directory.c_str() + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

			processOCTFrame(imageGray, number, needle_width);

			cv::waitKey(10);
		}

		//---------------------------------------------
		//optionally cut needle tip off
		//---------------------------------------------
		int end_index = needle_width->size();
		//regression to find cutting point where tip ends
		if (only_tip) {
			end_index = regression(needle_width);
		}
		//go through all frames
		for (int w = 0; w < end_index; w++) {
			std::tuple<int, int, cv::Mat, cv::Mat> tup = needle_width->at(w);
			std::vector<cv::Point> elipsePoints;
			MatToPointXYZ(std::get<2>(tup), std::get<3>(tup), elipsePoints, std::get<0>(tup), point_cloud_ptr, imageGray.rows, imageGray.cols);

			//compute center point of needle frame for translation
			if (elipsePoints.size() >= 50) { //to remove outliers, NOT RANSAC
				cv::RotatedRect elipse = cv::fitEllipse(cv::Mat(elipsePoints));
				pcl::PointXYZ peak;
				generatePoint(peak, elipse.center.x, elipse.center.y, std::get<0>(tup), imageGray.cols, imageGray.rows);
				peak_points->push_back(peak);
			}
		}
	}

	//downsample pointcloud
	float VOXEL_SIZE_ICP_ = 0.02f;
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_icp;
	voxel_grid_icp.setInputCloud(point_cloud_ptr);
	voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
	voxel_grid_icp.filter(*point_cloud_ptr);

	return needle_width;
}