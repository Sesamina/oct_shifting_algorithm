//--------------------------------------
//compute needle rotation
//--------------------------------------
Eigen::Matrix3f computeNeedleRotation(std::pair<Eigen::Vector3f, Eigen::Vector3f> direction) {
	Eigen::Vector3f zRotation = std::get<1>(direction);
	Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
	Eigen::Vector3f xRotation = up.cross(zRotation);
	xRotation.normalize();
	Eigen::Vector3f yRotation = zRotation.cross(xRotation);
	yRotation.normalize();
	Eigen::Matrix3f rotation;
	rotation << xRotation.x(), yRotation.x(), zRotation.x(),
		xRotation.y(), yRotation.y(), zRotation.y(),
		xRotation.z(), yRotation.z(), zRotation.z();
	return rotation;
}

//------------------------------------
//compute needle translation
//------------------------------------
Eigen::Vector3f computeNeedleTranslation(float tangencyPoint, Eigen::Vector3f pointOnOCTCloud, Eigen::Vector3f direction, float halfModelSize) {
	if (direction.z() < 0) {
		direction *= -1;
	}
	Eigen::Vector3f translation = pointOnOCTCloud;
	float dist = std::abs(pointOnOCTCloud.z() - tangencyPoint);
	float mult = std::abs(dist / direction.z());
	if (pointOnOCTCloud.z() < tangencyPoint) {
		translation += direction * mult;
	}
	else if (pointOnOCTCloud.z() > tangencyPoint) {
		translation -= direction * mult;
	}
	translation -= (halfModelSize / direction.z()) * direction;
	return translation;
}

//------------------------------------------------
//rotate point cloud around z axis by given angle
//------------------------------------------------
Eigen::Matrix3f rotateByAngle(float angleInDegrees, Eigen::Matrix3f currentRotation) {
	Eigen::Matrix3f rotationZ;
	Eigen::Matrix3f finalRotation = currentRotation;
	float angle = angleInDegrees * M_PI / 180.0f;
	rotationZ << std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle), 0, 0, 0, 1;
	finalRotation *= rotationZ;
	return finalRotation;
}

//---------------------------------------------------------
// compute translation given how much it should be shifted
//---------------------------------------------------------
Eigen::Vector3f shiftByValue(float shift, Eigen::Vector3f currentTranslation, Eigen::Vector3f direction) {
	Eigen::Vector3f finalTranslation = currentTranslation;
	finalTranslation += direction * (shift / direction.z());
	return finalTranslation;
}

//-----------------------------------------------------------------
// build transformation matrix from given rotation and translation
//-----------------------------------------------------------------
Eigen::Matrix4f buildTransformationMatrix(Eigen::Matrix3f rotation, Eigen::Vector3f translation) {
	Eigen::Matrix4f transformation;
	transformation.block(0, 0, 3, 3) = rotation;
	transformation.col(3).head(3) = translation;
	transformation.row(3) << 0, 0, 0, 1;
	return transformation;
}

//--------------------------------------------
//get z-rotation from transformation matrix
//--------------------------------------------
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
