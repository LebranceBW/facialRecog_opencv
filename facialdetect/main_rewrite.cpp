#include "stdafx.h"
using cv::Mat;
using cv::VideoCapture;
using cv::CascadeClassifier;
using std::vector;
using cv::Rect;
using cv::selectROI;
const std::string FACE_CASCADE_PATH = "D:\\OpencvSDK\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml";
const std::string EYE_CASCADE_PATH = "D:\\OpencvSDK\\build\\etc\\haarcascades\\haarcascade_eye.xml";

CascadeClassifier faceCC, eyeCC;
VideoCapture CameraInit()
{
	VideoCapture *cap = new VideoCapture(0);
	if (!cap->isOpened()) throw std::runtime_error("摄像头打开失败");
	else
	{
		cap->set(cv::CAP_PROP_FPS, 30);
		cap->set(cv::CAP_PROP_FRAME_WIDTH, 400);
		cap->set(cv::CAP_PROP_FRAME_HEIGHT, 300);
	}
	return *cap;
}
Mat FaceEyeDetection(const Mat frame,vector<Rect> &face,vector<Rect> &eye)
{
	faceCC.detectMultiScale(frame, face);
	eyeCC.detectMultiScale(frame, eye);
	Mat *temp = new Mat();
	frame.copyTo(*temp);
	for (auto r : face)
		cv::rectangle(*temp, r, cv::Scalar(255));
	for (auto r : eye)
		cv::rectangle(*temp, r, cv::Scalar(0));
	return *temp;
}
int main()
{
	VideoCapture camera = CameraInit(); //初始化并设置摄像头
	bool flag = true;
	Mat rawFrame, grayFrame;
	vector<Rect> faceVector, eyeVector;
	cv::namedWindow("实时视频");
	cv::namedWindow("旋转后的内容");
	cv::namedWindow("面部内容");
	faceCC.load(FACE_CASCADE_PATH);
	eyeCC.load(EYE_CASCADE_PATH);
	while (true)
	{
		camera >> rawFrame;
		cv::cvtColor(rawFrame, grayFrame, cv::COLOR_RGB2GRAY);
		cv::imshow("实时视频", FaceEyeDetection(grayFrame, faceVector, eyeVector));
		if(flag&&(faceVector.size() == 1)&&(eyeVector.size() == 2))
			if (((eyeVector[0].y - eyeVector[1].y) <= 15) || ((eyeVector[1].y - eyeVector[0].y) <= 15))
			{
				Mat raw = (Mat(grayFrame, faceVector[0])).clone();
				//flag = false;
				Mat rotate;
				
				//const int ANGLE = 30;
				const double SCALE = 1.1;
				int length = sqrt(raw.cols*raw.cols + raw.rows*raw.rows) * SCALE;
				Mat tempMat(length, length, raw.type());
				Rect ROIRect((length / 2 - raw.cols / 2), (length / 2 - raw.cols / 2), raw.cols, raw.rows);
				Mat temp2Mat(tempMat, ROIRect);
				raw.copyTo(temp2Mat);
				cv::Point2f center(length / 2, length / 2);

				double angle = atan((double)(eyeVector[0].y - eyeVector[1].y) / (eyeVector[0].x - eyeVector[1].x)) * 180 / CV_PI;

				Mat transfromeMat = cv::getRotationMatrix2D(center, angle, SCALE);
				cv::warpAffine(tempMat,rotate,transfromeMat,cv::Size(length,length));
			
				
				cv::imshow("面部内容", raw);
				cv::imshow("旋转后的内容",rotate);
			}
		if (-1 != cv::waitKey(1)) break;

	}
	return 0;
}