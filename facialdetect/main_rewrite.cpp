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

bool IsCorrect(vector<Rect> faceVector,vector<Rect> eyeVector)
{
	if ((faceVector.size() == 1) && (eyeVector.size() == 2)) //数量校验
		if ((faceVector[0].x < eyeVector[0].x) && (faceVector[0].x < eyeVector[1].x) && (faceVector[0].y < eyeVector[0].y) && (faceVector[0].y < eyeVector[1].y))//眼睛在脸部之内
			if ((abs(eyeVector[0].width - eyeVector[1].width) < 10) && (abs(eyeVector[0].height - eyeVector[1].height) < 10))
				if (abs(eyeVector[0].y - eyeVector[1].y) < 10)
					return true;
	return false;
}
Mat FaceAlign(Mat raw,vector<Rect> faceVector, vector<Rect> eyeVector)
{
	const double SCALE = 1; //缩放比例
	const int LENGTH = (int)sqrt(raw.cols*raw.cols + raw.rows*raw.rows) * SCALE; //旋转后的矩阵大小
	Mat rotateMat(LENGTH, LENGTH, raw.type()); //变换后的矩阵
	Rect ROIRect((LENGTH - raw.cols) / 2, (LENGTH - raw.rows) / 2, raw.rows, raw.cols);
	Mat ROIMat(rotateMat, ROIRect); //对目标矩阵进行ROI选区
	raw.copyTo(ROIMat);
	//以上完成了拷贝图像的操作
	cv::Point2f center(LENGTH / 2, LENGTH / 2);
	double angle = atan((double)(eyeVector[0].y - eyeVector[1].y) / (eyeVector[0].x - eyeVector[1].x)) * 180 / CV_PI;
	Mat transfromeMat = cv::getRotationMatrix2D(center, angle, SCALE);
	Mat result(raw.rows,raw.cols, raw.type());
	cv::warpAffine(rotateMat, result, transfromeMat, cv::Size(LENGTH,LENGTH));
	return result;

}
int main()
{
	VideoCapture camera = CameraInit(); //初始化并设置摄像头
	bool flag = true;
	Mat rawFrame, grayFrame;
	vector<Rect> faceVector, eyeVector;
	cv::namedWindow("实时视频");
	cv::namedWindow("旋转后的内容");
	faceCC.load(FACE_CASCADE_PATH);
	eyeCC.load(EYE_CASCADE_PATH);
	while (true)
	{
		camera >> rawFrame;
		cv::cvtColor(rawFrame, grayFrame, cv::COLOR_RGB2GRAY);
		cv::imshow("实时视频", FaceEyeDetection(grayFrame, faceVector, eyeVector));
		if(IsCorrect(faceVector,eyeVector))
		{
				Mat raw = (Mat(grayFrame, faceVector[0])).clone();
				Mat rotate = FaceAlign(raw, faceVector, eyeVector);
				cv::imshow("旋转后的内容",rotate);
		}
		if (-1 != cv::waitKey(1)) break;

	}
	return 0;
}