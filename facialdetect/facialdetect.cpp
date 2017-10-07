#include "stdafx.h"
using cv::Mat;
using cv::imread;
using cv::namedWindow;
using cv::imshow;
using cv::waitKey;
using cv::VideoCapture;
using std::vector;
using std::runtime_error;
using std::string;
bool isLoop = true;
string FACE_CASCADE_FILE = "G:\\opencv3\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml";
string EYE_CASCADE_FILE = "G:\\opencv3\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml";
vector<cv::Rect> FaceDetection_fun(cv::CascadeClassifier cc,Mat &raw,Mat &face)
{
	vector<cv::Rect> *result = new vector<cv::Rect>;
	cc.detectMultiScale(raw, *result);
	for (auto r : *result)
		cv::rectangle(raw, r, cv::Scalar(255, 0, 255));
	return *result;
}

vector<cv::Rect> EyeDetection_fun(cv::CascadeClassifier cc, Mat &raw, Mat &face)
{
	vector<cv::Rect> *result = new vector<cv::Rect>;
	cc.detectMultiScale(raw, *result);
	for (auto r : *result)
		cv::rectangle(raw, r, cv::Scalar(0, 0, 255));
	return *result;
}
int main()
{
	VideoCapture cap(0);
	if (!cap.isOpened()) throw runtime_error("invaid camera");
		cap.set(cv::CAP_PROP_FPS, 30);
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 400);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 320);

		Mat frame,face;
		namedWindow("实时视频");
		namedWindow("面部画面");
		cv::CascadeClassifier face_detection;
		face_detection.load(FACE_CASCADE_FILE);
		cv::CascadeClassifier eye_detection;
		eye_detection.load(EYE_CASCADE_FILE);

		while (isLoop)
		{
			Mat temp;
			cap>>temp;
			cv::cvtColor(temp, frame, cv::COLOR_RGB2GRAY);
			vector<cv::Rect> faceVector = FaceDetection_fun(face_detection,frame,face);
			vector<cv::Rect> eyeVector = EyeDetection_fun(eye_detection, frame, face);
			imshow("实时视频", frame);
			if ((faceVector.size() == 1) && (eyeVector.size() == 2))
				if (((eyeVector[0].y - eyeVector[1].y)<10)|| ((eyeVector[1].y - eyeVector[0].y)<10))
					imshow("面部画面", frame);
			waitKey(1);
		}
		return 0;
}

