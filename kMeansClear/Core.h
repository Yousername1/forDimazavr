#pragma once
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h> 
#include <vector>
#include "time.h"
#include <ctime>

using namespace std;
using namespace cv;


class Core
{
private:
	Mat inputImage;
	Mat workingImage = inputImage.clone();
	Mat grayscaleImage = Mat::zeros(workingImage.rows, workingImage.cols, CV_8U);

	int clusterNumbers;
	vector<int> histogramVec;
	vector<int> centers;
	vector<int> predictedCenters;
	vector<int> intraclusterCounter;
	vector<vector<int>> centroids;

	void setVectorY(Mat histogram);
	vector<int> getVector();

	int getPosition(vector<int> histogramVector, int value);
	int findPeak(vector<int> vec);
	int findMin(vector<int> vec);
	


public:
	Core(const Mat inputImage);

	void setClusterNumbers(int clusterNumbers);
	int getClusterNumbers();

	Mat getGrayscaledImg(Mat inputImage);
	Mat getHist(Mat nonColorImage);

	vector<int> doPredict();
	void showPredict();

	void setCenters(vector<int> centers);
	void setCenters();
	void getCenters();

	Mat getCusteredImg();

	//DEBUG ONLY
	void showVector();

};

