#include "Core.h"

Core::Core(const Mat inputImage)
{
	this->inputImage = inputImage;
}


void Core::setClusterNumbers(int clusterNumbers)
{
	this->clusterNumbers = clusterNumbers;
}

int Core::getClusterNumbers()
{
	return clusterNumbers;
}

Mat Core::getGrayscaledImg(Mat inputImage)
{
	cvtColor(inputImage, grayscaleImage, COLOR_BGR2GRAY);
	return grayscaleImage;
}

void Core::setVectorY(Mat histogram)
{
	vector<int> vec(histogram.cols, 0);
	this->histogramVec = vec;
}

//DEBUG ONLY
void Core::showVector()
{
	for (int i = 0; i < histogramVec.size(); i++) {
		cout << histogramVec[i] << ", ";
	}
}
//END OF DEBUG

vector<int> Core::getVector() {
	return histogramVec;
}


Mat Core::getHist(Mat nonColorImage)
{
	Mat hist = Mat::zeros(1, 256, CV_64FC1);

	Core::setVectorY(hist);

	for (int i = 0; i < nonColorImage.cols; i++) {
		for (int j = 0; j < nonColorImage.rows; j++) {
			int y = nonColorImage.at<unsigned char>(j, i);
			hist.at<double>(0, y) += 1;
			histogramVec[y] += 1;
		}
	}

	double m = 0, M = 0;
	minMaxLoc(hist, &m, &M);
	hist = hist / M;

	Mat histImg = Mat::zeros(100, 256, CV_8U);

	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 100; j++) {
			if (hist.at<double>(0, i) * 100 > j) {
				histImg.at<unsigned char>(99 - j, i) = 255;
			}
		}
	}

	bitwise_not(histImg, histImg);

	return histImg;
}


int Core::getPosition(vector<int> histogramVector, int value)
{
	//iterator??
	auto it = find(histogramVector.begin(), histogramVector.end(), value);
	if (it == histogramVector.end()) {
		cout << "Error.";
	}

	return it - histogramVector.begin();
}


int Core::findPeak(vector<int> histogramVector)
{
	vector<int> copy = histogramVector;
	sort(copy.begin(), copy.end());

	int max = *max_element(copy.begin(), copy.end());

	return Core::getPosition(histogramVector, max);
}

int Core::findMin(vector<int> histogramVector)
{
	vector<int> copy = histogramVector;
	sort(copy.begin(), copy.end());

	int min = *min_element(copy.begin(), copy.end());

	return Core::getPosition(histogramVector, min);
}



vector<int> Core::doPredict()
{
	predictedCenters.push_back(Core::findPeak(histogramVec));
	predictedCenters.push_back(Core::findMin(histogramVec));
	
	return predictedCenters;
}

void Core::showPredict()
{
	for (int i = 0; i < predictedCenters.size(); i++) {
		cout << predictedCenters[i] << " ";
	}
}


void Core::setCenters(vector<int> centers)
{
	this->centers = centers;
}

void Core::setCenters()
{
	cout << "Warning: extra centers will be deleted." << endl;
	cout << "Enter " << Core::getClusterNumbers() << " centers: ";
	int currentCenter = 0;
	vector<int> consoleTypingCenters(Core::getClusterNumbers(), 0);
	for (int i = 0; i < consoleTypingCenters.size(); i++) {
		cin >> currentCenter;
		consoleTypingCenters[i] = currentCenter;
	}
	cout << endl << endl << endl;	
	Core::setCenters(consoleTypingCenters);
}


void Core::getCenters()
{
	for (int i = 0; i < centers.size(); i++) {
		cout << centers[i] << " ";
	}
}


Mat Core::getCusteredImg()
{
	Mat clusteredImage = Mat::zeros(grayscaleImage.rows, grayscaleImage.cols, CV_8U);

	int y = 0;
	int y_out = 0;
	int D = 0;
	int min = 0;
	vector<int> distances(centers.size(), 0);
	vector<vector<int>> clusters(Core::getClusterNumbers());
	vector<int> temp(centers.size(), 0);
	vector<int> currentCenters;

	do {

		currentCenters.clear();
		temp = centers;

		for (int i = 0; i < Core::getClusterNumbers(); i++) {
			clusters[i].clear();
		}

		for (int i = 0; i < grayscaleImage.cols; i++) {
			for (int j = 0; j < grayscaleImage.rows; j++) {
				y = grayscaleImage.at<uchar>(j, i);

				for (int k = 0; k < temp.size(); k++) {
					D = sqrt(pow((y - temp[k]), 2));
					distances[k] = D;
				}

				min = *min_element(distances.begin(), distances.end());
				y_out = temp[Core::getPosition(distances, min)];

				int m = 0;
				while (y_out != temp[m]) {
					m++;
				}

				clusters[m].push_back(y);
				clusteredImage.at<uchar>(j, i) = y_out;
			}
		}

		int mediana = 0;
		for (vector<int> row : clusters) {
			int sum = 0;
			for (int val : row) {
				sum += val;
			}
			if (row.size() != 0)
			{
				mediana = sum / row.size();
				currentCenters.push_back(mediana);
			}
			else if (row.size() == 0) {
				int num = 0;
				for (int i = 0; i < centers.size(); i++)
				{
					do {
						num = 0 + rand() % 255;
					} while (num == centers[i]);
				}
				currentCenters.push_back(num);
			}
		}

		centers = currentCenters;

	} while (centers != temp);

	return clusteredImage;
}


