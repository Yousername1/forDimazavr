#include "Core.h"

void main()
{

    Mat img = imread("../images/img1.jpg");
    imshow("Source img", img);

    Core wImg(img);
    imshow("grayscaledImage", wImg.getGrayscaledImg(img));
    imshow("Histogram", wImg.getHist(wImg.getGrayscaledImg(img)));
    cout << endl << endl << endl;
    cv::waitKey(1);


    cout << "Enter number of clusters: ";
    int clustersFromUser = 0;
    cin >> clustersFromUser;
    wImg.setClusterNumbers(clustersFromUser);

    //DEBUG ONLY
    //wImg.showVector();
    //cout << endl << endl << endl;
    //END OF DEBUG

    wImg.doPredict();
    cout << "Extremums of histogram (max min): ";
    wImg.showPredict();
    cout << endl;

    wImg.setCenters();


    imshow("Clustered Image", wImg.getCusteredImg());
    imshow("HistClus", wImg.getHist(wImg.getCusteredImg()));
    cout << endl << endl << endl;

    cout << "Final centers: ";
    wImg.getCenters();

    cv::waitKey();
}