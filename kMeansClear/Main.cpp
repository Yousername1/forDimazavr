#include "Core.h"

void main()
{

    Mat img = imread("../images/img1.jpg");
    imshow("Source img", img);

    Core wImg(img);
    wImg.getGrayscaledImg(img);
    imshow("grayscaledImage", wImg.getGrayscaledImg(img));
    imshow("Histogram", wImg.getHist(wImg.getGrayscaledImg(img)));

    wImg.setClusterNumbers(3); //NADO SDELAT VVOD S KONSOLI

    //DEBUG ONLY
    wImg.showVector();
    cout << endl << endl << endl;
    cout << wImg.getClusterNumbers() << endl << endl;
    cout << endl << endl << endl;
    //END OF DEGBUG

    wImg.makePredict();
    cout << "Extremums of histogram (max min): ";
    wImg.showPredict();
    cout << endl << endl << endl;

    cout << "Enter " << wImg.getClusterNumbers() << " centers: ";
    int a, b, c; //THINK ABOUT IT
    cin >> a >> b >> c; // SAME
    //vector<int> cent = { 0, 135, 255 };
    //vector<int> cent = { rand() % 256, rand() % 256, rand() % 256 };
    vector<int> cent = { a, b, c };
    wImg.setCenters(cent);

    //DEBUG ONLY
    //cout << endl << endl << endl;
    //wImg.getCenters();
    //END OF DEGBUG

    imshow("Clustered Image", wImg.getCusteredImg());
    imshow("HistClus", wImg.getHist(wImg.getCusteredImg()));


    cv::waitKey();
}