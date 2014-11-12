#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "../common/HelperFunctions.h"

#include <cstdio>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

void paintAndDisplayResult();
void doWatersheds();

static void help()
{
    cout << "\nModified Watersheds Segmentation with better UI\n"
            "Usage:\n"
            "./watershed [image_name]\n" << endl;

	cout << "Mouse Bindings:\n"
		"Left Mouse:		 Draw Foreground Seeds\n"
		"Left Mouse + CTRL:  Draw Background Seeds / Erase pixels from regions\n"
		"Right Mouse:		 Erase Foreground/Backgorund Seeds\n"
		"Right Mouse + CTRL: Remove connected Seed Region completely\n"
		"\n";

    cout << "Hot keys: \n"
        "\tESC - quit the program\n"
		"\tu   - undo last step\n"
        "\tc   - clear seed mask\n"
		"\tC   - restore original image while keeping the seeds\n"
        "\tw or SPACE - run watershed segmentation algorithm\n"
        "\t        (before running it, *roughly* mark the areas to segment on the image)\n"
        "\t        (before that, roughly outline several markers on the image)\n"
		"\tg,G - apply 3x3 (5x5) gaussian filter to base image\n"
		"\tm,M - apply 3x3 (5x5) median filter to base image\n"
		"\tl[1-9] - (press l, then number) apply CLAHE enhancement of specified kernel size to base image\n"
		"\tb,B - apply bilateral filter\n"
		"\tr   - toggle recording\n"
		"\t1,2,3 - extract channel 1/2/3 from the image as grayscale\n"
		"\te   - erode image\n"
		"\tt   - threshold image using Otsu's method\n"
		"\th   - fill holes in binarized image\n"
		"\ts   - find automatic seeds\n";
}
Mat markerMask, img, markers, wshed, imgGray, img0, img_orig, wresultBoundaries, wshedResult;
Mat img_undo, markerMask_undo;
int compCount = 0;
Point prevPt(-1, -1);
vector<Vec3b> colorTab;

std::string recordingPattern = "";
bool recordingEnabled = false;
int recordingImageID = 1;
cv::Mat currentDisplay;

void recordFrame(std::string winName)
{
	if(!recordingEnabled)
		return;

	char filename[1024];
	sprintf_s(filename, recordingPattern.c_str(), winName.c_str(), recordingImageID);
	recordingImageID++;
	cv::imwrite(filename, currentDisplay);
}

void show()
{
	static cv::Mat mask3c;

	if(img.channels() == 3) {
		cv::cvtColor(markerMask, mask3c, CV_GRAY2BGR);
		currentDisplay = img + mask3c + wresultBoundaries;
		imshow("image", currentDisplay);
	}
	else {
		assert(false);
	}
	imshow("mask", markerMask);
}

static void onMouse( int event, int x, int y, int flags, void* )
{
	static bool drawingBackground = false;

    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;

	if(event == CV_EVENT_LBUTTONUP || event == CV_EVENT_RBUTTONUP ) {
		doWatersheds();
	}
    if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) ) {
        prevPt = Point(-1,-1);
	}
    else if( event == CV_EVENT_LBUTTONDOWN ) {
		img_undo = img.clone(); markerMask_undo = markerMask.clone();
		drawingBackground = (flags & CV_EVENT_FLAG_CTRLKEY) > 0;
        prevPt = Point(x,y);
		cv::circle(markerMask, prevPt, 1, cv::Scalar::all(drawingBackground?128:255), 5);
		show();
	}
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(drawingBackground?128:255), 5, 8, 0 );
        prevPt = pt;
        show();
    }
	
	if(event == CV_EVENT_RBUTTONDOWN) {
		std::cout << "R" << std::endl;
		if(flags & CV_EVENT_FLAG_CTRLKEY) {
			cv::floodFill(markerMask, Point(x,y), 0);
			show();
		}
		else {
			cv::circle(markerMask, prevPt, 1, cv::Scalar::all(0), 12);
			show();
		}
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_RBUTTON)) {
		Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(0), 12, 8, 0 );
        prevPt = pt;
        show();
	}
}

inline int safeAt(int i, int j) {
	if(i < 0 || i >= img.rows || j < 0 || j >= img.cols)		return -1;
	return markers.at<int>(i,j);
}
inline bool checkNeighbor(int i, int j, int &nClass) 
{
	int nc = safeAt(i, j) ;
	if(nc == -1) return true;
	if(nc != nClass) {
		if(nClass == -1)	nClass = nc;
		else				return false;
	}
	return true;
}

void removeBoundaries() {
	for(int i=0; i < markers.rows; i++)
		for(int j=0; j < markers.cols; j++) {
			if(markers.at<int>(i,j) == -1) {
				//check neighbors if they are either boundary or same class, if yes, we can assign this pixel to the class
				int nClass = -1;

				if(checkNeighbor(i+1,j,nClass) && checkNeighbor(i-1,j,nClass) && checkNeighbor(i,j+1,nClass) && checkNeighbor(i,j-1,nClass) &&
					checkNeighbor(i+1,j+1,nClass) && checkNeighbor(i+1,j-1,nClass) && checkNeighbor(i-1,j+1,nClass) && checkNeighbor(i-1,j-1,nClass))
					markers.at<int>(i,j) = nClass;	
			}
		}
}

cv::Mat findBoundaries() {
	cv::Mat boundaries(markers.size(), CV_8UC3);
	for(int i=0; i < markers.rows; i++)
		for(int j=0; j < markers.cols; j++) {
			//check neighbors if they are either boundary or same class, if yes, we can assign this pixel to the class
			int nClass = markers.at<int>(i,j);

			if(!checkNeighbor(i+1,j,nClass) || !checkNeighbor(i-1,j,nClass) || !checkNeighbor(i,j+1,nClass) || !checkNeighbor(i,j-1,nClass) || 
				!checkNeighbor(i+1,j+1,nClass) || !checkNeighbor(i+1,j-1,nClass) || !checkNeighbor(i-1,j+1,nClass) || !checkNeighbor(i-1,j-1,nClass)) 
			{
				boundaries.at<cv::Point3_<unsigned char>>(i,j) = cv::Point3_<unsigned char>(255,255,0);
			}
		}
	return boundaries;
}


static void onMouseComponentMap( int event, int x, int y, int flags, void* )
{
	static cv::Point downPt(-1, -1);
	static int markerAtDown = -1;
	static bool drawingBackground = false;

    if( x < 0 || x >= markers.cols || y < 0 || y >= markers.rows )
        return;
    if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) ) {
        downPt = Point(-1,-1);
	}
    else if( event == CV_EVENT_LBUTTONDOWN ) {
        downPt = Point(x,y);
		markerAtDown = markers.at<int>(y, x);
		std::cout << "Marker is " << markerAtDown << std::endl;
	}
	else if(event == CV_EVENT_RBUTTONDOWN || event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_RBUTTON) ) {
		if(markers.at<int>(y,x) >= 0 && markers.at<int>(y,x) <= compCount) {
			floodFill(markers, cv::Point(x, y), -2);
			paintAndDisplayResult();
			std::cout << "Cleared at " << y << " " << x << " to -2" << std::endl;
		}
	}
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) ) {
		if(markerAtDown != markers.at<int>(x, y)) {
			//flood fill the markers image from the current seed point and display new result
			floodFill(markers, cv::Point(x, y), markerAtDown);
			removeBoundaries();
			paintAndDisplayResult();
			std::cout << "Joined Region at " << y << " " << x << " to " << markerAtDown << std::endl;
		}
	}
}

void paintAndDisplayResult()
{
	int i, j;
	    // paint the watershed image
    for( i = 0; i < markers.rows; i++ )
        for( j = 0; j < markers.cols; j++ )
        {
            int index = markers.at<int>(i,j);
            if( index == -1 )
                wshedResult.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > compCount )
                wshedResult.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                wshedResult.at<Vec3b>(i,j) = colorTab[index - 1];
        }

	wresultBoundaries = findBoundaries();
	show();

    wshed = wshedResult*0.5 + imgGray*0.5;
	cv::namedWindow("watershed transform", CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("watershed transform", 600, 600);
    imshow( "watershed transform", wshed );
	cv::setMouseCallback("watershed transform", onMouseComponentMap);
	recordFrame("Watershed_Result");
}

int makeComponentMap()
{
	vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

	findContours(cv::Mat(markerMask - 128), contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int compCount = contours.size();

    if( contours.empty() ) {
		std::cout << "No contours found in mask. Is the mask empty?" << std::endl;
        return 0;
	}
    markers = cv::Mat(markerMask.size(), CV_32S);
    cv::Mat(markerMask == 128).convertTo(markers, CV_32S, compCount / 256.);
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
        drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

	return compCount;
}

void doWatersheds()
{
	int i;
	compCount = 0;


	//we need a 3ch-image for watersheds
	if(img.channels() == 1)
		cv::cvtColor(img, img, CV_GRAY2BGR);

    compCount = makeComponentMap();

    if( compCount == 0 )
        return;

    for( i = colorTab.size(); i < compCount; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    double t = (double)getTickCount();
    watershed( img, markers );
    t = (double)getTickCount() - t;
    printf( "execution time = %gms\n", t*1000./getTickFrequency() );

    wshed = cv::Mat(markers.size(), CV_8UC3);
	wshedResult = wshed.clone();

	paintAndDisplayResult();
}

int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"data/FVF ID2 - Series002 - Z35 - C1.png";
    img0 = imread(filename, 1);
	std::string imageLocation(filename);
	img_orig = img0.clone();

    if( img0.empty() )
    {
        cout << "Couldn't open image " << filename << ". Usage: watershed <image_name>\n";
        return 0;
    }
    help();
    namedWindow( "image", CV_WINDOW_KEEPRATIO );
	namedWindow("mask", CV_WINDOW_KEEPRATIO);

    img0.copyTo(img);
    cvtColor(img, markerMask, CV_BGR2GRAY);
    cvtColor(markerMask, imgGray, CV_GRAY2BGR);
    	
	markerMask = 0;
    show();
    setMouseCallback( "image", onMouse, 0 );

	img_undo = img.clone(); markerMask_undo = markerMask.clone();

	Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 3, 3 ), Point( 2, 2) );
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Mat> channels;

	time_t now;
	struct tm now_tm;
	char timestamp[1024];

    for(;;)
    {
        int c = waitKey(0);

        if( (char)c == 27 ) {
			size_t found = imageLocation.find_last_of("/\\");
			std::string inputFolder = imageLocation.substr(0,found);
			std::string t = imageLocation.substr(found+1);
			found = t.find_last_of(".");
			std::string inputFilename = t.substr(0, found);
			std::string outputFolder = inputFolder + "/results";
			system(std::string(std::string("mkdir \"") + outputFolder + "\"").c_str());
			ExportMat(outputFolder + "/" + inputFilename + ".png", wshedResult);
			break;
		}
	
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(40.0, cv::Size(3, 3));
		cv::Mat tmp;
		std::string folder;

		switch((char)c) {
		case 'r':
			recordingEnabled = !recordingEnabled;
			std::cout << "Recording " << (recordingEnabled?"ENABLED":"DISABLED") << std::endl;
			if(recordingEnabled) {
				time(&now);
				localtime_s(&now_tm, &now);
				strftime(timestamp, 1024, "%H%M%S_%Y%m%d", &now_tm);
				folder = std::string("rec_ws_") + timestamp;
				recordingPattern = folder + "/%s %4i.png";
				std::cout << "Output folder is " << recordingPattern << std::endl;
				system(std::string(std::string("mkdir \"") + folder + "\"").c_str());
			}
			break;
		case 'C': //reset to original
            //markerMask = Scalar::all(0);
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
            img_orig.copyTo(img);
            doWatersheds();	show();
			break;
		case 'c':
			markerMask_undo = markerMask.clone();
			markerMask = 0;
			doWatersheds(); show();
			break;
		case 'g'://3x3 gaussian blur
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::GaussianBlur(img, img, cv::Size(3,3), 0);
			doWatersheds();	show();
			break;
		case 'G'://5x5 gaussian blur
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::GaussianBlur(img, img, cv::Size(5,5), 0);
			doWatersheds();	show();
			break;
		case 'b':
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			tmp = img.clone();
			cv::bilateralFilter(tmp, img, 3, 40, 40);
			doWatersheds(); show();
			break;
		case 'B':
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			tmp = img.clone();
			cv::bilateralFilter(tmp, img, 3, 120, 120);
			doWatersheds(); show();
			break;
		case 'l':
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			c = cv::waitKey(0);
			if(c <= '9' && c >= '1')
				clahe = cv::createCLAHE(50, cv::Size(2*(c-'1'+1), 2*(c-'1'+1)));
			if(img.channels() == 3)
				cv::cvtColor(img, img, CV_BGR2GRAY);
			clahe->apply(img, img);
			doWatersheds(); show();
			break;
		case 'm': //3x3 median blur
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::medianBlur(img, img, 3);
			doWatersheds();	show();
			break;
		case 'M': //5x5 median blur
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::medianBlur(img, img, 5);
			doWatersheds();	show();
			break;
		case 'I': //store as new base image
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			std::cout << "Set current as Base Image" << std::endl;
			if(img.channels() == 1)
				cv::cvtColor(img, img0, CV_GRAY2BGR);
			else 
				img0 = img.clone();
			break;
		case 'e': //erode
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::erode(img, img, element, cv::Point(-1, -1), 3);
			doWatersheds();	show();
			break;
		case 's': //"standard" pipeline for automatic seeds
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			if(img.channels() > 1) cv::extractChannel(img, img, 0);
			cv::medianBlur(img, tmp, 3);
			cv::GaussianBlur(tmp, tmp, cv::Size(3,3), 0);
			cv::threshold(tmp, tmp, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

			cv::findContours(tmp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			cv::drawContours(tmp, contours, -1, cv::Scalar(255, 255, 255), CV_FILLED);
			
			cv::distanceTransform(tmp, tmp, CV_DIST_L2 , CV_DIST_MASK_PRECISE);
			tmp = findLocalMaxima(tmp, 8);
			markerMask = tmp;
			doWatersheds();	show();
			break;
		case 't': //thresholding
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			if(img.channels() == 1) {
				cv::threshold(img, img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
				doWatersheds();	show();
			}
			else
				std::cout << "Thresholding only on grayscale images!" << std::endl;
			break;
		case '1':
		case '2':
		case '3': //Extract one channel
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			if(img.channels() > 1) {
				cv::split(img, channels);
				extractChannel(img, img, (char)c - '1');
				doWatersheds();	show();
			}			
			break;

		case 'h': //fill holes in binary image
			img_undo = img.clone(); markerMask_undo = markerMask.clone();
			cv::findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			cv::drawContours(img, contours, -1, cv::Scalar(255, 255, 255), CV_FILLED);
			doWatersheds();	show();
			break;
		case 'u':
			img = img_undo.clone();
			markerMask = markerMask_undo.clone();
			doWatersheds(); show();
		}

        if( (char)c == 'w' || (char)c == ' ' )
        {
			doWatersheds();
        }
    }

    return 0;
}
