#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>

//#include "../common/HelperFunctions.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <fstream>
#include<direct.h>
#if defined(_MSC_VER) && _MSC_VER >= 1400 
#pragma warning(push) 
#pragma warning(disable:4996) 
#endif 


using namespace cv;
using namespace std;

void show();
void paintAndDisplayResult();
void doWatersheds();
void shownumbers(int index);
void showlabels();
void displayIntensity(string name,int i);
//void saveRef(int x, int y);
void saveindexes(int index);
void changeindex(int oldindex);

struct segmentation{
	Mat markerMask, img, imgt, markers, indexes, markers2, wshed, imgGray, img0, img_orig, wresultBoundaries, red, green, gray, numbers;
	vector<vector<Point> > contours;
	string resultsfolder;
	string sequencefolder;
	int compCount;
	int count1, count2, count3, count4;
	vector<Vec3b> colorTab;
	cv::Mat currentDisplay;
	string slice;
	int xref;
	int yref;
	bool change;
	Mat indexes2;
	bool changeindex;
	
};

static void help()
{
  /*  cout << "\nModified Watersheds Segmentation with better UI\n"
            "Usage:\n"
            "./watershed [image_name]\n" << endl;*/

	cout<<"================================== Usage =================================="<<endl<<endl;
	cout<<endl;
	cout << "Mouse Bindings:\n"
		"Left Mouse: \tDraw foreground seeds (segmentation mode)\n"
		" \t\tselect regions to label (labelling mode)\n"
		" \t\tselect the reference point (reference-selection mode)\n"
		"Left Mouse + CTRL: \tDraw background seeds (segmentation mode)\n"
		"Right Mouse: \t\tErase a part of a seed (segmentation mode)\n"
		"Right Mouse + CTRL: \tRemove a connected seed completely (segmentation mode)\n"
		"\n";

    cout << "Hot keys (when the interactive window is active): \n"
        "\th   - Show the help information\n" 
		"\tESC - Quit the program\n"
		/*"\tu   - undo last step\n"
        /*"\tc   - clear seed mask\n"
		"\tC   - restore original image while keeping the seeds\n"
        "\tw - run watershed segmentation algorithm\n"
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
		"\ts   - find automatic seeds\n"*/
		
		"\tg   - Indexing one cell inside the volume\n"
		"\tl   - Enter/exit labeling mode\n"
		"\tr   - Enter/exit the reference-selection mode\n"
		"\td   - Show the number of cells in every class\n"
		"\ts   - Show/hide cell labels\n"
		"\tn   - Show/hide cell indices\n"
		"\tG   - Show/Hide index of the cell inside the volume\n"
		"\t1,2,3,4	   - Inside labeling mode, label the selected region\n"
		//"\t		left click and 1,2,3,4   - assign label\n"
		"\tSPACE   - Finish segmenting the current image and save result\n"<<endl;

	cout<<endl<<"Label colours: Class 1 - Blue; Class 2 - Green; Class 3 - Red; Class 4 - Magenta"<<endl;

	cout<<endl<<"To quit the program when asked for the image name, type 'quit' and hit 'Enter'."<<endl<<endl;

	cout<<"==========================================================================="<<endl;
	cout<<endl<<endl;
}
Mat markerMask, img,imgt, markers,indexes, markers2, wshed, imgGray, img0, img_orig, wresultBoundaries, red, green, gray,numbers;
Mat img_undo, markerMask_undo;
vector<vector<Point> > contours;
string resultsfolder;
string sequencefolder;
int compCount = 0;
int count1, count2, count3, count4;
bool change = false;
Point prevPt(-1, -1);
vector<Vec3b> colorTab;
std::string recordingPattern = "";
bool recordingEnabled = false;
int recordingImageID = 1;
cv::Mat currentDisplay;
bool rfirst, xfirst, zfirst, pfirst,ifirst,gfirst;	//zfirst is the labelling mode, false being labelling mode on; rfirst is for reference;ifirst for the index
string slice;
int xref=-1;
int yref=-1;
vector<segmentation> results;
int currentidx;
int ilabel;

int labellingMarkerAtDown = -1;
unsigned char indexAtDown = -1;
bool bSliceFinish = 0;

int CreateResultDir(std::string& ResultDir)
{
	ResultDir = "..\\Results\\";
	_mkdir(ResultDir.c_str());
	char timeStr[9], dateStr[9];
	_strtime(timeStr);	_strdate(dateStr);
	char time[7] = {timeStr[0], timeStr[1], timeStr[3], timeStr[4],
					timeStr[6], timeStr[7], 0};
	char date[7] = {dateStr[0], dateStr[1], dateStr[3], dateStr[4],
					dateStr[6], dateStr[7], 0};	
	ResultDir += time;
	ResultDir += "_";
	ResultDir += date;
	ResultDir += "\\";
	_mkdir(ResultDir.c_str());

	return 1;
}

void recordFrame(std::string winName)
{
	if(!recordingEnabled)
		return;

	char filename[1024];
	sprintf_s(filename, recordingPattern.c_str(), winName.c_str(), recordingImageID);
	recordingImageID++;
	cv::imwrite(filename, currentDisplay);
}

void saveResult(int i){
	
		ofstream outputNum((results[i].resultsfolder + "/Num of Cells in every class.csv").c_str());
		displayIntensity(results[i].resultsfolder,i);
		imwrite(results[i].resultsfolder + "/Segmentation_Result.jpg", results[i].wshed);
		shownumbers(i);
		imwrite(results[i].resultsfolder + "/Cell indexes.jpg", results[i].wresultBoundaries + results[i].img0 + results[i].numbers);
		imwrite(results[i].resultsfolder + "/Segmentation Blue.jpg", results[i].img0 + results[i].wresultBoundaries);
		imwrite(results[i].resultsfolder + "/Segmentation Red.jpg", results[i].red + results[i].wresultBoundaries);
		imwrite(results[i].resultsfolder + "/Segmentation Green.jpg", results[i].green + results[i].wresultBoundaries);
		imwrite(results[i].resultsfolder + "/Segmentation Gray.jpg", results[i].gray + results[i].wresultBoundaries);
		imwrite(results[i].resultsfolder + "/Interaction window.jpg", results[i].currentDisplay);
		saveindexes(i);
		imwrite(results[i].resultsfolder + "/Volume Segmentation indices.png", results[i].indexes2);
		outputNum << results[i].count1 << "," << results[i].count2 << "," << results[i].count3 << "," << results[i].count4 << endl;
		outputNum.close();
	
}

void saveRef(int x, int y, const int sliceIdx){
	ofstream outputRef((results[currentidx].resultsfolder+"/reference point this slice.csv").c_str());// CHANGE RESULTS FOLDERS
	ofstream outputlastRef((results[currentidx].sequencefolder + "/Final reference point whole volume.csv").c_str());
	outputRef << x << "," << y << endl;
	outputlastRef << x << "," << y << ","<<currentidx<<endl;
	results[currentidx].xref = x;
	results[currentidx].yref = y;
	show();
	outputRef.close();
	outputlastRef.close();
}

void show()
{
	static cv::Mat mask3c;
	if (results[currentidx].img.channels() == 3) {
		cv::cvtColor(results[currentidx].markerMask, mask3c, COLOR_GRAY2BGR);
		results[currentidx].currentDisplay = results[currentidx].img + 0.5*mask3c + 0.5*results[currentidx].wresultBoundaries;
		if (xref > 0 && yref > 0){
			putText(results[currentidx].currentDisplay, "x", cvPoint(xref - 15, yref + 12),
				FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(255, 255, 255), 1, CV_AA);
		}
		imshow("Blue (Interaction Window)", results[currentidx].currentDisplay);
		
	}
	else {
		assert(false);
	}
	

	
	
}

void showcopy(int x, int y){

	Mat positionr = results[currentidx].red.clone();
	cv::circle(positionr, Point(x,y), 1, cvScalar(255,255,255,255), 5);
	namedWindow("red", CV_WINDOW_KEEPRATIO);
	imshow("red", positionr);
	cv::resizeWindow("red",280,300);
	moveWindow("red",640, 0);
	Mat positionge = results[currentidx].green.clone();
	cv::circle(positionge, Point(x, y), 1, cvScalar(255, 255, 255, 255), 5);
	namedWindow("green", CV_WINDOW_KEEPRATIO);
	imshow("green", positionge);
	cv::resizeWindow("green", 280, 300);
	moveWindow("green",640,340);
	Mat positionga = results[currentidx].gray.clone();
	cv::circle(positionga, Point(x, y), 1, cvScalar(255, 255, 255, 255), 5);
	namedWindow("gray", CV_WINDOW_KEEPRATIO);
	imshow("gray", positionga);
	cv::resizeWindow("gray", 280, 300);
	moveWindow("gray", 940,0);
}

static void onMouse( int event, int x, int y, int flags, void* )
{
	static bool drawingBackground = false;

	if (event == EVENT_MOUSEMOVE){
		showcopy(x, y);
	}
	if (x < 0 || x >= results[currentidx].img.cols || y < 0 || y >= results[currentidx].img.rows)
        return;

	if(event == EVENT_LBUTTONUP || event == EVENT_RBUTTONUP ) {
		doWatersheds();
	}
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) ) {
        prevPt = Point(-1,-1);
	}
    else if( event == EVENT_LBUTTONDOWN ) {
		img_undo = results[currentidx].img.clone(); markerMask_undo = results[currentidx].markerMask.clone();
		drawingBackground = (flags & EVENT_FLAG_CTRLKEY) > 0;
        prevPt = Point(x,y);
		cv::circle(results[currentidx].markerMask, prevPt, 1, cv::Scalar::all(drawingBackground ? 128 : 255), 5);
		show();
	}
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
		line(results[currentidx].markerMask, prevPt, pt, Scalar::all(drawingBackground ? 128 : 255), 5, 8, 0);
        prevPt = pt;
        show();
    }
	
	if(event == EVENT_RBUTTONDOWN) {
		std::cout << "Erasing seeds.." << std::endl;
		if (flags & EVENT_FLAG_CTRLKEY) {
			cv::floodFill(results[currentidx].markerMask, Point(x, y), 0);
			show();
		}
		else {
			cv::circle(results[currentidx].markerMask, prevPt, 1, cv::Scalar::all(0), 12);
			show();
		}
	}
	else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON)) {
		Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
		line(results[currentidx].markerMask, prevPt, pt, Scalar::all(0), 12, 8, 0);
        prevPt = pt;
        show();
	}
}

inline int safeAt(int i, int j) {
	if (i < 0 || i >= results[currentidx].img.rows || j < 0 || j >= results[currentidx].img.cols)		return -1;
	return results[currentidx].markers.at<int>(i, j);
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
	for (int i = 0; i < results[currentidx].markers.rows; i++)
	for (int j = 0; j < results[currentidx].markers.cols; j++) {
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
	cv::Mat boundaries(results[currentidx].markers.size(), CV_8UC3);
	for (int i = 0; i < results[currentidx].markers.rows; i++)
	for (int j = 0; j < results[currentidx].markers.cols; j++) {
			//check neighbors if they are either boundary or same class, if yes, we can assign this pixel to the class
		int nClass = results[currentidx].markers.at<int>(i, j);

			if(!checkNeighbor(i+1,j,nClass) || !checkNeighbor(i-1,j,nClass) || !checkNeighbor(i,j+1,nClass) || !checkNeighbor(i,j-1,nClass) || 
				!checkNeighbor(i+1,j+1,nClass) || !checkNeighbor(i+1,j-1,nClass) || !checkNeighbor(i-1,j+1,nClass) || !checkNeighbor(i-1,j-1,nClass)) 
			{
				boundaries.at<cv::Point3_<unsigned char>>(i,j) = cv::Point3_<unsigned char>(255,255,255);
			}
			else{
				boundaries.at<cv::Point3_<unsigned char>>(i, j) = cv::Point3_<unsigned char>(0, 0, 0);
			}
		}
	return boundaries;
}


void displayMarker(int cmarker)
{
	std::cout << cmarker << std::endl;
	int i, j ;
	cv::Mat cell;
	string Result;          // string which will contain the result
	ostringstream convert;   // stream used for the conversion

	convert << cmarker;      // insert the textual representation of 'Number' in the characters in the stream
	Result = convert.str();
	cell = img0.clone();
	for (i = 1; i < markers.cols; i++){
		for (j = 1; j < markers.rows; j++){
			if (markers2.at<int>(i, j) != cmarker){
				cell.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				
			}

		}
	}
	putText(cell, "data will appear here", cvPoint(10, 10),
		FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
	
	imwrite("Cell"+Result+".png", cell);


}

cv::Mat filterContours(Mat cont){// not used
	int i;
	
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Mat nonZeroCoordinates;
	cv::Mat cont2(markers.size(), CV_8UC3);

	cvtColor(cont, cont, COLOR_BGR2GRAY);
	minMaxLoc(cont, &minVal, &maxVal, &minLoc, &maxLoc);
	threshold(cont, cont, minVal, maxVal, THRESH_BINARY_INV);
	findNonZero(cont, nonZeroCoordinates);
	Mat x(nonZeroCoordinates.total(), 1, CV_32S);
	Mat y(nonZeroCoordinates.total(), 1, CV_32S);
	for (i = 0; i < (int)nonZeroCoordinates.total(); i++) {
		x.at<int>(i, 0) = nonZeroCoordinates.at<Point>(i).x;
		y.at<int>(i, 0) = nonZeroCoordinates.at<Point>(i).y;
	}
	
	GaussianBlur(x, x, cv::Size(5, 1),100.0);
	GaussianBlur(y, y, cv::Size(5, 1), 100.0);
	for (i = 0; i < (int)nonZeroCoordinates.total(); i++) {
		nonZeroCoordinates.at<Point>(i).x = x.at<int>(i, 0) ;
		nonZeroCoordinates.at<Point>(i).y = y.at<int>(i, 0);
	}
	
	for (i = 0; i < (int)nonZeroCoordinates.total(); i++){
		cont2.at<cv::Point3_<unsigned char>>(nonZeroCoordinates.at<Point>(i)) = cv::Point3_<unsigned char>(255, 255, 0);;
	}
	cout << "x:" << countNonZero(x) << endl;
	cout << "y:" << countNonZero(y) << endl;
	cvtColor(cont2, cont2, COLOR_BGR2GRAY);
	minMaxLoc(cont2, &minVal, &maxVal, &minLoc, &maxLoc);
	threshold(cont2, cont2, minVal, maxVal, THRESH_BINARY_INV);
	cout << "cont2:" << countNonZero(cont2) << endl;
	return(cont2);
	

}


void changeMarker(int cmarker,int m){
	int i, j;

	for (i = 0; i < results[currentidx].markers.cols; i++)
	{
		for (j = 0; j < results[currentidx].markers.rows; j++)
		{
			if (results[currentidx].markers.at<int>(i, j) == cmarker){
				results[currentidx].markers2.at<int>(i, j) = m;
				

			}

		}
	}
	results[currentidx].change = true;
	paintAndDisplayResult();
}

static void indexMouse(int event, int x, int y, int flags, void*){
	static cv::Point downPt(-1, -1);
	//static int markerAtDown = -1;
	//static int markerAtDown2 = -1;
	static bool drawingBackground = false;
	//int c = waitKey(30);


	if (x < 0 || x >= results[currentidx].indexes.cols || y < 0 || y >= results[currentidx].indexes.rows)
		return;


	if (event == EVENT_MOUSEMOVE){
		showcopy(x, y);
	}

	if (event == EVENT_LBUTTONDOWN) {
		downPt = Point(x, y);
		// check to make sure it's not a background		
		/*signed mm = results[currentidx].indexes.at<signed>(y, x);
		if (mm <= results[currentidx].compCount / 2 || mm > results[currentidx].compCount)
		{
			std::cout << "Cannot index the background. Please select a cell to index. " << std::endl;
			return;
		}*/
		//
		indexAtDown = results[currentidx].indexes.at<unsigned char>(y, x);
		changeindex(indexAtDown);
		//changeindex(indexAtDown, ilabel);
		////unsigned char indexAtDown2 = results[currentidx].indexes2.at<unsigned char>(y, x);
		//int c = indexAtDown2;
		//cout << c << endl;
		//markerAtDown = markers.at<int>(y, x);
		//markerAtDown2 = markers2.at<int>(y, x);
		/*int c = waitKey();
		switch (c){
		case '4':
		changeMarker(markerAtDown, 4);
		break;
		case '1':
		changeMarker(markerAtDown, 1);
		break;
		case '2':
		changeMarker(markerAtDown, 2);
		break;
		case '3':
		changeMarker(markerAtDown, 3);
		break;

		}*/


	}
	else {
		downPt = Point(-1, -1);
	}
}

static void labelMouse(int event, int x, int y, int flags, void*){
	static cv::Point downPt(-1, -1);
	//static int markerAtDown = -1;
	//static int markerAtDown2 = -1;
	static bool drawingBackground = false;
	//int c = waitKey(30);


	if (x < 0 || x >= results[currentidx].markers.cols || y < 0 || y >= results[currentidx].markers.rows)
		return;

	
	if (event == EVENT_MOUSEMOVE){
		showcopy(x, y);
	}

	if (event == EVENT_LBUTTONDOWN) {
		downPt = Point(x, y);
		// check to make sure it's not a background		
		signed mm = results[currentidx].markers.at<signed>(y, x);
		if (mm <= results[currentidx].compCount / 2 || mm > results[currentidx].compCount)
		{
			std::cout<<"Cannot label the background. Please select a cell to label. "<<std::endl;
			return;
		}
		//
		labellingMarkerAtDown = results[currentidx].markers.at<int>(y, x);
		//markerAtDown = markers.at<int>(y, x);
		//markerAtDown2 = markers2.at<int>(y, x);
		/*int c = waitKey();
		switch (c){
		case '4':
			changeMarker(markerAtDown, 4);
			break;
		case '1':
			changeMarker(markerAtDown, 1);
			break;
		case '2':
			changeMarker(markerAtDown, 2);
			break;
		case '3':
			changeMarker(markerAtDown, 3);
			break;

		}*/


	}
	else {
		downPt = Point(-1, -1);
	}
}

static void referenceMouse(int event, int x, int y, int flags, void* params){
	static cv::Point downPt(-1, -1);
	if (event == EVENT_MOUSEMOVE){
		showcopy(x, y);
	}
	if (x < 0 || x >= results[currentidx].img0.cols || y < 0 || y >= results[currentidx].img0.rows)
	{
		return;
	}
	if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) {
		downPt = Point(-1, -1);
	}
	else if (event == EVENT_LBUTTONDOWN) {
		int* pSliceIdx = (int*)params;
		saveRef(x, y, *pSliceIdx);														// has to be changed
		cout << "Reference: (" << x << "," << y << ") has been set" << endl;

	}

}

static void onMouseComponentMap( int event, int x, int y, int flags, void* )//not used
{
	static cv::Point downPt(-1, -1);
	static int markerAtDown = -1;
	static int markerAtDown2 = -1;
	static bool drawingBackground = false;
	//int c = waitKey(30);
	

    if( x < 0 || x >= markers.cols || y < 0 || y >= markers.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) ) {
        downPt = Point(-1,-1);
	}
	else if (event == EVENT_LBUTTONDOWN) {
		
	}

		


		//cout << "Marker is " << markerAtDown2 << std::endl;
	

	else if(event == EVENT_RBUTTONDOWN || event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON) ) {
		if(markers.at<int>(y,x) >= 0 && markers.at<int>(y,x) <= compCount) {
			floodFill(markers, cv::Point(x, y), -2);
			paintAndDisplayResult();
			std::cout << "Cleared at " << y << " " << x << " to -2" << std::endl;
		}
	}
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) ) {
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
	cv::Mat filteredCont(results[currentidx].markers.size(), CV_8UC3);
	int i, j;
	//wresultBoundaries = findBoundaries();
	
	for (i = 0; i < results[currentidx].markers.rows; i++)
	for (j = 0; j < results[currentidx].markers.cols; j++)
        {
		int index = results[currentidx].markers.at<int>(i, j);
		int index2 = results[currentidx].markers2.at<int>(i, j);
            if( index == -1 )
				results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
            //else if( index <= 0 || index > compCount )
               // wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
			else if (results[currentidx].change && index2>0){
				switch (index2){
				case 1:
					results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(255, 0, 0);
						break;
				case 2:
					results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
					break;
				case 3:
					results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
					break;
				case 4:
					results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(255, 0, 255);
					break;
						
				}
				
			}
			else 
				results[currentidx].wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                
        }
	
	

	results[currentidx].wshed = results[currentidx].wshed + 3 * results[currentidx].imgGray + results[currentidx].wresultBoundaries;
	/*cv::namedWindow("Segmentation Result", CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("Segmentation Result", 280, 300);
	moveWindow("Segmentation Result",940, 340);*/
	imshow("Segmentation Result", results[currentidx].wshed);
	//cv::setMouseCallback("Segmentation Result", onMouseComponentMap);
	//cout << "assign next label" << endl;
	//recordFrame("Segmentation Result");

}



int makeComponentMap()
{
	
    vector<Vec4i> hierarchy;

	findContours(cv::Mat(results[currentidx].markerMask - 128), results[currentidx].contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	int compCount = results[currentidx].contours.size();


	if (results[currentidx].contours.empty()) {
		//std::cout << "No contours found in mask. Is the mask empty?" << std::endl;
        return 0;
	}
	results[currentidx].markers = cv::Mat(results[currentidx].markerMask.size(), CV_32S);
	if (!results[currentidx].change)
		results[currentidx].markers2 = cv::Mat(results[currentidx].markerMask.size(), CV_32S);
	cv::Mat(results[currentidx].markerMask == 128).convertTo(results[currentidx].markers, CV_32S, compCount / 256.);
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
		drawContours(results[currentidx].markers, results[currentidx].contours, idx, Scalar::all(compCount + 1), CV_FILLED, 8, hierarchy, INT_MAX);
	return compCount;
}


void displayIntensity(string name, int index){
	Mat channelblue[3];
	Mat channelred[3];
	Mat channelgreen[3];
	Mat imb,imr,imge,imga;
	ofstream outputCells((name+"/Cell wise information.csv").c_str());
	ofstream outputLabels((name+"/Class wise information.csv").c_str());
	Mat cont_avgsBlue=cv::Mat::zeros(1,results[index].contours.size(),CV_32F);
	Mat cont_avgsRed = cv::Mat::zeros(1, results[index].contours.size(), CV_32F);
	Mat cont_avgsGreen = cv::Mat::zeros(1, results[index].contours.size(), CV_32F);
	Mat cont_avgsGray = cv::Mat::zeros(1, results[index].contours.size(), CV_32F);
	split(results[index].img0, channelblue);
	imb = channelblue[0];
	split(results[index].red, channelred);
	imr = channelred[2];
	split(results[index].green, channelgreen);
	imge = channelgreen[1];
	cvtColor(results[index].gray, imga, COLOR_BGR2GRAY);
	int nbins = 256; 
	int hsize[] = { nbins }; 
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	int channels[] = { 0 };
	results[index].count1 = 0;
	results[index].count2 = 0;
	results[index].count3 = 0;
	results[index].count4 = 0;
	bool first = true;
	MatND histb, histr, histge, histga, hist1b, hist1r, hist1ge,hist1ga, hist2b, hist2r, hist2ge, hist2ga, hist3b, hist3r, hist3ge, hist3ga, hist4b, hist4r, hist4ge, hist4ga;

	for (size_t i = 0; i < (int)results[index].contours.size(); ++i)
	{
		cv::Rect roi = cv::boundingRect(results[index].contours[i]);
		int markerxy = results[index].markers2.at<int>((int)mean(results[index].contours[i]).val[1], (int)mean(results[index].contours[i]).val[0]);
		//blue
		calcHist(&imb(roi), 1, channels, Mat(), histb, 1, hsize, ranges);
		histb = histb / sum(histb)[0];
		//red
		calcHist(&imr(roi), 1, channels, Mat(), histr, 1, hsize, ranges);
		histr = histr / sum(histr)[0];
		//green
		calcHist(&imge(roi), 1, channels, Mat(), histge, 1, hsize, ranges);
		histge = histge / sum(histge)[0];
		//gray
		calcHist(&imga(roi), 1, channels, Mat(), histga, 1, hsize, ranges);
		histga = histga / sum(histga)[0];
		if (first){//check how to initialize a MatND
			first = false;
			hist1b = histb - histb;
			hist1r = histr - histr;
			hist1ge = histge - histge;
			hist1ga = histga - histga;
			hist2b = histb - histb;
			hist2r = histr - histr;
			hist2ge = histge - histge;
			hist2ga = histga - histga;
			hist3b = histb - histb;
			hist3r = histr - histr;
			hist3ge = histge - histge;
			hist3ga = histga - histga;
			hist4b = histb-histb;
			hist4r = histr-histr;
			hist4ge = histge-histge;
			hist4ga = histga-histga;
		}
		switch (markerxy){
		case 1:
			hist1b += histb;
			hist1r += histr;
			hist1ge += histge;
			hist1ga += histga;
			count1++;
			break;
		case 2:
			hist2b += histb;
			hist2r += histr;
			hist2ge += histge;
			hist2ga += histga;
			count2++;
			break;
		case 3:
			hist3b += histb;
			hist3r += histr;
			hist3ge += histge;
			hist3ga += histga;
			count3++;
			break;
		case 4:
			hist4b += histb;
			hist4r += histr;
			hist4ge += histge;
			hist4ga += histga;
			count4++;
			break;
		}
		outputCells<< "Cell " << i << endl;
		outputCells << "Blue channel" << endl;
		for (int j = 0; j < nbins; j++){
			outputCells << histb.at<float>(j, 0) << ",";
		}
		outputCells << endl;
		outputCells << "Red channel" << endl;
		for (int j = 0; j < nbins; j++){
			outputCells << histr.at<float>(j, 0) << ",";
		}
		outputCells << endl;
		outputCells << "Green channel" <<  endl;
		for (int j = 0; j < nbins; j++){
			outputCells << histge.at<float>(j, 0) << ",";
		}
		outputCells << endl;
		outputCells << "Gray channel" << endl;
		for (int j = 0; j < nbins; j++){
			outputCells << histga.at<float>(j, 0) << ",";
		}
		outputCells << endl;
	}
	
	outputLabels << "Label 1" << endl;
	outputLabels << "blue" <<  endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist1b.at<float>(j, 0)/count1 << ",";
	}
	outputLabels<< endl;
	outputLabels << "red" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist1r.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "green" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist1ge.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "gray" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist1ga.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "Label 2" << endl;
	outputLabels << "blue" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist2b.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "red" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist2r.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "green" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist2ge.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "gray" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist2ga.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "Label 3" << endl;
	outputLabels << "blue" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist3b.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "red" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist3r.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "green" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist3ge.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "gray" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist3ga.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "Label 4" << endl;
	outputLabels << "blue" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist4b.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "red" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist4r.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "green" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist4ge.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;
	outputLabels << "gray" << endl;
	for (int j = 0; j < nbins; j++){
		outputLabels << hist4ga.at<float>(j, 0) / count1 << ",";
	}
	outputLabels << endl;

	outputCells.close();
	outputLabels.close();
	/*ofstream outputLabels("results/Histograms Labels.csv");
	
	for (int l = 1; l <5 ; l++){
		outputLabels<< "Label" << l << endl;
		int j = 0;
		for (size_t i = 0; i < (int)contours.size(); ++i)
		{
			cv::Rect roi = cv::boundingRect(contours[i]);
			cv::Scalar meanb = cv::mean(imb(roi));
			cv::Scalar meanr = cv::mean(imr(roi));
			cv::Scalar meange = cv::mean(imge(roi));
			cv::Scalar meanga = cv::mean(imga(roi));
			int markermask = markers2.at<int>((int)mean(contours[i]).val[1], (int)mean(contours[i]).val[0]);
			if (markermask == l){
				cont_avgsBlue.at<float>(0, j) = (float)meanb[0];
				cont_avgsRed.at<float>(0, j) = (float)meanr[0];
				cont_avgsGreen.at<float>(0, j) = (float)meange[0];
				cont_avgsGray.at<float>(0, j) = (float)meanga[0];
				j++;
			}
		}
			Mat cont_avgsBlue2 = cv::Mat::zeros(1, countNonZero(cont_avgsBlue), CV_32F);
			Mat cont_avgsRed2 = cv::Mat::zeros(1, countNonZero(cont_avgsRed), CV_32F);
			Mat cont_avgsGreen2 = cv::Mat::zeros(1, countNonZero(cont_avgsGreen), CV_32F);
			Mat cont_avgsGray2 = cv::Mat::zeros(1, countNonZero(cont_avgsGray), CV_32F);
			for (int i = 0; i < countNonZero(cont_avgsBlue); i++){
				cont_avgsBlue2.at<float>(0, i) = cont_avgsBlue.at<float>(0, i);
				cont_avgsRed2.at<float>(0, i) = cont_avgsRed.at<float>(0, i);
				cont_avgsGreen2.at<float>(0, i) = cont_avgsGreen.at<float>(0, i);
				cont_avgsGray2.at<float>(0, i) = cont_avgsGray.at<float>(0, i);
			}
			calcHist(&cont_avgsBlue2, 1, channels, Mat(), histb, 1, hsize, ranges);
			calcHist(&cont_avgsRed2, 1, channels, Mat(), histr, 1, hsize, ranges);
			calcHist(&cont_avgsGreen2, 1, channels, Mat(), histge, 1, hsize, ranges);
			calcHist(&cont_avgsGray2, 1, channels, Mat(), histga, 1, hsize, ranges);
			outputLabels << "blue\n" << histb << endl;
			outputLabels << "red\n" << histr << endl;
			outputLabels << "green\n" << histge << endl;
			outputLabels << "gray\n" << histga << endl;


		}
	
	outputLabels.close();*/

}

void shownumbers(int index){
	int i;
	   // stream used for the conversion
	results[index].numbers = cv::Mat::zeros(results[index].img0.size(), results[index].img0.type());
	
	for (i = 0; i < (int)results[index].contours.size(); i++)
	{
		string Result;          // string which will contain the result
		ostringstream convert;
		convert << i;      
		Result = convert.str();
		putText(results[index].numbers, Result, cvPoint((int)mean(results[index].contours[i]).val[0], (int)mean(results[index].contours[i]).val[1]),
			FONT_HERSHEY_COMPLEX_SMALL, 1.8, cvScalar(200, 200, 250), 1, CV_AA);
	}
	imshow("Blue (Interaction Window)", results[index].currentDisplay + results[index].numbers);
}

void saveindexes(int index){
	results[index].indexes = cv::Mat(results[index].markers.size(), CV_8UC1, cvScalar(255));
	int m;

	for (int i = 0; i < results[index].markers.rows; i++){
		for (int j = 0; j < results[index].markers.cols; j++){
			m = results[index].markers.at<int>(i, j);
			if (m>results[index].compCount / 2 && m <= results[index].compCount){
				results[index].indexes.at<unsigned char>(i, j) = m - results[index].compCount / 2 - 1;
			}
			else
				results[index].indexes.at<unsigned char>(i, j) = 255;
			
		}
	}
	if (!results[index].changeindex)
		results[index].indexes2 = cv::Mat(results[index].indexes.size(), CV_8UC1, cvScalar(255));





	imwrite(results[index].resultsfolder + "/Segmentation indices.png", results[index].indexes);
}



void changeindex(int indexold){
	int i, j;
	string strlabel;
	results[currentidx].changeindex = true;
	saveindexes(currentidx);
	cout << "Segmentation mode off. Index mode on." << endl;
	cout << "Please introduce index" << endl;
	getline(cin, strlabel);
	ilabel = stoi(strlabel);
	if (strlabel == "stop")
		return;
	
	
	for (i = 0; i < results[currentidx].indexes.cols; i++)
		{
			for (j = 0; j < results[currentidx].indexes.rows; j++)
			{
				if (results[currentidx].indexes.at<unsigned char>(i, j) == indexold){
					results[currentidx].indexes2.at<unsigned char>(i, j) = ilabel;
				}

			}
		}
	setMouseCallback("Blue (Interaction Window)", onMouse, 0);
		
		
	}


void showlabels(){
	int i;
	int markerAtDown;
	Mat labels = cv::Mat::zeros(results[currentidx].img0.size(), results[currentidx].img0.type());

	for (i = 0; i < (int)results[currentidx].contours.size(); i++)
	{
		markerAtDown = results[currentidx].markers2.at<int>((int)mean(results[currentidx].contours[i]).val[1], (int)mean(results[currentidx].contours[i]).val[0]);
		string Result;          
		ostringstream convert;
		convert << markerAtDown;
		Result = convert.str();
		putText(labels, Result, cvPoint((int)mean(results[currentidx].contours[i]).val[0] + 20, (int)mean(results[currentidx].contours[i]).val[1] + 20),
			FONT_HERSHEY_COMPLEX_SMALL, 1.8, cvScalar(126, 20, 250), 1, CV_AA);
	}
	imshow("Blue (Interaction Window)", results[currentidx].currentDisplay + labels);
}

void showindexes2(){
	int i;
	unsigned char charatdown;
	Mat showind = cv::Mat::zeros(results[currentidx].img0.size(), results[currentidx].img0.type());

	for (i = 0; i < (int)results[currentidx].contours.size(); i++)
	{
		charatdown = results[currentidx].indexes2.at<unsigned char>((int)mean(results[currentidx].contours[i]).val[1], (int)mean(results[currentidx].contours[i]).val[0]);
		string Result;
		int indexAtDown = charatdown;
		ostringstream convert;
		convert << indexAtDown;
		Result = convert.str();
		if (indexAtDown!=255)
		putText(showind, Result, cvPoint((int)mean(results[currentidx].contours[i]).val[0] + 20, (int)mean(results[currentidx].contours[i]).val[1] + 20),
			FONT_HERSHEY_COMPLEX_SMALL, 1.8, cvScalar(126, 20, 250), 1, CV_AA);
	}
	imshow("Blue (Interaction Window)", results[currentidx].currentDisplay + showind);
}


void doWatersheds()
{
	int i;
	results[currentidx].compCount = 0;

	
	//we need a 3ch-image for watersheds
	if (results[currentidx].img.channels() == 1)
		cv::cvtColor(results[currentidx].img, results[currentidx].img, COLOR_GRAY2BGR);

	results[currentidx].compCount = makeComponentMap();

	if (results[currentidx].compCount == 0)
        return;

	for (i = results[currentidx].colorTab.size(); i < results[currentidx].compCount; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

		results[currentidx].colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    double t = (double)getTickCount();
	watershed(results[currentidx].img, results[currentidx].markers);
	results[currentidx].wresultBoundaries = findBoundaries();
	show();
    t = (double)getTickCount() - t;
    //printf( "execution time = %gms\n", t*1000./getTickFrequency() );

	//cout << "Number of segmentationts =" << compCount/2 << endl;

	results[currentidx].wshed = cv::Mat(results[currentidx].markers.size(), CV_8UC3);
	
	paintAndDisplayResult();
}

int ParseSliceIdx(const string& str)
{
	int idx = -1;
	idx = atoi(str.c_str());

	return idx;
}

void open(string name, const string& resultDir, int& sliceIdx)
{

	Mat testImg = imread(name);
	if (testImg.empty())
		return;
	int length = name.length() - 8;
	/*char fold[9] = " ";
	char* subfold = new char;*/
	/*char* names = new char[length];
	name.copy(names, name.length() - 8, 0);*/
	string names(name.substr(0,length));
	string name2(names);
	string test(name.substr(0, length - 4));
	string test2(test);
	std::stringstream inx;
	int indexsl=4;
	inx << setfill('0') <<setw(3)<< indexsl;
	string slicei = inx.str();

	xref = -1;
	yref = -1;
	zfirst = true;
	xfirst = true;
	pfirst = true;
	rfirst = true;
	markers2 = cv::Mat(markerMask.size(), CV_32S);
	wresultBoundaries = cv::Mat(markers.size(), CV_8UC3);


	// find the slice index
	string sliceIdxStr(name2.substr(length-4, length-1));
	sliceIdx = ParseSliceIdx(sliceIdxStr);

	//
	unsigned lastSlashFound = name2.find_last_of("/\\");
	string fold(name2.substr(lastSlashFound+1, length-lastSlashFound-2));

	string subfold(name2.substr(lastSlashFound-1,1));
	//name2.copy(fold, 8, name2.length() - 9);
	//name2.copy(subfold, 1, name2.length() - 15);
	img0 = imread(name2 + "c001.jpg", 1);
	if (img0.empty()){
		return;
	}
	red = imread(name2 + "c002.jpg", 1);
	green = imread(name2 + "c003.jpg", 1);
	gray = imread(name2 + "c004.jpg", 1);
	img_orig = img0.clone();
	slice=string(fold);
	string sequence(subfold);
	sequencefolder = resultDir + "FVF ID2 - " + sequence;
	resultsfolder = resultDir + "FVF ID2 - " + sequence+"/" + slice;
	//cout << resultsfolder << endl;
	//system(std::string(std::string("mkdir \"") + resultsfolder + "\"").c_str());
	_mkdir(resultDir.c_str());
	_mkdir(sequencefolder.c_str());
	_mkdir(resultsfolder.c_str());
	namedWindow("Blue (Interaction Window)", CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("Blue (Interaction Window)", 620, 640);
	moveWindow("Blue (Interaction Window)", 0, 0);
	img0.copyTo(img);
	cvtColor(img, markerMask, COLOR_BGR2GRAY);
	cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);

	markerMask = 0;
	show();
	setMouseCallback("Blue (Interaction Window)", onMouse, 0);
}

void openall(string name, const string& resultDir){
	
	Mat testImg = imread(name);
	if (testImg.empty())
		return;
	int length = name.length() - 12;
	string names(name.substr(0, length+4));
	string name2(names);
	string test(name.substr(0, length));
	string test2(test);
	unsigned lastSlashFound = name2.find_last_of("/\\");
	string subfold(name2.substr(lastSlashFound - 1, 1));
	string sequence(subfold);
	_mkdir(resultDir.c_str());
	for (int i = 0;i < 15; i++){
		std::stringstream inx;
		inx << setfill('0') << setw(3) << i+1;
		string slicei = inx.str();
		results.push_back(segmentation());
		cout << 100*i/15<<"%" << endl;
		results[i].sequencefolder = resultDir + "FVF ID2 - " + sequence;
		results[i].resultsfolder = resultDir + "FVF ID2 - " + sequence + "/" + slicei;
		_mkdir(results[i].sequencefolder.c_str());
		_mkdir(results[i].resultsfolder.c_str());
		results[i].img0 = imread(test2+slicei+"_c001.jpg");
		results[i].red = imread(test2 + slicei + "_c002.jpg");
		results[i].green = imread(test2 + slicei + "_c003.jpg");
		results[i].gray = imread(test2 + slicei + "_c004.jpg");
		results[i].markers2 = cv::Mat(results[i].markerMask.size(), CV_32S);
		results[i].wresultBoundaries = cv::Mat(markers.size(), CV_8UC3);
		results[i].img_orig = results[i].img0.clone();
		results[i].img0.copyTo(results[i].img);
		cvtColor(results[i].img, results[i].markerMask, COLOR_BGR2GRAY);
		cvtColor(results[i].markerMask, results[i].imgGray, COLOR_GRAY2BGR);
		results[i].xref = -1;
		results[i].yref = -1;
		results[i].markerMask = 0;
		results[i].wshed = results[i].imgGray;
		results[i].currentDisplay = results[i].img0;
		results[i].change = false;
		results[i].changeindex = false;
		//results[i].indexes = cv::Mat(results[i].markers2.size(), CV_8UC1, cvScalar(255));

	}
	zfirst = true;
	xfirst = true;
	pfirst = true;
	rfirst = true;
	ifirst = true;
	namedWindow("Blue (Interaction Window)", CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("Blue (Interaction Window)", 620, 640);
	moveWindow("Blue (Interaction Window)", 0, 0);
	currentidx = 5;
	show();
	cv::namedWindow("Segmentation Result", CV_WINDOW_KEEPRATIO);
	cv::resizeWindow("Segmentation Result", 280, 300);
	moveWindow("Segmentation Result", 940, 340);
	imshow("Segmentation Result", results[currentidx].imgGray);
	setMouseCallback("Blue (Interaction Window)", onMouse, 0);
	
}

int main( int argc, char** argv )
{
	if (argc != 1)
	{
		cout<<"Usage: crypt.exe"<<endl;
		return 0;
	}
	
	/*char number[3] = " ";
	name.copy(number,3 , name.length() - 12);
	string numb(number);
	int currentf ;
	char* file_index=new char[];
	istringstream(numb) >> currentf;*/

	help(); 


	//Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 3, 3 ), Point( 2, 2) );
	//std::vector<std::vector<cv::Point>> contours;
	//std::vector<cv::Mat> channels;

	string resultDir;
	CreateResultDir(resultDir);
	int u = 0;
	string strlabel;
	

	while (1)
	{
		string name;
		cout<<"*******************************************************"<<endl;
		cout << "Please type in the name of the image for segmentation:" << endl;
		getline(cin, name);
		if (name == "quit")
			return 0;	
		
		int sliceIdx = -1;
		openall(name, resultDir);
		if (results[0].img0.empty()){
			cout << "Couldn't open image " << name << endl;
			cout << "Please type in the name of the image for segmentation again:" << endl;
			continue;
		}
		//open(name, resultDir, sliceIdx);
		/*if (img0.empty()){
			cout << "Couldn't open image " << name << endl;
			cout<<"Please type in the name of the image for segmentation again:"<<endl;
			continue;
		}*/

		bSliceFinish = false;

		for(;;)
		{
			//char* filename = new char[];
			



			int c = waitKey(0);

			if( (char)c == 27 )
				return 0;
		
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(40.0, cv::Size(3, 3));
			cv::Mat tmp;
			std::string folder;

			switch((char)c) {
			case 'u':
				currentidx++;
				if (currentidx < 15){
					labellingMarkerAtDown = -1;
					zfirst = true;
					xfirst = true;
					pfirst = true;
					rfirst = true;
					gfirst = true;
					imshow("Blue (Interaction Window)", results[currentidx].currentDisplay);
					imshow("red", results[currentidx].red);
					imshow("green", results[currentidx].green);
					imshow("gray", results[currentidx].gray);
					imshow("Segmentation Result", results[currentidx].wshed);

				}
				else
					currentidx = 14;
				break;
			case 'w':
				currentidx--;
				if (currentidx >= 0){
					labellingMarkerAtDown = -1;
					zfirst = true;
					xfirst = true;
					pfirst = true;
					rfirst = true;
					imshow("Blue (Interaction Window)", results[currentidx].currentDisplay);
					imshow("red", results[currentidx].red);
					imshow("green", results[currentidx].green);
					imshow("gray", results[currentidx].gray);
					imshow("Segmentation Result", results[currentidx].wshed);

				}
				else
					currentidx = 0;
				break;
			case 'l':
				if (!rfirst)
				{
					cout << "Still in the reference-selection mode. Please exit the it before entering the labelling mode." << endl;
					break;
				}
				if (zfirst){
					zfirst = false;
					cout << "Segmentation mode off. Labeling mode on." << endl;
					setMouseCallback("Blue (Interaction Window)", labelMouse, 0);
					
				}
				else{
					zfirst = true;
					cout << "Labeling mode off. Segmentation mode on." << endl;
					setMouseCallback("Blue (Interaction Window)", onMouse, 0);
				}
				
				break;

			case 'n':
				if (xfirst){
					xfirst = false;
					shownumbers(currentidx);
				}
				else{
					xfirst = true;
					show();
				}
				setMouseCallback("Blue (Interaction Window)", onMouse, 0);
				break;



			case 's':
				if (pfirst&&!results[currentidx].markers2.empty()){
					pfirst = false;
					showlabels();
				}
				else{
					pfirst = true;
					show();
				}
				setMouseCallback("Blue (Interaction Window)", onMouse, 0);
				break;
			case 'G':
				if (gfirst&&!results[currentidx].indexes2.empty()){
					gfirst = false;
					showindexes2();
				}
				else{
					gfirst = true;
					show();
				}
				setMouseCallback("Blue (Interaction Window)", onMouse, 0);
				break;

			case 'h':
				help();
			
			case 'r':
				if (!zfirst)
				{
					cout << "Still in the labelling mode. Please exit it before entering the reference-selection mode." << endl;
					break;
				}
				if (rfirst){
					rfirst = false;
					cout << "Segmentation mode off. Please select a reference point." << endl;
					setMouseCallback("Blue (Interaction Window)", referenceMouse, (void*)(&sliceIdx));
				}
				else{
					rfirst = true;
					cout << "Reference-selection mode off. Segmentation mode on." << endl;
					setMouseCallback("Blue (Interaction Window)", onMouse, 0);
				}

				break;

			// the labelling
			case '4':
				if (!zfirst)
					changeMarker(labellingMarkerAtDown, 4);
				break;
			case '1':
				if (!zfirst)
					changeMarker(labellingMarkerAtDown, 1);
				break;
			case '2':
				if (!zfirst)
					changeMarker(labellingMarkerAtDown, 2);
				break;
			case '3':
				if (!zfirst)
					changeMarker(labellingMarkerAtDown, 3);
				break;

			case 'd':
				displayIntensity(resultsfolder,currentidx);
				cout<<"Number of cells in classes: ";
				cout << "Class 1: " << count1 <<"; ";
				cout << "Class 1: " << count2 <<"; ";
				cout << "Class 1: " << count3 <<"; ";
				cout << "Class 1: " << count4 <<endl;
				break;

			case 'g':
				indexAtDown = -1;
				saveindexes(currentidx);
				if (!rfirst)
				{
					cout << "Still in the reference-selection mode. Please exit the it before entering the indexing mode." << endl;
					break;
				}
			

					setMouseCallback("Blue (Interaction Window)", indexMouse, 0);
					
				break;

			case ' ':
				cout<<"One sequence segmentation is finished. "<<endl<<endl;
				for (int i=0; i < (int)results.size() - 1; i++){
					if ((!results[i].markers2.empty()) && (!results[i].wshed.empty()) && (!results[i].wresultBoundaries.empty())){
						cout << i << endl;
						saveResult(i);
					}
				}
				destroyAllWindows();
				zfirst = true;
				xfirst = true;
				pfirst = true;
				rfirst = true;


				bSliceFinish = 1;
				
				
				break;
				
			
			}

			if (bSliceFinish){
				destroyAllWindows();
				break;
			}
			


			/*if( (char)c == 'w' )
			{
				/*doWatersheds();
			}*/
		}
		/*if (!results[currentidx].contours.empty())
			saveResult(currentidx);*/
	}
	
    return 0;
}
#if defined(_MSC_VER) && _MSC_VER >= 1400 
#pragma warning(pop) 
#endif 
