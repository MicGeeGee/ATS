#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;






int main()
{
	
	ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\reEidted.bmp");
	//ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_1.jpg");

	frame.detect_holes();
	frame.show();
	frame.save("C:\\Users\\Administrator\\Desktop\\img\\img.jpg");
	frame.save_g("C:\\Users\\Administrator\\Desktop\\img\\img_g.jpg");
	waitKey();
	return 0;
}