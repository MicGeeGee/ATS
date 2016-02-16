#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


namespace ats
{
	void console_lines(const ats_frame& frame)
	{
		
		
		int index_y[]={9,1531,15,19,206,969,3360,1944,517,818,364,1307,832,3196,3647,3434,482,37};
		int index_n[]={55,54,36,1051,1081,2816,1140,1828,1873};
		map<int,int> map_y;
		map<int,int> map_n;
	

		for(int i=0;i<sizeof(index_y)/sizeof(int);i++)
			map_y[index_y[i]]=1;
		for(int i=0;i<sizeof(index_n)/sizeof(int);i++)
			map_n[index_n[i]]=-1;
		list<ats::hole>::iterator it;
		list<ats::hole> h_set=frame.get_hole_set();
		for(it=h_set.begin();it!=h_set.end();it++)
		{
			if(map_y.find(it->get_index())!=map_y.end())
			{
				ats::ats_svm::training_data.push_back(it->get_m_ft());
				ats::ats_svm::labels.push_back(1);
			}
			if(map_n.find(it->get_index())!=map_n.end())
			{
				ats::ats_svm::training_data.push_back(it->get_m_ft());
				ats::ats_svm::labels.push_back(-1);
			}
		}

		ats::ats_svm::train<float>(ats::ats_svm::training_data,ats::ats_svm::labels);
		//cout<<ats::ats_svm::predict<float>(ats::ats_svm::test_data);
		ats::ats_svm::save("svm_classifier.xml");
	
	}
}



int main()
{
	
	ats::ats_svm::load("svm_classifier.xml");

	//ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\reEidted.bmp");
	ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_1.jpg");

	frame.detect_holes();
	frame.show();
	frame.save("C:\\Users\\Administrator\\Desktop\\img\\img.jpg");
	frame.save_g("C:\\Users\\Administrator\\Desktop\\img\\img_g.jpg");
	
	
	//ats::console_lines(frame);
	

	waitKey();
	return 0;
}