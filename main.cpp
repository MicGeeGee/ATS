#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


namespace ats
{
	void console_lines_1(ats_frame& frame)
	{
		
		
		//int index_y[]={9,1531,15,19,206,969,3360,1944,517,818,364,1307,832,3196,3647,3434,482,37};
		//int index_n[]={55,54,36,1051,1081,2816,1140,1828,1873};
		int index_y[]={3510,293,3434};
		int index_n[]={1217,761};
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

		//ats::ats_svm::train<float>(ats::ats_svm::training_data,ats::ats_svm::labels);
		//cout<<ats::ats_svm::predict<float>(ats::ats_svm::test_data);
		//ats::ats_svm::save("svm_classifier.xml");

		//ats::ats_svm::save_data("training_data.txt");
		ats::ats_svm::add_data("training_data.txt");
	}

	void console_lines_2(const ats_frame& frame)
	{
		ats::ats_svm::load_data("training_data.txt");
		ats::ats_svm::train<float>(ats::ats_svm::training_data,ats::ats_svm::labels);
		ats::ats_svm::save("svm_classifier.xml");
	}

	void patch_process(int start_index)
	{
		ats::ats_svm::load("svm_classifier.xml");

		ats::ats_frame* pframe_l;
		ats::ats_frame* pframe_c;

		holes_matching::load_file_path();

		char file_path[100];
		int i=start_index;
		while(true)
		{
			if(i==start_index)
			{
				sprintf(file_path,"E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_%d.jpg",i++);
				pframe_l=new ats::ats_frame(file_path);
			}
			sprintf(file_path,"E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_%d.jpg",i++);
			pframe_c=new ats::ats_frame(file_path);

			if(pframe_c->data==NULL||pframe_l->data==NULL)
			{
				cout<<"Done."<<endl;
				return;
			}

			if(pframe_c->get_index()==95)
				cout<<95<<endl;


			if(pframe_l->get_index()==0)
				pframe_l->detect_holes();

			pframe_c->detect_holes();

			holes_matching::load_last_frame(pframe_l);
			holes_matching::load_current_frame(pframe_c);
			bool matching_res=holes_matching::run();
			holes_matching::print_result();
			
			if(pframe_l->get_index()==0)
			{
				sprintf(file_path,"G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#1\\%Img_%d.jpg",pframe_l->get_index()+1);
				pframe_l->save(file_path);
			}

			

			sprintf(file_path,"G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#1\\%Img_%d.jpg",pframe_c->get_index()+1);
			pframe_c->save(file_path);

			delete pframe_l;
			pframe_l=pframe_c;
			
			
				
			
		}
			
	}

	void sing_img_process()
	{
		ats::ats_svm::load("svm_classifier.xml");
		ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_94.jpg");
		frame.detect_holes();
		frame.show();
		frame.save("C:\\Users\\Administrator\\Desktop\\img\\img_94.jpg");
		frame.save_hole_set("hole_set.txt");

		cout<<ats::ats_svm::get_suprt_vecs()<<endl;
	}
	
}



int main()
{
	//ats::sing_img_process();
	
	ats::patch_process(1);
	waitKey();

	
	return 0;


}