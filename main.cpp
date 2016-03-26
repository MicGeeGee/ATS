#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


namespace ats
{
	

	

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

			


			if(pframe_l->get_index()==0)
				pframe_l->detect_holes();

			pframe_c->detect_holes();

			holes_matching::load_last_frame(pframe_l);
			holes_matching::load_current_frame(pframe_c);
			holes_matching::run();
			

			
			
			if(pframe_l->get_index()==0)
			{
				sprintf(file_path,"G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#1\\%Img_%d.jpg",pframe_l->get_index());
				pframe_l->save(file_path);
			}

			

			sprintf(file_path,"G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#1\\%Img_%d.jpg",pframe_c->get_index());
			pframe_c->save(file_path);

			delete pframe_l;
			pframe_l=pframe_c;
			
			
				
			
		}
			
	}

	void sing_img_process(int img_index)
	{
		char file_path[100];
		sprintf(file_path,"E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_%d.jpg",img_index);

		ats::ats_svm::load("svm_classifier.xml");
		ats::ats_frame frame(file_path);
		frame.detect_holes();
		frame.show();
		
		
		sprintf(file_path,"C:\\Users\\Administrator\\Desktop\\img\\img_%d.jpg",img_index);
		frame.save(file_path);
		frame.save_hole_set("hole_set.txt");

		//cout<<ats::ats_svm::get_suprt_vecs()<<endl;
		waitKey();
	}
	
}



int main()
{
	//ats::sing_img_process(202);
	
	ats::patch_process(1);
	

	
	return 0;


}