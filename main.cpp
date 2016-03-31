#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


namespace ats
{
	
	void patch_process_RBF(int start_index)
	{
		ats_svm::load_RBF("svm_rbf_classifier.xml");

		ats::ats_frame* pframe_l;
		ats::ats_frame* pframe_c;

		holes_matching::load_file_path();

		char file_path[100];
		int i=start_index;
		while(true)
		{
			bool is_decreasing=false;
			
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
				pframe_l->detect_holes(50,30);

			pframe_c->detect_holes(50,30);

			if(pframe_c->get_hole_num()<pframe_l->get_hole_num())
			{
				delete pframe_c;
				continue;
				//*pframe_c=ats_frame(*pframe_l);
			}
			

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

	void patch_process_LINEAR(int start_index)
	{
		
		ats::ats_svm::load_LINEAR("svm_classifier.xml");

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
				pframe_l->detect_holes(30,10);

			pframe_c->detect_holes(30,10);

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

	void sing_img_process_LINEAR(int img_index)
	{
		char file_path[100];
		sprintf(file_path,"E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_%d.jpg",img_index);
		ats::ats_svm::load_LINEAR("svm_linear_classifier.xml");
		
		ats::ats_frame frame(file_path);
		frame.detect_holes(50,30);
		frame.show();
		
		
		sprintf(file_path,"C:\\Users\\Administrator\\Desktop\\img\\img_%d.jpg",img_index);
		frame.save(file_path);
		frame.save_hole_set("hole_set.txt");
		waitKey();
	}

	void sing_img_process_RBF(int img_index)
	{
		char file_path[100];
		sprintf(file_path,"E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_%d.jpg",img_index);
		ats_svm::load_RBF("svm_rbf_classifier.xml");
		
		ats::ats_frame frame(file_path);
		frame.detect_holes(30,10);
		frame.show();
		
		
		sprintf(file_path,"C:\\Users\\Administrator\\Desktop\\img\\img_%d.jpg",img_index);
		frame.save(file_path);
		frame.save_hole_set("hole_set.txt");
		waitKey();
	}
	
}



int main()
{
	//ats::sing_img_process_RBF(91);
	
	

	
	ats::patch_process_RBF(1);
	//ats::patch_process_LINEAR(1);
	

	
	return 0;


}