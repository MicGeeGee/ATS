#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


namespace ats
{
	

	
	void patch_process(const string& src_dir,const string& dst_img_dir,const string& dst_doc_dir)
	{
		ats::ats_frame* pframe_l;
		ats::ats_frame* pframe_c;

		char file_path[100];
		char str[100];
		string str_img;
		
		holes_matching::load_file_path(dst_doc_dir);

		int i=1;
		while(true)
		{
			bool is_decreasing=false;
			
			if(i==1)
			{

				sprintf(str,"\\Img_%d.jpg",i++);
				pframe_l=new ats::ats_frame(src_dir+string(str));
			}
			

			sprintf(str,"\\Img_%d.jpg",i++);
			pframe_c=new ats::ats_frame(src_dir+string(str));

			if(pframe_c->data==NULL||pframe_l->data==NULL)
			{
				cout<<"Done."<<endl;
				return;
			}

			

			if(pframe_l->get_index()==0)
				pframe_l->detect_holes();

			pframe_c->detect_holes();

			if(pframe_c->get_hole_num()<pframe_l->get_hole_num())
			{
				delete pframe_c;
				continue;
				
			}
			

			holes_matching::load_last_frame(pframe_l);
			holes_matching::load_current_frame(pframe_c);
			holes_matching::run();
			

			
			
			if(pframe_l->get_index()==0)
			{

				sprintf(str,"\\Img_%d.jpg",pframe_l->get_index());
				pframe_l->save(dst_img_dir+string(str));
			}

			
			sprintf(str,"\\Img_%d.jpg",pframe_c->get_index());
			pframe_c->save(dst_img_dir+string(str));

			delete pframe_l;
			pframe_l=pframe_c;
			
			
				
			
		}
	}
	void sing_process(const string& src_path,const string& dst_path)
	{
		ats::ats_frame frame(src_path);
		frame.detect_holes();
		frame.show();
		frame.save(dst_path);
		frame.save_hole_set("hole_set.txt");
		waitKey();
	}

	void process()
	{
		bool p_or_s=param_manager::get_p_or_s;
		string img_src_dir=param_manager::get_src_dir();
		string img_dst_dir=param_manager::get_dst_dir();
		string txt_dst_dir=param_manager::get_txt_dir();


		if(p_or_s)
			patch_process(img_src_dir,img_dst_dir,txt_dst_dir);
		else
			sing_process(img_src_dir,img_dst_dir);

	}


}



int main()
{

	
	ats::param_manager::set_detection("svm_rbf_classifier_0.xml",true,-1);
	ats::param_manager::set_thre(50,150,30);
	ats::param_manager::set_process(1,"G:\\OPENCV_WORKSPACE\\ATS_EXP_SRC\\#0_normal_overlapping","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#0","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#0_file");
	ats::param_manager::save("param_setting_0.xml");

	ats::param_manager::load("param_setting_0.xml");
	ats::ats_svm::load();

	ats::process();

	//ats::sing_img_process_RBF("E:\\OPENCV_WORKSPACE\\videos\\BulletHoleDetectionImg\\reEidted.bmp",11111);
	
	
	//ats::patch_process_RBF("G:\\OPENCV_WORKSPACE\\ATS_EXP_SRC\\#2_gone_and_back_with_vibration","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#2_file","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#2");


	//ats::patch_process_RBF("G:\\OPENCV_WORKSPACE\\ATS_EXP_SRC\\#6_uneven_illuminated","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#6_file","G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#6");
	
	//ats::patch_process_RBF(1);
	//ats::patch_process_LINEAR(1);
	

	
	return 0;


}