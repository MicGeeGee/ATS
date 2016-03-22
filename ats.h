#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdio>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml.hpp>


using namespace std;
using namespace cv;

namespace ats
{
	
	class hole;
	
	
	
	
	class ats_frame : public Mat
	{
	public:

		

		ats_frame(const Mat& RGB_img);
		ats_frame(const string& img_path);

		ats_frame(const ats_frame& frame);

		ats_frame();

		void load_img(const string& img_path)
		{
			*this=imread(img_path);
			origin_img=imread(img_path);
			calc_gradient(*this,grad_val,grad_x,grad_y);
			calc_mid_brgtnss();
		}

		void show(double zoom_scale=0.7);
		void save(const string& tar_path);
		void save_g(const string& tar_path);
		void save_hole_set(const string& tar_path);

		void detect_holes(int thre_min=10);

		int get_grad(int x,int y)const;
		int get_grad(const Point& pos)const;
		int get_phase(int x,int y)const;
		int get_phase(const Point& pos)const;

		int get_brgtnss(int x,int y)const;
		int get_brgtnss(const Point& pos)const;
		int get_grad_x(int x,int y)const;
		int get_grad_y(int x,int y)const;
		int get_grad_x(const Point& p)const;
		int get_grad_y(const Point& p)const;

		uchar mat_at(const Point& p)const;
		list<hole>& get_hole_set();

		int get_dis(const hole& h1,const hole& h2)const;

		int get_max_dis()const;
		static int generate_index()
		{
			return index_generator++;
		}

		list<hole>::iterator get_hole(int index);
		int get_hole_num()const;

		void enroll_overlapping(int index);

		void set_overlapping_num(int n);
		void update_hole(int index,const hole& h);

		int get_index()
		{
			return index;
		}

	private:
		
		static int index_generator;
		int index;

		bool is_labeled;

		//int mid_dis;
		int max_dis;

		int overlapping_num;

		set<int> overlapping_list; //index list of overlapping holes

		
		Mat dis_mat;

		int mid_brgtnss;

		map<int,int> index_map;//index to i in dis_mat

		map<int,list<hole>::iterator> ptr_map;//index to ptr

		Mat origin_img;
	
		Mat grad_val;
		Mat grad_x;
		Mat grad_y;

		

		void reorganize_frags();

		list<hole>::iterator merge_holes(list<hole>::iterator& p_host,list<hole>::iterator& p_guest);

		void calc_max_dis();

		float frag_assessment(hole& h1,hole& h2);

		int calc_dis(const Point& a,const Point& b);
		int calc_dis_sq(const Point& a,const Point& b);

		bool is_holes_detected;

		std::list<hole> hole_set;

		void resize_img(Mat& img,Mat& dst,double scale);
	
		void print_num(Mat& img,int num);
		void print_num(Mat& img,const Point& p,int num);
		void print_num(Mat& img,int num1,int num2);

		Mat convert_to_gray_img(const Mat& img);

		int find_contours(const Mat& binary_img,vector<vector<Point> >& contour_container);
	
		bool bounding_rect_filter(Rect& a,Point gp,float threRatio=0.69);

		void calc_gradient(const Mat& src,Mat& dst,Mat& dst_x,Mat& dst_y);
		void calc_mid_brgtnss();
		void label_pixel(const Point& p,const Scalar& color,int r=0);
		void label_hole(hole& h,const Scalar& color);
		void label_hole(hole& h,const Scalar& color,int num);
		void label_holes();
		

		void morphology_filter(Mat& img,Mat& dst,int morph_operator,int morph_elem,int morph_size);

	};

	class hole
	{
	public:
		
		class shape_ft
		{
		public:
			shape_ft()
			{
				is_loaded=false;
				memset(bins,0,60*sizeof(int));
			}
			shape_ft(const shape_ft& ft)
			{
				this->is_loaded=ft.is_loaded;
				if(is_loaded)
				{
					for(int i=0;i<60;i++)
					this->bins[i]=ft.bins[i];
				}
				else
					memset(bins,0,60*sizeof(int));
			}
			
			
			void calc_ft(ats_frame& frame,const hole& h)
			{
				

				int max_dis=frame.get_max_dis();

				list<hole> hole_set=frame.get_hole_set();
				list<hole>::iterator it;
				for(it=hole_set.begin();it!=hole_set.end();it++)
				{
					if(it->get_index()!=h.get_index())
					{
						
						int rho=frame.get_dis(*it,h);
						//float theta=(-1)*atan((it->get_gp().y-h.get_gp().y)/(it->get_gp().x-h.get_gp().x));//the axis are in different directions from normal one
						float theta=calc_theta(h.get_gp(),it->get_gp());
						int bin_rho;
						int bin_theta;
						bin_rho=calc_bin_rho(rho,max_dis);
						bin_theta=calc_bin_theta(theta);
						int k=bin_rho*12+bin_theta;

						bins[k]++;
					}
					
				}
				is_loaded=true;
			}
			int get_val(ats_frame& frame, const hole& h,int k)
			{
				if(is_loaded)
					return bins[k];
				else
				{
					calc_ft(frame,h);
					return bins[k];
				}
			
			}

			static int calc_bin(ats_frame& frame,const hole& cen_h,const hole& h)
			{
				int rho=frame.get_dis(cen_h,h);
				int max_dis=frame.get_max_dis();
				float theta=calc_theta(cen_h.get_gp(),h.get_gp());
				
				int k=calc_bin_rho(rho,max_dis)*12+calc_bin_theta(theta);
				return k;
			}

			static float calc_theta(const Point& p_cen,const Point& p_h)
			{
				Point delta_div=p_h-p_cen;
				if(delta_div.x==0)
					return 1.57;
				if(delta_div.x>0)
				{
					return (-1)*atan(delta_div.y/(delta_div.x+0.0));
				}
				else
				{

					return 3.14+(-1)*atan(delta_div.y/(delta_div.x+0.0));
					
				}
			}
			static int calc_bin_rho(float rho,int max_dis)
			{
				if(16*rho/max_dis>1)
					return rho?log(16*rho/max_dis)/log(2):0;
				else
					return 0;
			}
			static int calc_bin_theta(float theta)
			{
				return theta>=0?theta/(3.14/6):(6.28+theta)/(3.14/6);
				
			}

		private:
			int bins[60];
			bool is_loaded;
			



		};


		class manual_ft : public Mat
		{
		public:
			manual_ft(const ats_frame&  frame);
			manual_ft(const manual_ft& ft);

			Mat get_val(const ats_frame&  frame,hole& h)
			{
				if(is_loaded)
					return *this;
				else
				{
					calc_ft(frame,h);
					return *this;
				}
			}
			
			float get_val(const ats_frame&  frame,hole& h,int dim_i);

			int assess_ft(const ats_frame&  frame,hole& h)
			{
				if(!is_loaded)
					calc_ft(frame,h);
					
				
				
				int body_contrast_param=100*get_val(frame,h,1);
				int cir_param=100*get_val(frame,h,2);
				int bg_contrast_param=300*get_val(frame,h,3);
				int grad_contrast_param=200*get_val(frame,h,4);

				//int area_param=h.get_area();

				return cir_param+body_contrast_param+bg_contrast_param+grad_contrast_param;


			}

		private:
			

			struct f_point
			{
				float x;
				float y;
				f_point(float x,float y);
				f_point& operator+=(const f_point& p);

			};
			
			bool is_loaded;
			void calc_ft(const ats_frame&  frame,hole& h);

			f_point normalize_vector(float x,float y);
			void set_val(int dim_i,float val);
		};

		hole(ats_frame& frame,const vector<Point>& contour);
		hole(const hole& h);
	
		
		float get_m_ft(int dim_i)
		{
			return this->m_ft.get_val(*p_frame,*this,dim_i);
		}
		int assess_m_ft()
		{
			return this->m_ft.assess_ft(*p_frame,*this);
		}
		Mat get_m_ft()
		{

			return this->m_ft.get_val(*p_frame,*this);
		}


		int get_s_ft(int k)
		{
			return s_ft.get_val(*p_frame,*this,k);
		}


		int get_grad_mid()const;
		int get_area()const;
		Point get_gp()const;
		vector<Point>& get_contour();

		static bool same_pos(const hole& h1,const hole& h2);

		void merge_spe(hole& h);

		void incorporate(const hole& h)
		{
			for(int i=0;i<h.contour.size();i++)
				push_back(h.contour[i],i==(h.contour.size()-1));
		}

		int get_index()const;
		static void index_generator_clear()
		{
			index_generator=0;
		};

		void enroll_overlapping()
		{
			overlapping_num++;
		}
		
		int get_overlapping_num()const
		{
			return overlapping_num;
		}
		void update(const hole& h)
		{
			if(h.area>this->area)
				this->area=h.area;
			this->overlapping_num=h.overlapping_num;
		}

		void set_area_in_series(int val)
		{
			area_in_series=val;
		}
		int get_area_in_series()const
		{
			return area_in_series;
		}

		int get_body_brghtnss_mid()
		{
			if(body_brghtnss_mid>0)
				return body_brghtnss_mid;
			else
			{
				calc_body_info();
				return body_brghtnss_mid;
			}
		}
		int get_body_grad_mid()
		{
			if(body_grad_mid>0)
				return body_grad_mid;
			else
			{
				calc_body_info();
				return body_grad_mid;
			}
		}


	private:
		int index;
		int area;
		Point gp;
		
		int area_in_series;

		int body_brghtnss_mid;
		int body_grad_mid;

		ats_frame* p_frame;

		manual_ft m_ft;
		shape_ft s_ft;
		
		static int index_generator;
		int generate_index();
		
		
		vector<Point> contour;
		
		int grad_mid;
		vector<int> con_grad_arr; 
		
		
	
		int overlapping_num;

		void calc_body_info()
		{
			vector<int> body_brghtnss_arr;
			vector<int> body_grad_arr;
			Rect rect =boundingRect(this->contour);
			for(int i=rect.width/4;i<rect.width*3/4;i++)
				for(int j=rect.height/4;j<rect.height*3/4;j++)
				{
					body_brghtnss_arr.push_back(p_frame->get_brgtnss(rect.x+i,rect.y+j));
					body_grad_arr.push_back(p_frame->get_grad(rect.x+i,rect.y+j));
				}
			std::sort(body_brghtnss_arr.begin(),body_brghtnss_arr.end());
			std::sort(body_grad_arr.begin(),body_grad_arr.end());

			body_brghtnss_mid=body_brghtnss_arr[body_brghtnss_arr.size()/2];
			body_grad_mid=body_grad_arr[body_grad_arr.size()/2];

		}

		static int dis_sq(const Point& p1,const Point& p2);
		void calc_gp();
		void push_back(const Point& p,bool is_last);
		
	};

	
	
	class ats_svm
	{
	public:
		static list<Mat> training_data;
		static list<int> labels;
		static list<Mat> test_data;
		

		static void load(const string& file_name)
		{
			classifier=Algorithm::load<ml::SVM>(file_name);
			
			is_trained=true;
		}

		static void save(const string& file_name)
		{
			if(is_trained)
				classifier->save(file_name);
			else
			{
				cout<<"Error in saving: it has not been trained."<<endl;
				return;
			}
		}
		
		static void save_data(const string& file_name)
		{
			FILE* fp;
			fp=fopen(file_name.c_str(),"w");
			list<Mat>::iterator it=training_data.begin();
			list<int>::iterator lb_it=labels.begin();
			for(;it!=training_data.end();it++,lb_it++)
				fprintf(fp,"%f %f %f %f %d\n",it->at<float>(0,0),it->at<float>(0,1),it->at<float>(0,2),it->at<float>(0,3),*lb_it);
			fclose(fp);
		}
		static void add_data(const string& file_name)
		{
			FILE* fp;
			fp=fopen(file_name.c_str(),"a");
			list<Mat>::iterator it=training_data.begin();
			list<int>::iterator lb_it=labels.begin();
			for(;it!=training_data.end();it++,lb_it++)
				fprintf(fp,"%f %f %f %f %d\n",it->at<float>(0,0),it->at<float>(0,1),it->at<float>(0,2),it->at<float>(0,3),*lb_it);
			fclose(fp);
		}


		static void load_data(const string& file_name)
		{
			FILE* fp;
			fp=fopen(file_name.c_str(),"r");
			float data[4];
			int label;
			int num;
			while(true)
			{
				num=fscanf(fp,"%f%f%f%f%d",&data[0],&data[1],&data[2],&data[3],&label);
				if(num<4)
					break;
				
				Mat data_m(1,4,CV_32F);
				data_m.at<float>(0,0)=data[0];
				data_m.at<float>(0,1)=data[1];
				data_m.at<float>(0,2)=data[2];
				data_m.at<float>(0,3)=data[3];
				training_data.push_back(data_m);
				labels.push_back(label);
			}


			fclose(fp);
		}
		
		static Mat get_suprt_vecs()
		{
			support_vectors = classifier->getSupportVectors();
			return support_vectors;
		}

		template<typename _type> 
		static int predict(const Mat& vec)
		{
			if(is_trained)
			{
				Mat dst;
				param_convert<_type>(vec,dst);

				Mat res(1, 1, CV_32F);
				classifier->predict(dst,res);
				return res.at<float>(0,0);
			}
			else
			{
				cout<<"Error in predicting: it has not been trained."<<endl;
				return 0;
			}
		}
		static void predict(const Mat& vecs,Mat& res)
		{
			res=Mat(vecs.rows, 1, CV_32F);

			if(is_trained)
				classifier->predict(vecs,res);
			else
			{
				cout<<"Error in predicting: it has not been trained."<<endl;
				res=Mat::zeros(vecs.rows,1,CV_32F);
				return;
			}
		}

		template<typename _type> 
		static Mat predict(const list<Mat>& test_data_list)
		{
			if(test_data_list.empty())
			{
				cout<<"Error in predicting: test data is empty."<<endl;
				return Mat();
			}
			int num=test_data_list.size();
			int n_dim=test_data_list.front().cols;

			Mat test_data(num,n_dim,CV_32FC1);

			list<Mat>::const_iterator it=test_data_list.begin();
			int k=0;

			for(;it!=test_data_list.end();it++,k++)
			{	
				for(int i=0;i<n_dim;i++)
					test_data.at<float>(k,i)=it->at<_type>(0,i);
			}
			Mat res;
			predict(test_data,res);
			return res;
		}

		static void train(const Mat& training_data,const Mat& labels)
		{
			classifier->setType(ml::SVM::Types::C_SVC);
			//classifier->setKernel(ml::SVM::KernelTypes::RBF);
			classifier->setKernel(ml::SVM::KernelTypes::LINEAR);

			//classifier->setGamma(20);  // for poly/rbf/sigmoid
			//classifier->setC(7);       // for CV_classifier_C_SVC, CV_classifier_EPS_SVR and CV_classifier_NU_SVR
			classifier->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1E-6));
			classifier->train(training_data, ml::SampleTypes::ROW_SAMPLE, labels);

			is_trained=true;
			support_vectors = classifier->getSupportVectors();
		}
		
		template<typename _type> 
		static void train(const list<Mat>& training_data_list,const list<int>& label_list)
		{
			if(training_data_list.empty())
			{
				cout<<"Error in training: training data is empty."<<endl;
				return;
			}
			int num=training_data_list.size();
			int n_dim=training_data_list.front().cols;

			Mat training_data(num,n_dim,CV_32FC1);
			Mat labels(num, 1, CV_32SC1);

			list<Mat>::const_iterator t_it=training_data_list.begin();
			list<int>::const_iterator l_it=label_list.begin();

			int k=0;
			for(;t_it!=training_data_list.end();t_it++,l_it++,k++)
			{	
				for(int i=0;i<n_dim;i++)
					training_data.at<float>(k,i)=t_it->at<_type>(0,i);
				
				labels.at<int>(k,0)=*l_it;
			}
			train(training_data,labels);
		}
		
		
		
	private:
		template<typename _type> 
		static void param_convert(const Mat& src,Mat& dst)
		{
			if(src.empty())
			{
				cout<<"Error in converting: source mat is emtpy."<<endl;
				return;
			}
			
			int num=src.rows;
			int n_dim=src.cols;

			dst=Mat(num,n_dim,CV_32FC1);

			for(int i=0;i<num;i++)
				for(int j=0;j<n_dim;j++)
					dst.at<float>(i,j)=src.at<_type>(i,j);

			
		}
		template<typename _type> 
		static void param_convert(const list<Mat>& src,Mat& dst)
		{
			if(src.empty())
			{
				cout<<"Error in converting: source mat is emtpy."<<endl;
				return;
			}
			list<Mat>::const_iterator it=src.begin();
			int num=src.size();
			int n_dim=src.front().cols;

			dst=Mat(num,n_dim,CV_32FC1);

			int k=0;
			for(;it!=src.end();it++,k++)
				for(int i=0;i<n_dim;i++)
					dst.at<float>(k,i)=it->at<_type>(k,i);
		}


		static Ptr<cv::ml::SVM> classifier;
		static bool is_trained;
		static Mat support_vectors;
	};

	class holes_matching
	{
	public:
		static void load_file_path()
		{
			char str[1000];
			sprintf(str,"G:\\OPENCV_WORKSPACE\\ATS_IMG_RESULT\\#1_file\\record_%d.txt",time((time_t*)NULL));
				
			file_path=string(str);
			
		}
		static void load_last_frame(ats_frame* p_frame)
		{
			last_frame=p_frame;
		}
		static void load_current_frame(ats_frame* p_frame)
		{
			current_frame=p_frame;
		}
		
		static bool run()
		{
			total_cost=calc_matching_cost();

			current_frame->set_overlapping_num(overlapping_num);


			if(total_cost==-1||total_cost>0.3)
				return	false;//means the hole num has decreased.
			else
				return true;
		}
		/*
		static void print_result()
		{
			cout<<"#"<<last_frame->get_index()<<" & "<<current_frame->get_index()<<":"<<endl;
			//matching result:
			cout<<"matching pairs(last,current):"<<endl;
			for(int i=0;i<cost_m.rows;i++)
				cout<<"("<<revindex_map_l[i]<<","<<revindex_map_c[row_res[i]]<<")"<<endl;
			cout<<"total matching cost:"<<total_cost<<endl;
			cout<<"cost matrix:"<<endl<<cost_m<<endl;

			//index map:
			cout<<"row index:"<<endl;
			for(int i=0;i<cost_m.rows;i++)
				cout<<revindex_map_l[i]<<endl;
			cout<<"column index:"<<endl;
			for(int i=0;i<cost_m.rows;i++)
				cout<<revindex_map_c[i]<<endl;
		}*/
		static void print_result()
		{
			cout<<"#"<<last_frame->get_index()<<" & "<<current_frame->get_index()<<":"<<total_cost<<endl;
			//cout<<cost_m<<endl;

			FILE* fp=fopen(file_path.c_str(),"a");

			fprintf(fp,"#%d & #%d:\n",last_frame->get_index(),current_frame->get_index());
			fprintf(fp,"matching pairs(last,current):\n");
			for(int i=0;i<cost_m.rows;i++)
				fprintf(fp,"(%d,%d)\n",revindex_map_l[i],revindex_map_c[row_res[i]]);
			fprintf(fp,"total matching cost:%f\n",total_cost);
			
		

			fclose(fp);
		
		}
	private:
		static int overlapping_num;

		static string file_path;

		static map<int,int> result_row;
		static map<int,int> result_col;

		static float total_cost;

		static int row_res[100000];
		static int col_res[100000];

		static map<int,int> revindex_map_c;//i to index
		static map<int,int> revindex_map_l;

		static ats_frame* last_frame;
		static ats_frame* current_frame;
		static bool is_l_loaded;
		static bool is_c_loaded;
		

		static float calc_matching_cost()
		{
		    bool is_successfull=calc_cost_m();
			
			if(!is_successfull)
				return -1;

			float res=hungarian<float>(cost_m,row_res,col_res);
			res/=cost_m.cols;


			area_matching(file_path);

			//if the matching method fails(i.e. the position pattern has changed a lot)
			//then the overlapping detection will not be run.
			if(res>0.3)
				return -1;

			/*
			//for those holes matching dumppy holes
			list<hole>::iterator it;
			list<hole>& cur_hole_set=current_frame->get_hole_set();
			for(int i=last_frame->get_hole_num();i<cost_m.cols;i++)
			{
				for(it=cur_hole_set.begin();it!=cur_hole_set.end();it++)
				{	
					if(frag_assessment(*current_frame,*(current_frame->get_hole(revindex_map_c[row_res[i]])),*it)<1)
					{
						
					}
				}
			}
			*/


			handle_matching_res(file_path);


			//return total matching cost
			return res;
		}

		static void area_matching(const string& file_path)
		{
			FILE* fp=fopen(file_path.c_str(),"a");
			for(int i=0;i<cost_m.rows;i++)
			{
	
				if(revindex_map_l[i]==-1)
					continue;

				
				int area_c=(current_frame->get_hole(revindex_map_c[row_res[i]]))->get_area();
				int area_l=(last_frame->get_hole(revindex_map_l[i]))->get_area_in_series();
				if(area_c>area_l)
					(current_frame->get_hole(revindex_map_c[row_res[i]]))->set_area_in_series(area_c);
				else
					(current_frame->get_hole(revindex_map_c[row_res[i]]))->set_area_in_series(area_l);

				printf("(%d,%d):(%d,%d)\n",revindex_map_l[i],revindex_map_c[row_res[i]],area_l,area_c);
				fprintf(fp,"(%d,%d):(%d,%d)\n",revindex_map_l[i],revindex_map_c[row_res[i]],area_l,area_c);
			}
			fclose(fp);
		
		}

		static void handle_matching_res()
		{
			for(int i=0;i<cost_m.rows;i++)
			{
				if(revindex_map_l[i]==-1)
					continue;

				//first, update all the overlapping numbers of the current frame to the ones of the previous frame
				current_frame->update_hole(revindex_map_c[row_res[i]],*(last_frame->get_hole(revindex_map_l[i])));
				//then judge overlappings and make increment
				if(judge_overlapping(*(last_frame->get_hole(revindex_map_l[i])),*(current_frame->get_hole(revindex_map_c[row_res[i]]))))
				{
					cout<<"new overlapping location(pre_index,cur_index:pre_area->cur_area):"<<endl;
					cout<<revindex_map_l[i]<<","<<revindex_map_c[row_res[i]]<<":"
					<<last_frame->get_hole(revindex_map_l[i])->get_area()<<"->"<<(current_frame->get_hole(revindex_map_c[row_res[i]]))->get_area()<<endl;
					current_frame->enroll_overlapping(revindex_map_c[row_res[i]]);
				}

				int area_c=(current_frame->get_hole(revindex_map_c[row_res[i]]))->get_area();
				int area_l=(last_frame->get_hole(revindex_map_l[i]))->get_area_in_series();
				if(area_c>area_l)
					(current_frame->get_hole(revindex_map_c[row_res[i]]))->set_area_in_series(area_c);
				else
					(current_frame->get_hole(revindex_map_c[row_res[i]]))->set_area_in_series(area_l);


			}
		}
		static void handle_matching_res(const string& file_path)
		{
			FILE* fp=fopen(file_path.c_str(),"a");
			for(int i=0;i<cost_m.rows;i++)
			{
				if(revindex_map_l[i]==-1)
					continue;

				//first, update all the overlapping numbers of the current frame to the ones of the previous frame
				current_frame->update_hole(revindex_map_c[row_res[i]],*(last_frame->get_hole(revindex_map_l[i])));
				//then judge overlappings and make increment
				if(judge_overlapping(*(last_frame->get_hole(revindex_map_l[i])),*(current_frame->get_hole(revindex_map_c[row_res[i]]))))
				{
					overlapping_num++;

					cout<<"new overlapping location(pre_index,cur_index:pre_area->cur_area):"<<endl;
					cout<<revindex_map_l[i]<<","<<revindex_map_c[row_res[i]]<<":"
					<<last_frame->get_hole(revindex_map_l[i])->get_area()<<"->"<<(current_frame->get_hole(revindex_map_c[row_res[i]]))->get_area()<<endl;
					
					fprintf(fp,"new overlapping location(pre_index,cur_index:pre_area->cur_area):\n%d,%d:%d->%d\n",revindex_map_l[i],revindex_map_c[row_res[i]],last_frame->get_hole(revindex_map_l[i])->get_area_in_series(),(current_frame->get_hole(revindex_map_c[row_res[i]]))->get_area());

					//current_frame->enroll_overlapping(revindex_map_c[row_res[i]]);

					
				}



				


			}
			

			fclose(fp);
		}

		
	


		static bool judge_overlapping(const hole& pre_h,const hole& cur_h)
		{
			return cur_h.get_area()>pre_h.get_area_in_series()*2.5&&calc_dis_sq(pre_h.get_gp(),cur_h.get_gp())>=2;
		}


		


		
		

		static bool calc_cost_m()
		{


			list<hole>& h_set_l=last_frame->get_hole_set();
			list<hole>& h_set_c=current_frame->get_hole_set();

			int num_l=h_set_l.size();
			int num_c=h_set_c.size();
			if(num_l<=num_c)
			{
				int num=num_c;
				float inf=0;
				cost_m=Mat::zeros(num,num,CV_32F);
				
				int m=0;
				int n=0;

				list<hole>::iterator it_l;
				list<hole>::iterator it_c;
				for(it_l=h_set_l.begin(),m=0;it_l!=h_set_l.end();it_l++,m++)
					for(it_c=h_set_c.begin(),n=0;it_c!=h_set_c.end();it_c++,n++)
					{
						revindex_map_l[m]=it_l->get_index();
						revindex_map_c[n]=it_c->get_index();

						float cost=chi_sq_test(*it_l,*it_c);
						cost_m.at<float>(m,n)=cost;
						//cost_m.at<float>(n,m)=cost;
						if(cost>inf)
							inf=cost;
					}

				//left part of the reminder:
				for(int i=m;i<n;i++)
					for(int j=0;j<m;j++)
					{
						revindex_map_l[i]=-1;
						//cost_m.at<float>(i,j)=inf*10000;
						cost_m.at<float>(i,j)=inf;
					}
				//right part of the reminder:
				for(int i=m;i<n;i++)
					for(int j=m;j<n;j++)
						//cost_m.at<float>(i,j)=inf*10000;
						cost_m.at<float>(i,j)=inf;
				return true;

			}
			else
			{
				cout<<"Warning:number of holes have decreased."<<endl;
				return false;
			}

		}
		static Mat cost_m;//row index for last frame, column index of current one
		static float chi_sq_test(hole& h1,hole& h2)
		{
			float val=0;
			for(int i=0;i<60;i++)
			{
				if((h1.get_s_ft(i)+h2.get_s_ft(i))!=0)
					val+=(h1.get_s_ft(i)-h2.get_s_ft(i))*(h1.get_s_ft(i)-h2.get_s_ft(i))/((h1.get_s_ft(i)+h2.get_s_ft(i))+0.0);
				else
					val+=0;
			}

			return val/2;
		}
		
		
		static int calc_dis_sq(const Point& a,const Point& b)
		{
			return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y);
		}

		template<typename _type> 
		//static float hungarian(const Mat& assigncost,map<int,int>& rowsol,map<int,int>& colsol)
		static float hungarian(const Mat& assigncost,int* rowsol,int* colsol)
		{

		  int dim=assigncost.cols;
		  unsigned char unassignedfound;
		  int  i, imin, numfree = 0, prvnumfree, f, i0, k, freerow, *pred, *free;
		  int  j, j1, j2, endofpath, last, low, up, *collist, *matches;
		  float min, h, umin, usubmin, v2, *d, *v;

		  free = new int[dim];       // list of unassigned rows.
		  collist = new int[dim];    // list of columns to be scanned in various ways.
		  matches = new int[dim];    // counts how many times a row could be assigned.
		  d = new float[dim];         // 'cost-distance' in augmenting path calculation.
		  pred = new int[dim];       // row-predecessor of column in augmenting/alternating path.
		  v = new float[dim];

		  // init how many times a row will be assigned in the column reduction.
		  for (i = 0; i < dim; i++)  
			matches[i] = 0;

		  // COLUMN REDUCTION 
		  for (j = dim-1; j >= 0; j--)    // reverse order gives better results.
		  {
			// find minimum cost over rows.
	  
			min =assigncost.at<_type>(0,j);
			imin = 0;
			for (i = 1; i < dim; i++)  
			  if (assigncost.at<_type>(i,j) < min) 
			  { 
				min = assigncost.at<_type>(i,j);
				imin = i;
			  }
			v[j] = min; 

			if (++matches[imin] == 1) 
			{ 
			  // init assignment if minimum row assigned for first time.
			  rowsol[imin] = j; 
			  colsol[j] = imin; 
			}
			else
			  colsol[j] = -1;        // row already assigned, column not assigned.
		  }

		  // REDUCTION TRANSFER
		  for (i = 0; i < dim; i++) 
			if (matches[i] == 0)     // fill list of unassigned 'free' rows.
			  free[numfree++] = i;
			else
			  if (matches[i] == 1)   // transfer reduction from rows that are assigned once.
			  {
				j1 = rowsol[i]; 
				//min = BIG;
				min = 1e+10;
				for (j = 0; j < dim; j++)  
				  if (j != j1)
					if (assigncost.at<_type>(i,j) - v[j] < min) 
					  min = assigncost.at<_type>(i,j) - v[j];
				v[j1] = v[j1] - min;
			  }

		  // AUGMENTING ROW REDUCTION 
	  int loopcnt = 0;           // do-loop to be done twice.
	  do
	  {
		loopcnt++;

		// scan all free rows.
		// in some cases, a free row may be replaced with another one to be scanned next.
		k = 0; 
		prvnumfree = numfree; 
		numfree = 0;             // start list of rows still free after augmenting row reduction.
		while (k < prvnumfree)
		{
		  i = free[k]; 
		  k++;

		  // find minimum and second minimum reduced cost over columns.
		  umin = assigncost.at<_type>(i,0)- v[0]; 
		  j1 = 0; 
		  //usubmin = BIG;
		  usubmin =1e+10;
		  for (j = 1; j < dim; j++) 
		  {
			h = assigncost.at<_type>(i,j) - v[j];
			if (h < usubmin)
			  if (h >= umin) 
			  { 
				usubmin = h; 
				j2 = j;
			  }
			  else 
			  { 
				usubmin = umin; 
				umin = h; 
				j2 = j1; 
				j1 = j;
			  }
		  }

		  i0 = colsol[j1];
	  
		  /* Begin modification by Yefeng Zheng 03/07/2004 */
		  //if( umin < usubmin )
		  if (fabs(umin-usubmin) > 1e-10) 
		  /* End modification by Yefeng Zheng 03/07/2004 */

			// change the reduction of the minimum column to increase the minimum
			// reduced cost in the row to the subminimum.
			v[j1] = v[j1] - (usubmin - umin);
		  else                   // minimum and subminimum equal.
			if (i0 >= 0)         // minimum column j1 is assigned.
			{ 
			  // swap columns j1 and j2, as j2 may be unassigned.
			  j1 = j2; 
			  i0 = colsol[j2];
			}

		  // (re-)assign i to j1, possibly de-assigning an i0.
		  rowsol[i] = j1; 
		  colsol[j1] = i;

		  if (i0 >= 0)           // minimum column j1 assigned earlier.

	/* Begin modification by Yefeng Zheng 03/07/2004 */
			  //if( umin < usubmin )
			  if (fabs(umin-usubmin) > 1e-10) 
	/* End modification by Yefeng Zheng 03/07/2004 */
          
			  // put in current k, and go back to that k.
			  // continue augmenting path i - j1 with i0.
			  free[--k] = i0; 
			else 
			  // no further augmenting reduction possible.
			  // store i0 in list of free rows for next phase.
			  free[numfree++] = i0; 
		}
	  }
	  while (loopcnt < 2);       // repeat once.

	  // AUGMENT SOLUTION for each free row.
	  for (f = 0; f < numfree; f++) 
	  {
		freerow = free[f];       // start row of augmenting path.

		// Dijkstra shortest path algorithm.
		// runs until unassigned column added to shortest path tree.
		for (j = 0; j < dim; j++)  
		{ 
		
		  d[j] = assigncost.at<_type>(freerow,j) - v[j]; 
		  pred[j] = freerow;
		  collist[j] = j;        // init column list.
		}

		low = 0; // columns in 0..low-1 are ready, now none.
		up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
				 // columns in up..dim-1 are to be considered later to find new minimum, 
				 // at this stage the list simply contains all columns 
		unassignedfound = false;
		do
		{
		  if (up == low)         // no more columns to be scanned for current minimum.
		  {
			last = low - 1; 

			// scan columns for up..dim-1 to find all indices for which new minimum occurs.
			// store these indices between low..up-1 (increasing up). 
			min = d[collist[up++]]; 
			for (k = up; k < dim; k++) 
			{
			  j = collist[k]; 
			  h = d[j];
			  if (h <= min)
			  {
				if (h < min)     // new minimum.
				{ 
				  up = low;      // restart list at index low.
				  min = h;
				}
				// new index with same minimum, put on undex up, and extend list.
				collist[k] = collist[up]; 
				collist[up++] = j; 
			  }
			}

			// check if any of the minimum columns happens to be unassigned.
			// if so, we have an augmenting path right away.
			for (k = low; k < up; k++) 
			  if (colsol[collist[k]] < 0) 
			  {
				endofpath = collist[k];
				unassignedfound = true;
				break;
			  }
		  }

		  if (!unassignedfound) 
		  {
			// update 'distances' between freerow and all unscanned columns, via next scanned column.
			j1 = collist[low]; 
			low++; 
			i = colsol[j1]; 
			h = assigncost.at<_type>(i,j1) - v[j1] - min;

			for (k = up; k < dim; k++) 
			{
			  j = collist[k]; 
			  v2 = assigncost.at<_type>(i,j) - v[j] - h;
			  if (v2 < d[j])
			  {
				pred[j] = i;
				if (v2 == min)   // new column found at same minimum value
				  if (colsol[j] < 0) 
				  {
					// if unassigned, shortest augmenting path is complete.
					endofpath = j;
					unassignedfound = true;
					break;
				  }
				  // else add to list to be scanned right away.
				  else 
				  { 
					collist[k] = collist[up]; 
					collist[up++] = j; 
				  }
				d[j] = v2;
			  }
			}
		  } 
		}
		while (!unassignedfound);

		// update column prices.
		for (k = 0; k <= last; k++)  
		{ 
		  j1 = collist[k]; 
		  v[j1] = v[j1] + d[j1] - min;
		}

		// reset row and column assignments along the alternating path.
		do
		{
		  i = pred[endofpath]; 
		  colsol[endofpath] = i; 
		  j1 = endofpath; 
		  endofpath = rowsol[i]; 
		  rowsol[i] = j1;
		}
		while (i != freerow);
	  }

	  // calculate optimal cost.
	  float lapcost = 0;
	  for (i = 0; i < dim; i++)  
	  {
		j = rowsol[i];
		lapcost = lapcost + assigncost.at<_type>(i,j); 
	  }

	  // free reserved memory.
	  delete[] pred;
	  delete[] free;
	  delete[] collist;
	  delete[] matches;
	  delete[] d;
	  delete[] v;
	  return lapcost;
	}
		

	};

	
}