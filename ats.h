#include <vector>
#include <list>
#include <cmath>
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
		void show(double zoom_scale=0.7);
		void save(const string& tar_path);
		void save_g(const string& tar_path);


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
		list<hole> get_hole_set()const
		{
			return hole_set;
		}

		int get_dis(const hole& h1,const hole& h2)const
		{
			int index1=h1.get_index();
			int index2=h2.get_index();

			return dis_mat.at<int>(index_map.find(index1)->second,index_map.find(index2)->second);
		}

		int get_mid_dis()const
		{
			return mid_dis;
		}

	private:
		
		int mid_dis;

		Mat dis_mat;

		int mid_brgtnss;

		map<int,int> index_map;

		Mat origin_img;
	
		Mat grad_val;
		Mat grad_x;
		Mat grad_y;

		void calc_mid_dis()
		{
			int dis_mat_i=0;
			dis_mat=Mat::zeros(hole_set.size(),hole_set.size(),CV_32S);

			vector<int> dis_arr;
			list<hole>::iterator it1;
			list<hole>::iterator it2;
			for(it1=hole_set.begin();it1!=hole_set.end();it1++)
				for(it2=hole_set.begin();it2!=hole_set.end();it2++)
				{
					if(index_map.find(it1->get_index())==index_map.end())
						index_map[it1->get_index()]=dis_mat_i++;
					if(index_map.find(it2->get_index())==index_map.end())
						index_map[it2->get_index()]=dis_mat_i++;
					


					dis_arr.push_back(calc_dis(it1->get_gp(),it2->get_gp()));
					dis_mat.at<int>(index_map[it1->get_index()],index_map[it2->get_index()])=dis_arr.front();
					
				}
			mid_dis=dis_arr[dis_arr.size()/2];	
		}

		int calc_dis(const Point& a,const Point& b)
		{
			return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
		}

		bool is_holes_detected;

		std::list<hole> hole_set;

		void resize_img(Mat& img,Mat& dst,double scale);
	
		void print_num(Mat& img,int num);
		void print_num(Mat& img,const Point& p,int num);
		Mat convert_to_gray_img(const Mat& img);

		int find_contours(const Mat& binary_img,vector<vector<Point> >& contour_container);
	
		bool bounding_rect_filter(Rect& a,Point gp,float threRatio=0.69);

		void calc_gradient(const Mat& src,Mat& dst,Mat& dst_x,Mat& dst_y);
		void calc_mid_brgtnss();
		void label_pixel(const Point& p,const Scalar& color,int r=0);
		void label_hole(const hole& h,const Scalar& color);
		void label_hole(const hole& h,const Scalar& color,int num);

		

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
			void calc_ft(const ats_frame& frame,const hole& h)
			{
				int mid_dis=frame.get_mid_dis();

				list<hole> hole_set=frame.get_hole_set();
				list<hole>::iterator it;
				for(it=hole_set.begin();it!=hole_set.end();it++)
				{
					if(it->get_index!=h.get_index())
					{
						int rho=frame.get_dis(*it,h);
						float theta=atan((it->get_gp().y-h.get_gp().y)/(it->get_gp().x-h.get_gp().x));
						int k=(calc_bin_rho(rho)-1)*12+calc_bin_theta(theta,mid_dis);
						bins[k]++;
					}
					
				}
				is_loaded=true;
			}
			int get_val(const ats_frame& frame, const hole& h,int k)
			{
				if(is_loaded)
					return bins[k];
				else
				{
					calc_ft(frame,h);
					return bins[k];
				}
			
			}
		private:
			int bins[60];
			bool is_loaded;

			int calc_bin_rho(float theta)
			{
				return theta?(theta>0?theta/(3.14/6):3.14-theta/(3.14/6)):1;
			}
			int calc_bin_theta(int rho,int mid_dis)
			{
				return rho?log(16*rho/mid_dis)/log(2)+1:1;
			}


		};


		class manual_ft : public Mat
		{
		public:
			manual_ft(const ats_frame&  frame,const vector<Point>& contour);
			manual_ft(const manual_ft& ft);

			Mat get_val(const ats_frame&  frame,const vector<Point>& contour)
			{
				if(is_loaded)
					return *this;
				else
				{
					calc_ft(frame,contour);
					return *this;
				}
			}
			
			float get_val(const ats_frame&  frame,const vector<Point>& contour,int dim_i);

			int assess_ft(const ats_frame&  frame,const vector<Point>& contour)
			{
				if(is_loaded)
				{
					int cir_param=get_val(frame,contour,2)*100;
					int body_contrast_param=100*get_val(frame,contour,1);
					int bg_contrast_param=150*get_val(frame,contour,3);
					return cir_param+body_contrast_param+bg_contrast_param;
				}
				else
				{
					calc_ft(frame,contour);
					int cir_param=get_val(frame,contour,2)*100;
					int body_contrast_param=100*get_val(frame,contour,1);
					int bg_contrast_param=150*get_val(frame,contour,3);
					return cir_param+body_contrast_param+bg_contrast_param;
				}
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
			void calc_ft(const ats_frame&  frame,const vector<Point>& contour);

			f_point normalize_vector(float x,float y);
			void set_val(int dim_i,float val);
		};

		hole(const ats_frame& frame,const vector<Point>& contour);
		hole(const hole& h);
	
		
		float get_m_ft(int dim_i)
		{
			return this->m_ft.get_val(*p_frame,this->contour,dim_i);
		}
		int assess_m_ft()
		{
			return this->m_ft.assess_ft(*p_frame,this->contour);
		}
		Mat get_m_ft()
		{

			return this->m_ft.get_val(*p_frame,contour);
		}


		int get_s_ft(int k)const
		{
			return s_ft.get_val(*p_frame,*this,k);
		}


		int get_grad_mid()const;
		int get_area()const;
		Point get_gp()const;
		vector<Point> get_contour()const;

		static bool same_pos(const hole& h1,const hole& h2);

		void merge_spe(hole& h);
		int get_index()const;
		static void index_generator_clear()
		{
			index_generator=0;
		};
	private:
		const ats_frame* p_frame;

		manual_ft m_ft;
		shape_ft s_ft;
		int index;
		static int index_generator;
		int generate_index();
		
		Point gp;
		vector<Point> contour;
		
		int grad_mid;
		vector<int> con_grad_arr; 
		
		int area;
	
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
				fprintf(fp,"%f %f %f %d\n",it->at<float>(0,0),it->at<float>(0,1),it->at<float>(0,2),*lb_it);
			fclose(fp);
		}
		static void add_data(const string& file_name)
		{
			FILE* fp;
			fp=fopen(file_name.c_str(),"a");
			list<Mat>::iterator it=training_data.begin();
			list<int>::iterator lb_it=labels.begin();
			for(;it!=training_data.end();it++,lb_it++)
				fprintf(fp,"%f %f %f %d\n",it->at<float>(0,0),it->at<float>(0,1),it->at<float>(0,2),*lb_it);
			fclose(fp);
		}


		static void load_data(const string& file_name)
		{
			FILE* fp;
			fp=fopen(file_name.c_str(),"r");
			float data[3];
			int label;
			int num;
			while(true)
			{
				num=fscanf(fp,"%f%f%f%d",&data[0],&data[1],&data[2],&label);
				if(num<4)
					break;
				
				Mat data_m(1,3,CV_32F);
				data_m.at<float>(0,0)=data[0];
				data_m.at<float>(0,1)=data[1];
				data_m.at<float>(0,2)=data[2];
				training_data.push_back(data_m);
				labels.push_back(label);
			}


			fclose(fp);
		}
		
		static Mat get_suprt_vecs()
		{
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
		static void load_last_frame(const ats_frame* p_frame)
		{
			last_frame=p_frame;
		}
		static void load_current_frame(const ats_frame* p_frame)
		{
			current_frame=p_frame;
		}
		static int calc_matching_cost()
		{
			
		
		}
	private:
		static const ats_frame* last_frame;
		static const ats_frame* current_frame;
		static bool is_l_loaded;
		static bool is_c_loaded;
		
		static void calc_cost_m()
		{
		
		}
		static Mat cost_m;
		static float chi_sq_test(const hole& h1,const hole& h2)
		{
			float val=0;
			for(int i=0;i<60;i++)
			{
				if((h1.get_s_ft(i)+h2.get_s_ft(i))!=0)
					val+=(h1.get_s_ft(i)-h2.get_s_ft(i))*(h1.get_s_ft(i)-h2.get_s_ft(i))/(h1.get_s_ft(i)+h2.get_s_ft(i));
				else
					val+=0;
			}

			return val/2;
		}
		

	};

	
}