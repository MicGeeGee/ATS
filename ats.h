#include <vector>
#include <list>
#include <cmath>
#include <iostream>
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

		static list<Mat> training_data;
		static list<int> labels;
		static list<Mat> test_data;

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


	private:
		
		int mid_brgtnss;
		Mat origin_img;
	
		Mat grad_val;
		Mat grad_x;
		Mat grad_y;

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

		uchar mat_at(const Point& p);

		void morphology_filter(Mat& img,Mat& dst,int morph_operator,int morph_elem,int morph_size);

	};



	class hole
	{
	public:
		
		



		class shape_ft : public Mat
		{

		};
		class manual_ft : public Mat
		{
		public:
			manual_ft(const ats_frame& src_frame,const vector<Point>& src_contour);
			manual_ft(const manual_ft& ft);

			
			int get_val(int dim_i)const;

			int assess_ft()const
			{

				int cir_param=(get_val(8)+0.0)/10*0.2;
				int contrast_param=1000/(get_val(4)+1);
					
				return cir_param+contrast_param;
			}

		private:
			ats_frame* p_frame;
			vector<Point>* p_contour;

			struct f_point
			{
				float x;
				float y;
				f_point(float x,float y);
				f_point& operator+=(const f_point& p);

			};
			
			bool is_loaded;
			void calc_ft(const ats_frame& src_frame,const vector<Point>& src_contour);

			f_point normalize_vector(int x,int y);
			void set_val(int dim_i,int val);
		}m_ft;

		hole(const ats_frame& frame,const vector<Point>& contour);
		hole(const hole& h);
	
		
		int get_m_ft(int dim_i)const
		{
			return this->m_ft.get_val(dim_i);
		}
		int assess_m_ft()const
		{
			return this->m_ft.assess_ft();
		}


		int get_grad_mean()const;
		int get_area()const;
		Point get_gp()const;
		vector<Point> get_contour()const;

		static bool same_pos(const hole& h1,const hole& h2);

		void merge_spe(const ats_frame& frame,const hole& h);
		int get_index()const;

	private:
		int index;
		static int index_generator;
		int generate_index();
		Point gp;
		vector<Point> contour;
		int grad_mean;
		int area;
	
		static int dis_sq(const Point& p1,const Point& p2);
		void calc_gp();
		void push_back(const ats_frame& frame,const Point& p);
		void update_area();
	};
	
	class ats_svm
	{
	public:
		

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
			classifier->setKernel(ml::SVM::KernelTypes::RBF);
			classifier->setGamma(20);  // for poly/rbf/sigmoid
			classifier->setC(7);       // for CV_classifier_C_SVC, CV_classifier_EPS_SVR and CV_classifier_NU_SVR
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

	
}