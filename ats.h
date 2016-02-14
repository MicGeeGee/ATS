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

		static list<hole> y_list;
		static list<hole> n_list;
		static list<hole> test_list;

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
	
	
}