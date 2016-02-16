#include "ats.h"

namespace ats
{
	
	list<Mat> ats_svm::training_data=list<Mat>();
	list<int> ats_svm::labels=list<int>();
	list<Mat> ats_svm::test_data=list<Mat>();


	ats_frame::ats_frame(const Mat& RGB_img):Mat(convert_to_gray_img(RGB_img))
	{
		this->origin_img=RGB_img;
		calc_gradient(*this,grad_val,grad_x,grad_y);
		calc_mid_brgtnss();
	}
	ats_frame::ats_frame(const string& img_path):Mat(imread(img_path,0))
	{
		origin_img=imread(img_path);
		calc_gradient(*this,grad_val,grad_x,grad_y);
		calc_mid_brgtnss();
	}

	ats_frame::ats_frame(const ats_frame& frame): Mat(frame)
	{
		origin_img=frame.origin_img;
		grad_val=frame.grad_val;
		grad_x=frame.grad_x;
		grad_y=frame.grad_y;
	}
	int ats_frame::get_grad(int x,int y)const
	{
		return grad_val.at<short>(y,x);
	}
	int ats_frame::get_grad(const Point& pos)const
	{
		return get_grad(pos.x,pos.y);
	}
	int ats_frame::get_phase(int x,int y)const
	{
		return atan((grad_y.at<short>(y,x)+0.0)/grad_x.at<short>(y,x))*180/3.14;
	}
	int ats_frame::get_phase(const Point& pos)const
	{
		return get_phase(pos.x,pos.y);
	}

	int ats_frame::get_brgtnss(int x,int y)const
	{
		return this->at<uchar>(y,x);
	}
	int ats_frame::get_brgtnss(const Point& pos)const
	{
		return get_brgtnss(pos.x,pos.y);
	}

	int ats_frame::get_grad_x(int x,int y)const
	{
		return grad_x.at<short>(Point(x,y));
	}
	int ats_frame::get_grad_y(int x,int y)const
	{
		return grad_y.at<short>(Point(x,y));
	}
	int ats_frame::get_grad_x(const Point& p)const
	{
		return grad_x.at<short>(p);
	}
	int ats_frame::get_grad_y(const Point& p)const
	{
		return grad_y.at<short>(p);
	}


	void ats_frame::resize_img(Mat& img,Mat& dst,double scale)
	{
	
		Size refS = Size((int) img.cols*scale,
			(int) img.rows*scale);
		cv::resize(img,dst,refS);
	}
	
	void ats_frame::print_num(Mat& img,int num)
	{
		char num_char_array[10];
		sprintf(num_char_array,"%d",num);
		string num_str= num_char_array;
		putText( img, num_str, Point( img.rows/2,img.cols/4),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
	}
	void ats_frame::print_num(Mat& img,const Point& p,int num)
	{
		char num_char_array[10];
		sprintf(num_char_array,"%d",num);
		string num_str= num_char_array;
		putText( img, num_str,p,CV_FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0) );
	}
	Mat ats_frame::convert_to_gray_img(const Mat& img)
	{
		Mat dst;
		cvtColor(img,dst,CV_BGR2GRAY);
		return dst;
	}

	int ats_frame::find_contours(const Mat& binary_img,vector<vector<Point> >& contour_container)
	{
		vector<Vec4i> hierarchy;
		findContours(binary_img,contour_container,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0, 0));
		vector<vector<Point> >::iterator it;
		return contour_container.size();
	}
	
	bool ats_frame::bounding_rect_filter(Rect& a,Point gp,float threRatio)
	{
		int w=a.width+a.width+a.width;
		int h=a.height+a.height+a.height;
		int dx=a.x+w/2-gp.x;	
		int dy=a.y+h/2-gp.y;
		//Rect iterRect(a.x-dx,a.y-dy,w,h);
	
	


		vector<unsigned char> grayScale;
		grayScale.push_back(this->mat_at(gp));
		grayScale.push_back(this->mat_at(Point(gp.x-1,gp.y)));
		grayScale.push_back(this->mat_at(Point(gp.x,gp.y-1)));
		grayScale.push_back(this->mat_at(Point(gp.x+1,gp.y)));
		grayScale.push_back(this->mat_at(Point(gp.x,gp.y+1)));
		sort(grayScale.begin(),grayScale.end());
		unsigned char grayScale_gp=grayScale[grayScale.size()/2];

		grayScale.clear();

		int i=0;
		int j=0;

		Point lp(a.x-a.width,a.y-a.height);


		for(;i<w||j<h;i++,j++)
		{
			if(i<w)
			{ 
				grayScale.push_back(this->mat_at(Point(lp.x+i,lp.y)));
				grayScale.push_back(this->mat_at(Point(lp.x+i,lp.y+h)));
			}
			if(j<h)
			{
				grayScale.push_back(this->mat_at(Point(lp.x+w,lp.y+j)));
				grayScale.push_back(this->mat_at(Point(lp.x,lp.y+j)));
			}
		}
	

		sort(grayScale.begin(),grayScale.end());

	
		bool rectD1=grayScale[grayScale.size()/2]*threRatio>grayScale_gp;
		



		return rectD1;


	}

	void ats_frame::calc_gradient(const Mat& src,Mat& dst,Mat& dst_x,Mat& dst_y)
	{
		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;
		Sobel( src, dst_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		
		/// 求Y方向梯度
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel( src, dst_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		

		/// 合并梯度(近似)
		addWeighted( dst_x, 0.5, dst_y, 0.5, 0, dst);
	}
	void ats_frame::calc_mid_brgtnss()
	{
		vector<uchar> brgtnss_arr;
		for(int i=0;i<this->rows;i+=100)
			for(int j=0;j<this->cols;j+=100)
				brgtnss_arr.push_back(this->at<uchar>(i,j));

		std::sort(brgtnss_arr.begin(),brgtnss_arr.end());
		mid_brgtnss=brgtnss_arr[brgtnss_arr.size()/2];
	}
	void ats_frame::label_pixel(const Point& p,const Scalar& color,int r)
	{
		circle(origin_img,p,r,color,-1);
	}


	void ats_frame::detect_holes(int thre_min)
	{
		//GaussianBlur(*this,*this,Size(5,5),0,0);
		this->morphology_filter(*this,*this,0,2,2);
		
		for(int i=thre_min;i<=mid_brgtnss;i+=10)
		{
			Mat thre_img;
			vector<vector<Point> > contour_container;
			cv::threshold(*this,thre_img,i,255,THRESH_BINARY);
			find_contours(thre_img,contour_container);
				
			for(int j=0;j<contour_container.size();j++)
			{

				

				hole h(*this,contour_container[j]);

				int area_lb=1;
				int area_ub=(this->cols/30)*(this->rows/30);

				//area filter
				if(h.get_area()>area_ub||h.get_area()<area_lb)
					continue;
				//bounding rectangle filter
				if(!bounding_rect_filter(boundingRect(contour_container[j]),h.get_gp()))
					continue;
				


				//std::cout<<h.get_index()<<" : "<<h.m_ft<<" , "<<h.assess_m_ft();
				//std::cout<<endl;

				//if(h.assess_m_ft()>300)
					//continue;
				
				
				
				int k=0;
				std::list<hole>::iterator it;
				bool is_new=true;
				for(it=hole_set.begin();it!=hole_set.end();it++)
				{
					k++;
					if(hole::same_pos(h,*it))
					{	
						if(i==20&&j==4&&k==2)
							cout<<"aaaa"<<endl;

						it->merge_spe(h);
						is_new=false;
					}
				}
				if(is_new)
				{
					hole_set.push_back(h);
				}
					
			}
		}
		
		
		//return;
		
		//manual feature filter
		
		std::list<hole>::iterator it;
		int k=0;
		for(it=hole_set.begin();it!=hole_set.end();k++)
		{
			cout<<it->get_index()<<": "<<it->get_m_ft()<<", "<<it->assess_m_ft();
				cout<<endl;
			

			//if(it->assess_m_ft()>200)
			if(ats::ats_svm::predict<float>(it->get_m_ft())==-1)
			{		
				/*
				if(k<hole_set.size()/2)
				{
					training_data.push_back(it->get_m_ft());
					labels.push_back(-1);
				}
				else
					test_data.push_back(it->get_m_ft());
			*/
					
			
				it=hole_set.erase(it);
			}
			else
			{
			
				
				it++;
			}
		}

/*		
		for(it=hole_set.begin();it!=hole_set.end();it++)
		{
			training_data.push_back(it->get_m_ft());
			labels.push_back(1);
		}
		
		*/
		std::cout<<"end"<<endl;
	}
	void ats_frame::show(double zoom_scale)
	{
		Mat dst;

		list<hole>::iterator it;
		for(it=hole_set.begin();it!=hole_set.end();it++)
			label_hole(*it,Scalar(0, 255, 0),it->get_index());
			

		print_num(origin_img,hole_set.size());
		resize_img(origin_img,dst,zoom_scale);
		imshow("frame",dst);
		resize_img(*this,dst,zoom_scale);
		imshow("frame_g",dst);
		
	}

	void ats_frame::save(const string& tar_path)
	{
		imwrite(tar_path,this->origin_img);
	}
	void ats_frame::save_g(const string& tar_path)
	{
		imwrite(tar_path,*this);
	}
	void ats_frame::label_hole(const hole& h,const Scalar& color)
	{
		vector<Point> contour=h.get_contour();
		for(int i=0;i<contour.size();i++)
			label_pixel(contour[i],color,0);
		label_pixel(h.get_gp(),color,2);
	}

	void ats_frame::label_hole(const hole& h,const Scalar& color,int num)
	{
		vector<Point> contour=h.get_contour();
		for(int i=0;i<contour.size();i++)
			label_pixel(contour[i],color,0);
		
		print_num(this->origin_img,h.get_gp(),num);

	}

	uchar ats_frame::mat_at(const Point& p)const
	{
		if(p.x>=this->cols-1 || p.y>=this->rows-1||p.x<0||p.y<0)
			return 255;
		else
			return this->at<uchar>(p);
	}

	void ats_frame::morphology_filter(Mat& img,Mat& dst,int morph_operator,int morph_elem,int morph_size)
	{
  
		int operation = morph_operator + 2;

		Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

		/// 运行指定形态学操作
		morphologyEx( img, dst, operation, element );

		element.release();
  
	}

	int hole::index_generator=0;

	hole::manual_ft::manual_ft(const ats_frame&  frame,const vector<Point>& contour):Mat(1,3,CV_32F)
	{
	
		this->is_loaded=false;
		
	}
	hole::manual_ft::manual_ft(const manual_ft& ft):Mat(ft)
	{
		
		this->is_loaded=ft.is_loaded;
		
	}
	void hole::manual_ft::set_val(int dim_i,float val)
	{
		this->at<float>(0,dim_i-1)=val;
	}
	void hole::manual_ft::calc_ft(const ats_frame&  frame,const vector<Point>& contour)
	{
		
		//[body_brgtnss_mid/(con_brgtnss_mid+1),poly_norm,body_brgtnss_mid/(bg_brgtnss_mid+1)]

		this->is_loaded=true;

		vector<int> contour_brgtnss_arr;
		vector<int> body_brghtnss_arr;
		vector<int> bg_brghtnss_arr;
		
		f_point poly_param(0,0);//length 8

		Rect rect =boundingRect(contour);
		

		for(int i=rect.width/4;i<rect.width*3/4;i++)
			for(int j=rect.height/4;j<rect.height*3/4;j++)
				body_brghtnss_arr.push_back(frame.get_brgtnss(rect.x+i,rect.y+j));
				
		
		for(int i=0;i<contour.size();i++)
		{
			contour_brgtnss_arr.push_back(frame.get_brgtnss(contour[i]));	
			poly_param+=this->normalize_vector(frame.get_grad_x(contour[i]),frame.get_grad_y(contour[i]));
		}
		
		

		Point lu_p(rect.x-rect.width,rect.y-rect.height);
		for(int i=0;i<rect.width*3;i++)
			bg_brghtnss_arr.push_back(frame.mat_at(Point(lu_p.x+i,lu_p.y)));
		for(int j=0;j<rect.height*3;j++)
			bg_brghtnss_arr.push_back(frame.mat_at(Point(lu_p.x,lu_p.y+j)));
		for(int i=0;i<rect.width*3;i++)
			bg_brghtnss_arr.push_back(frame.mat_at(Point(lu_p.x+i,lu_p.y+rect.height*3)));
		for(int j=0;j<rect.height*3;j++)
			bg_brghtnss_arr.push_back(frame.mat_at(Point(lu_p.x+rect.width*3,lu_p.y+j)));
		

		
				
		std::sort(contour_brgtnss_arr.begin(),contour_brgtnss_arr.end());
		std::sort(body_brghtnss_arr.begin(),body_brghtnss_arr.end());
		std::sort(bg_brghtnss_arr.begin(),bg_brghtnss_arr.end());

		set_val(1,body_brghtnss_arr[body_brghtnss_arr.size()/2]/(contour_brgtnss_arr[contour_brgtnss_arr.size()/2]+1.0));
		//set_val(1,(contour_brgtnss_arr[contour_brgtnss_arr.size()/2]+0.0)/(body_brghtnss_arr[body_brghtnss_arr.size()/2]+1));
		f_point normalized_vec(poly_param.x/contour.size(),poly_param.y/contour.size());
		
		set_val(2,sqrt(normalized_vec.x*normalized_vec.x+normalized_vec.y*normalized_vec.y));

		set_val(3,body_brghtnss_arr[body_brghtnss_arr.size()/2]/(bg_brghtnss_arr[bg_brghtnss_arr.size()/2]+1.0));

	}
	float hole::manual_ft::get_val(const ats_frame&  frame,const vector<Point>& contour,int dim_i)
	{
		if(is_loaded)
			return this->at<float>(0,dim_i-1);
		else
		{
			calc_ft(frame,contour);
			return this->at<float>(0,dim_i-1);
		}
	}

	hole::manual_ft::f_point::f_point(float x,float y)
	{
		this->x=x;
		this->y=y;
	}
	hole::manual_ft::f_point& hole::manual_ft::f_point::operator+=(const f_point& p)
	{
		this->x+=p.x;
		this->y+=p.y;
		return *this;
	}
	hole::manual_ft::f_point hole::manual_ft::normalize_vector(float x,float y)
	{
		float r=sqrt(x*x+y*y);
		if(r)
			return f_point(x/r,y/r);
		else
			return f_point(0,0);
	}

	int hole::get_index()const
	{
		return index;
	}
	int hole::generate_index()
	{
		return index_generator++;
	}

	int hole::get_grad_mean()const
	{
		return grad_mean;
	}
	int hole::get_area()const
	{
		return area;
	}
	Point hole::get_gp()const
	{
		return gp;
	}
	vector<Point> hole::get_contour()const
	{
		return contour;
	}

	bool hole::same_pos(const hole& h1,const hole& h2)
	{
		//return dis_sq(h1.gp,h2.gp)<(h1.area+h2.area)/2;
		return dis_sq(h1.gp,h2.gp)<(h1.area<h2.area?h2.area:h1.area);
	}

	hole::hole(const ats_frame& frame,const vector<Point>& contour):m_ft(frame,contour)
	{
		this->p_frame=&frame;
		this->contour=contour;
		calc_gp();
				
		grad_mean=0;
		for(int i=0;i<contour.size();i++)
			grad_mean+=frame.at<uchar>(contour[i]);
		grad_mean/=contour.size();
		area=contourArea(contour);
		index=generate_index();
	} 
	hole::hole(const hole& h):m_ft(h.m_ft)
	{
		

		gp=h.gp;
		contour=h.contour;
		grad_mean=h.grad_mean;
		area=h.area;
		index=h.index;

		p_frame=h.p_frame;
	}
	

	void hole::merge_spe(hole& h)
	{
		if(h.area>this->area)
		{
			if(h.assess_m_ft()<this->assess_m_ft())
				*this=hole(h);
			return;
		}

		for(int i=0;i<h.contour.size();i++)
			if(p_frame->get_grad(h.contour[i])>this->grad_mean)
				push_back(h.contour[i]);
		update_area();
	}

	void hole::push_back(const Point& p)
	{
		Point pos;
		pos.x=gp.x*contour.size();
		pos.y=gp.y*contour.size();

		pos+=p;

		gp.x=pos.x/(contour.size()+1);
		gp.y=pos.y/(contour.size()+1);
		
		grad_mean*=contour.size();
		grad_mean+=p_frame->get_grad(p);
		grad_mean/=contour.size()+1;

		contour.push_back(p);
	}
	
	int hole::dis_sq(const Point& p1,const Point& p2)
	{
		return (p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);
	}
	void hole::calc_gp()
	{
		gp=Point(0,0);
		for(int i=0;i<contour.size();i++)
			gp+=contour[i];
		gp.x/=contour.size();
		gp.y/=contour.size();
	}

	void hole::update_area()
	{
		area=contourArea(contour);
	}

	Ptr<cv::ml::SVM> ats_svm::classifier=ml::SVM::create();
	bool ats_svm::is_trained=false;
	Mat ats_svm::support_vectors=Mat();

}