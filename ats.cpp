#include "ats.h"

namespace ats
{
	list<hole> ats_frame::y_list=list<hole>();
	list<hole> ats_frame::n_list=list<hole>();
	list<hole> ats_frame::test_list=list<hole>();

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
				
			for(int i=0;i<contour_container.size();i++)
			{
				hole h(*this,contour_container[i]);

				int area_lb=1;
				int area_ub=(this->cols/30)*(this->rows/30);

				//area filter
				if(h.get_area()>area_ub||h.get_area()<area_lb)
					continue;
				//bounding rectangle filter
				if(!bounding_rect_filter(boundingRect(contour_container[i]),h.get_gp()))
					continue;
				

				
				//std::cout<<h.get_index()<<" : "<<h.m_ft<<" , "<<h.assess_m_ft();
				//std::cout<<endl;

				//if(h.assess_m_ft()>300)
					//continue;
				
				

				
				std::list<hole>::iterator it;
				bool is_new=true;
				for(it=hole_set.begin();it!=hole_set.end();it++)
				{
					if(hole::same_pos(h,*it))
					{	
						it->merge_spe(*this,h);
						is_new=false;
					}
				}
				if(is_new)
				{
					hole_set.push_back(h);
				}
					
			}
		}
		
		//manual feature filter
		
		std::list<hole>::iterator it;
		int k=0;
		for(it=hole_set.begin();it!=hole_set.end();k++)
		{
			//cout<<it->m_ft;
			//cout<<endl;

			if(it->assess_m_ft()>300)
			{		
//				if(it->assess_m_ft()>=600)
				if(k<hole_set.size()/2)
					n_list.push_back(*it);
				else
					test_list.push_back(*it);


				it=hole_set.erase(it);
			}
			else
				it++;
		}


		for(it=hole_set.begin();it!=hole_set.end();it++)
			y_list.push_back(*it);

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

	uchar ats_frame::mat_at(const Point& p)
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

	hole::manual_ft::manual_ft(const ats_frame& src_frame,const vector<Point>& src_contour):Mat(1,8,CV_32SC1)
	{
		this->calc_ft(src_frame,src_contour);
	}
	hole::manual_ft::manual_ft(const manual_ft& ft):Mat(ft)
	{
		
	}
	void hole::manual_ft::set_val(int dim_i,int val)
	{
		this->at<int>(0,dim_i-1)=val;
	}
	void hole::manual_ft::calc_ft(const ats_frame& src_frame,const vector<Point>& src_contour)
	{
		
		//[con_brgtnss_mid,body_brgtnss_mid,
		//con_grad_mid,con_brgtnss_mid/(body_brgtnss_mid+1),
		//con_s,body_s,
		//con_grad_s,poly_norm]

		this->is_loaded=true;

		vector<int> contour_brgtnss_arr;
		vector<int> body_brghtnss_arr;
		vector<int> contour_grad_arr;
		f_point poly_param(0,0);//length 8

		Rect rect =boundingRect(src_contour);
		int ave_brgtnss=0;
		int ave_body_brgtnss=0;
		for(int i=0;i<rect.width;i++)
			for(int j=0;j<rect.height;j++)
			{
				int val=src_frame.get_brgtnss(rect.x+i,rect.y+j);
				ave_brgtnss+=val;
				if(i>=rect.width/4&&i<=rect.width*3/4&&j>=rect.height/4&&j<=rect.height*3/4)
				{
					body_brghtnss_arr.push_back(val);
					ave_body_brgtnss+=val;
				}
			}
		ave_brgtnss/=rect.width*rect.height;
		ave_body_brgtnss/=rect.width*rect.height/4;


		int ave_contour_brgtnss=0;
		int ave_contour_grad=0;
		for(int i=0;i<src_contour.size();i++)
		{
			contour_brgtnss_arr.push_back(src_frame.get_brgtnss(src_contour[i]));	
			ave_contour_brgtnss+=contour_brgtnss_arr.back();

			contour_grad_arr.push_back(std::abs(src_frame.get_grad(src_contour[i])));
			ave_contour_grad+=contour_grad_arr.back();
			

			poly_param+=this->normalize_vector(src_frame.get_grad_x(src_contour[i]),src_frame.get_grad_y(src_contour[i]));
		}
		ave_contour_brgtnss/=src_contour.size();
		ave_contour_grad/=src_contour.size();

		int body_brgtnss_variance=0;
		for(int i=rect.width/4;i<rect.width*3/4;i++)
			for(int j=rect.height/4;j<rect.height*3/4;j++)
				body_brgtnss_variance+=(src_frame.get_brgtnss(rect.x+i,rect.y+j)-ave_body_brgtnss)*(src_frame.get_brgtnss(rect.x+i,rect.y+j)-ave_body_brgtnss);
		body_brgtnss_variance/=rect.width*rect.height/4;
		body_brgtnss_variance=sqrt(body_brgtnss_variance);

		
		int con_brgtnss_variance=0;
		int con_grad_variance=0;
		for(int i=0;i<src_contour.size();i++)
		{
			con_brgtnss_variance+=(src_frame.get_brgtnss(src_contour[i])-ave_contour_brgtnss)*(src_frame.get_brgtnss(src_contour[i])-ave_contour_brgtnss);
			con_grad_variance+=(src_frame.get_grad(src_contour[i])-ave_contour_grad)*(src_frame.get_grad(src_contour[i])-ave_contour_grad);
		}		
		con_brgtnss_variance/=src_contour.size();
		con_brgtnss_variance=sqrt(con_brgtnss_variance);
		con_grad_variance/=src_contour.size();
		con_grad_variance=sqrt(con_grad_variance);
				
		std::sort(contour_brgtnss_arr.begin(),contour_brgtnss_arr.end());
		std::sort(body_brghtnss_arr.begin(),body_brghtnss_arr.end());
		std::sort(contour_grad_arr.begin(),contour_grad_arr.end());

		set_val(1,contour_brgtnss_arr[contour_brgtnss_arr.size()/2]*1000/ave_brgtnss);
		set_val(2,body_brghtnss_arr[contour_brgtnss_arr.size()/2]*1000/ave_brgtnss);
		set_val(3,contour_grad_arr[contour_brgtnss_arr.size()/2]);
		set_val(4,contour_brgtnss_arr[contour_brgtnss_arr.size()/2]/(body_brghtnss_arr[contour_brgtnss_arr.size()/2]+1));

		set_val(5,con_brgtnss_variance);
		set_val(6,body_brgtnss_variance);
		

		set_val(7,con_grad_variance);
		set_val(8,(poly_param.x*poly_param.x+poly_param.y*poly_param.y)*10);

	}
	int hole::manual_ft::get_val(int dim_i)const
	{
		return this->at<int>(0,dim_i-1);
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
	hole::manual_ft::f_point hole::manual_ft::normalize_vector(int x,int y)
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
	}
	

	void hole::merge_spe(const ats_frame& frame,const hole& h)
	{
		if(h.area>this->area)
		{
			if(h.assess_m_ft()<this->assess_m_ft())
				*this=h;
			return;
		}

		for(int i=0;i<h.contour.size();i++)
			if(frame.get_grad(h.contour[i])>this->grad_mean)
				push_back(frame,h.contour[i]);
		update_area();
	}

	void hole::push_back(const ats_frame& frame,const Point& p)
	{
		Point pos;
		pos.x=gp.x*contour.size();
		pos.y=gp.y*contour.size();

		pos+=p;

		gp.x=pos.x/(contour.size()+1);
		gp.y=pos.y/(contour.size()+1);
		
		grad_mean*=contour.size();
		grad_mean+=frame.get_grad(p);
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

}