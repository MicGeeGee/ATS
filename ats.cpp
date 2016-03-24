#include "ats.h"

namespace ats
{
	
	list<Mat> ats_svm::training_data=list<Mat>();
	list<int> ats_svm::labels=list<int>();
	list<Mat> ats_svm::test_data=list<Mat>();


	int ats_frame::index_generator;

	ats_frame::ats_frame(const Mat& RGB_img):Mat(convert_to_gray_img(RGB_img))
	{
		index=ats_frame::generate_index();

		is_labeled=false;
		overlapping_num=0;

		this->origin_img=RGB_img;
		calc_gradient(*this,grad_val,grad_x,grad_y);
		calc_mid_brgtnss();
		is_holes_detected=false;

	}
	ats_frame::ats_frame(const string& img_path):Mat(imread(img_path,0))
	{
		index=ats_frame::generate_index();

		is_labeled=false;
		overlapping_num=0;

		origin_img=imread(img_path);
		calc_gradient(*this,grad_val,grad_x,grad_y);
		calc_mid_brgtnss();
		is_holes_detected=false;
	}
	ats_frame::ats_frame():Mat()
	{
		is_labeled=false;
		index=ats_frame::generate_index();
		overlapping_num=0;
		is_holes_detected=false;
	}


	ats_frame::ats_frame(const ats_frame& frame): Mat(frame)
	{

		overlapping_num=frame.overlapping_num;

		overlapping_list=frame.overlapping_list;
		is_labeled=frame.is_labeled;
		max_dis=frame.max_dis;
		dis_mat=frame.dis_mat;
		index_map=frame.index_map;
		ptr_map=frame.ptr_map;

		index=frame.index;

		mid_brgtnss=frame.mid_brgtnss;

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
	
	void ats_frame::print_num(Mat& img,int num1,int num2)
	{
		char num_char_array[10];
		sprintf(num_char_array,"%d+%d",num1,num2);
		string num_str= num_char_array;
		putText( img, num_str, Point( img.rows/2,img.cols/4),CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 0) );
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

	int ats_frame::get_hole_num()const
	{
		return this->hole_set.size();
	}

	void ats_frame::enroll_overlapping(int index)
	{
		if(overlapping_list.find(index)==overlapping_list.end())
			overlapping_list.insert(index);

		overlapping_num++;
		ptr_map[index]->enroll_overlapping();
			
	}

	void ats_frame::set_overlapping_num(int n)
	{
		overlapping_num=n;
	}

	void ats_frame::update_hole(int index,const hole& h)
	{
		if(overlapping_list.find(index)==overlapping_list.end()&&h.get_overlapping_num()>0)
			overlapping_list.insert(index);
		int incre_num=(h.get_overlapping_num()-ptr_map[index]->get_overlapping_num());
		overlapping_num+=incre_num;
		ptr_map[index]->update(h);
			
	}


	void ats_frame::detect_holes(int thre_min)
	{

		hole::index_generator_clear();//for different frames

		GaussianBlur(*this,*this,Size(3,3),0,0);
		//this->morphology_filter(*this,*this,0,2,3);
		
		for(int i=thre_min;i<=((mid_brgtnss+60)>255?255:(mid_brgtnss+30));i+=30)
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
				//if(!bounding_rect_filter(boundingRect(contour_container[j]),h.get_gp()))
					//continue;

				//cout<<h.get_index()<<": "<<h.get_m_ft()<<", "<<h.assess_m_ft();
				//cout<<endl;
				//if(h.assess_m_ft()>400)
					//continue;
				//if(h.get_m_ft(3)>0.2)
					//continue;
				if(ats::ats_svm::predict<float>(h.get_m_ft())==-1)
					continue;
				
				


				//std::cout<<h.get_index()<<" : "<<h.m_ft<<" , "<<h.assess_m_ft();
				//std::cout<<endl;

				//if(h.assess_m_ft()>200)
					//continue;
				
				
			
				list<hole>::iterator it;
				vector<list<hole>::iterator> hole_container;
				bool is_new=true;
			
				//get all the holes at the same pos of the new detected hole
				for(it=hole_set.begin();it!=hole_set.end();it++)
				{
					if(hole::same_pos(h,*it))
					{	
						hole_container.push_back(it);
						is_new=false;
					}
				}
				//merge the holes together
				if(hole_container.size()>1)
				{
					for(int i=1;i<hole_container.size();i++)
						merge_holes(hole_container[0],hole_container[i]);
				}
				if(hole_container.size()>0)
					hole_container[0]->merge_spe(h);

				if(is_new)
				{
					hole_set.push_back(h);
				}


				
			}
		}
		
		this->is_holes_detected=true;
		this->calc_max_dis();


		/*
		list<hole>::iterator it;
		for(it=hole_set.begin();it!=hole_set.end();it++)
		{
			cout<<it->get_index()<<": "<<it->get_m_ft(4)<<endl;
		}
		*/

		this->reorganize_frags();

		cout<<"The end."<<endl;
		
		
		
	}
	void ats_frame::show(double zoom_scale)
	{
		Mat dst;

		label_holes();

		resize_img(origin_img,dst,zoom_scale);
		char label[20];
		sprintf(label,"frame_%d",this->index);
		
		imshow(label,dst);


		//resize_img(*this,dst,zoom_scale);
		//imshow(label_gray,dst);
		
	}

	void ats_frame::save(const string& tar_path)
	{
		label_holes();
		imwrite(tar_path,this->origin_img);
	}
	void ats_frame::save_g(const string& tar_path)
	{
		imwrite(tar_path,*this);
	}

	void ats_frame::save_hole_set(const string& tar_path)
	{
		FILE* fp;
		fp=fopen(tar_path.c_str(),"w");
		list<hole>::iterator it;
		for(it=hole_set.begin();it!=hole_set.end();it++)
			fprintf(fp,"%d:%f %f %f %f label\n",it->get_index(),it->get_m_ft(1),it->get_m_ft(2),it->get_m_ft(3),it->get_m_ft(4));

		fclose(fp);

	}

	void ats_frame::label_hole(hole& h,const Scalar& color)
	{
		vector<Point> contour=h.get_contour();
		for(int i=0;i<contour.size();i++)
			label_pixel(contour[i],color,0);
		label_pixel(h.get_gp(),color,2);
	}

	void ats_frame::label_hole(hole& h,const Scalar& color,int num)
	{
		vector<Point> contour=h.get_contour();
		for(int i=0;i<contour.size();i++)
			label_pixel(contour[i],color,0);
		
		print_num(this->origin_img,h.get_gp(),num);

	}

	void ats_frame::label_holes()
	{
		if(is_labeled)
			return;
		else
		{
			list<hole>::iterator it;
			for(it=hole_set.begin();it!=hole_set.end();it++)
				label_hole(*it,Scalar(0, 255, 0),it->get_index());
			
			char label[20];
			char label_gray[20];
			sprintf(label,"frame_%d",this->index);
			sprintf(label_gray,"frame_g_%d",this->index);

			print_num(origin_img,hole_set.size(),this->overlapping_num);
			is_labeled=true;
		}
	}
		

	uchar ats_frame::mat_at(const Point& p)const
	{
		if(p.x>=this->cols-1 || p.y>=this->rows-1||p.x<0||p.y<0)
			return 255;
		else
			return this->at<uchar>(p);
	}

	list<hole>& ats_frame::get_hole_set()
	{
		return hole_set;
	}

	void ats_frame::reorganize_frags()
	{
		//list<hole> h_set=hole_set;

		list<hole>::iterator it1;
		list<hole>::iterator it2;



		for(it1=hole_set.begin();it1!=hole_set.end();it1++)
			for(it2=hole_set.begin();it2!=hole_set.end();)
			{
				/*
				if(frag_assessment(*it1,*it2)<1)
				{
					cout<<"("<<it1->get_index()<<","<<it2->get_index()<<"):"<<frag_assessment(*it1,*it2)<<endl;
				}
				*/

				if(it1->get_index()==12&&it2->get_index()==349)
					cout<<15355<<endl;

				if(it1->get_index()==it2->get_index())
				{
					it2++;
					continue;
				}
				
				if(!hole::same_pos(*it1,*it2))
					it2++;
				else
					it2=merge_holes(it1,it2);
				
				/*
				if(calc_dis_sq(it1->get_gp(),it2->get_gp())>(it1->get_area()>it2->get_area()?it1->get_area():it2->get_area()))
				{
					it2++;
					continue;
				}
				cout<<"("<<it1->get_index()<<","<<it2->get_index()<<"):"<<frag_assessment(*it1,*it2)<<endl;

				if(frag_assessment(*it1,*it2)<0.02)
					it2=merge_holes(it1,it2);
				else
					it2++;
				*/
			}
	}

	float ats_frame::frag_assessment(hole& h1,hole& h2)
	{
		list<hole>& hole_set=this->get_hole_set();
		list<hole>::iterator it;

		float theta_diff=0;
		float rho_diff=0;

		for(it=hole_set.begin();it!=hole_set.end();it++)
		{
			if(it->get_index()==h1.get_index()||it->get_index()==h2.get_index())
				continue;
			  
			theta_diff+=std::abs(hole::shape_ft::calc_theta(h1.get_gp(),it->get_gp())-hole::shape_ft::calc_theta(h2.get_gp(),it->get_gp()))/3.14;
			rho_diff+=std::abs(calc_dis(h1.get_gp(),it->get_gp())-calc_dis(h2.get_gp(),it->get_gp()))/(this->get_max_dis()+0.0);
			
		}

		return (theta_diff+rho_diff)/(hole_set.size()-2+0.1);
	}

	int ats_frame::get_dis(const hole& h1,const hole& h2)const
	{
		int index1=h1.get_index();
		int index2=h2.get_index();

		return dis_mat.at<int>(index_map.find(index1)->second,index_map.find(index2)->second);
	}
	int ats_frame::get_max_dis()const
	{
		return max_dis;
	}
	void ats_frame::calc_max_dis()
	{
		int dis_mat_i=0;
		dis_mat=Mat::zeros(hole_set.size(),hole_set.size(),CV_32S);

		max_dis=0;
		//vector<int> dis_arr;
		list<hole>::iterator it1;
		list<hole>::iterator it2;
		for(it1=hole_set.begin();it1!=hole_set.end();it1++)
		{
			ptr_map[it1->get_index()]=it1;
			for(it2=hole_set.begin();it2!=hole_set.end();it2++)
			{
				if(index_map.find(it1->get_index())==index_map.end())
					index_map[it1->get_index()]=dis_mat_i++;
				if(index_map.find(it2->get_index())==index_map.end())
					index_map[it2->get_index()]=dis_mat_i++;
					
				int dis=calc_dis(it1->get_gp(),it2->get_gp());
				if(dis>max_dis)
					max_dis=dis;

			//	dis_arr.push_back(calc_dis(it1->get_gp(),it2->get_gp()));
				//dis_mat.at<int>(index_map[it1->get_index()],index_map[it2->get_index()])=dis_arr.front();
				dis_mat.at<int>(index_map[it1->get_index()],index_map[it2->get_index()])=dis;
					
			}
		}
		//std::sort(dis_arr.begin(),dis_arr.end());

		//mid_dis=dis_arr[dis_arr.size()/2];	
	}

	int ats_frame::calc_dis(const Point& a,const Point& b)
	{
		return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
	}
	int ats_frame::calc_dis_sq(const Point& a,const Point& b)
	{
		return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
	}

	list<hole>::iterator ats_frame::merge_holes(list<hole>::iterator& p_host,list<hole>::iterator& p_guest)
	{
		//make *p_guest incorprated into *p_host,and then erase p_guest

		p_host->incorporate(*p_guest);
		return hole_set.erase(p_guest);
	}
	list<hole>::iterator ats_frame::get_hole(int index)
	{
		return ptr_map[index];
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

	hole::manual_ft::manual_ft(const ats_frame&  frame):Mat(1,4,CV_32F)
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
	void hole::manual_ft::calc_ft(const ats_frame&  frame,hole& h)
	{
		
		//[body_brgtnss_mid/con_brgtnss_mid,
		//poly_norm,
		//body_brgtnss_mid/bg_brgtnss_mid)
		//body_grad_mid/con_grad_mid,]

		this->is_loaded=true;

		vector<int> contour_brgtnss_arr;
		vector<int> bg_brghtnss_arr;

		
		
		f_point poly_param(0,0);//length 8

		Rect rect =boundingRect(h.get_contour());
		

				
		
		for(int i=0;i<h.get_contour().size();i++)
		{
			contour_brgtnss_arr.push_back(frame.get_brgtnss((h.get_contour())[i]));	
			poly_param+=this->normalize_vector(frame.get_grad_x((h.get_contour())[i]),frame.get_grad_y((h.get_contour())[i]));
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
		
		std::sort(bg_brghtnss_arr.begin(),bg_brghtnss_arr.end());

		

		set_val(1,h.get_body_brghtnss_mid()/(contour_brgtnss_arr[contour_brgtnss_arr.size()/2]+0.1));
		//set_val(1,(contour_brgtnss_arr[contour_brgtnss_arr.size()/2]+0.0)/(body_brghtnss_arr[body_brghtnss_arr.size()/2]+1));
		f_point normalized_vec(poly_param.x/h.get_contour().size(),poly_param.y/h.get_contour().size());
		
		set_val(2,sqrt(normalized_vec.x*normalized_vec.x+normalized_vec.y*normalized_vec.y));

		set_val(3,h.get_body_brghtnss_mid()/(bg_brghtnss_arr[bg_brghtnss_arr.size()/2]+0.1));
		
		set_val(4,abs((h.get_body_grad_mid()/(h.get_grad_mid()+0.1))));


	}
	float hole::manual_ft::get_val(const ats_frame&  frame,hole& h,int dim_i)
	{
		if(is_loaded)
			return this->at<float>(0,dim_i-1);
		else
		{
			calc_ft(frame,h);
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

	int hole::get_grad_mid()const
	{
		return grad_mid;
	}
	int hole::get_area()const
	{
		return area;
	}
	Point hole::get_gp()const
	{
		return gp;
	}
	vector<Point>& hole::get_contour()
	{
		return contour;
	}

	bool hole::same_pos(const hole& h1,const hole& h2)
	{
		//return dis_sq(h1.gp,h2.gp)<(h1.area+h2.area)*1.5;
		//return dis_sq(h1.gp,h2.gp)<(h1.area+h2.area)/2;
		//return dis_sq(h1.gp,h2.gp)<(h1.area<h2.area?h2.area:h1.area);
		return dis_sq(h1.gp,h2.gp)<(h2.area+h1.area)*0.8;

	}

	hole::hole(ats_frame& frame,const vector<Point>& contour):m_ft(frame),s_ft()
	{
		overlapping_num=0;

		this->p_frame=&frame;
		this->contour=contour;
		calc_gp();
		
		for(int i=0;i<contour.size();i++)
			con_grad_arr.push_back(frame.get_grad(contour[i]));
		std::sort(con_grad_arr.begin(),con_grad_arr.end());
		grad_mid=con_grad_arr[con_grad_arr.size()/2];
		
		area=contourArea(contour);
		index=generate_index();

		area_in_series=area;
		body_brghtnss_mid=-1;
		body_grad_mid=-1;

	} 
	hole::hole(const hole& h):m_ft(h.m_ft),s_ft(h.s_ft)
	{
		overlapping_num=h.overlapping_num;

		gp=h.gp;
		contour=h.contour;
		
		area=h.area;
		index=h.index;

		p_frame=h.p_frame;
		grad_mid=h.grad_mid;
		con_grad_arr=h.con_grad_arr;

		area_in_series=h.area_in_series;

		body_brghtnss_mid=h.body_brghtnss_mid;
		body_grad_mid=h.body_grad_mid;
	}
	

	void hole::merge_spe(hole& h)
	{
		if(h.area>this->area)
		{
			if(h.assess_m_ft()<=this->assess_m_ft())
				*this=hole(h);
			return;
		}
		for(int i=0;i<h.contour.size();i++)
			if(p_frame->get_grad(h.contour[i])>this->get_body_grad_mid()&&p_frame->get_brgtnss(h.get_gp())<this->get_body_brghtnss_mid())
				push_back(h.contour[i],i==(h.contour.size()-1));
	}

	void hole::push_back(const Point& p,bool is_last)
	{
		//if is_last is true, then update the params

		Point pos;
		pos.x=gp.x*contour.size();
		pos.y=gp.y*contour.size();

		pos+=p;

		gp.x=pos.x/(contour.size()+1);
		gp.y=pos.y/(contour.size()+1);
		

		con_grad_arr.push_back(p_frame->get_grad(p));
		
		contour.push_back(p);

		if(is_last)
		{
			area=contourArea(contour);
			std::sort(con_grad_arr.begin(),con_grad_arr.end());
			grad_mid=con_grad_arr[con_grad_arr.size()/2];
		}

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

	

	Ptr<cv::ml::SVM> ats_svm::classifier=ml::SVM::create();
	bool ats_svm::is_trained=false;
	Mat ats_svm::support_vectors=Mat();

	ats_frame* holes_matching::current_frame=NULL;
	ats_frame* holes_matching::last_frame=NULL;

	bool holes_matching::is_l_loaded=false;
	bool holes_matching::is_c_loaded=false;

	Mat holes_matching::cost_m;

	map<int,int> holes_matching::result_row;
	map<int,int> holes_matching::result_col;

	map<int,int> holes_matching::revindex_map_c;
	map<int,int> holes_matching::revindex_map_l;

	int holes_matching::row_res[100000];
	int holes_matching::col_res[100000];
	float holes_matching::total_cost;

	string holes_matching::file_path;
	int holes_matching::overlapping_num=0;


}