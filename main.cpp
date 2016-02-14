#include <cstdlib>
#include <cstdio>
#include <iostream>


#include "ats.h"

using namespace std;
using namespace cv;


Mat svm(cv::Mat& trainingData, cv::Mat& trainingClasses,cv::Mat& testData)
{

  Mat traning_label(trainingClasses.rows, 1, CV_32SC1);
  for (int i = 0; i < trainingClasses.rows; i++)
    traning_label.at<int>(i, 0) = trainingClasses.at<float>(i, 0);
  

  cv::Ptr<cv::ml::SVM> svm = ml::SVM::create();
  svm->setType(ml::SVM::Types::C_SVC);
  svm->setKernel(ml::SVM::KernelTypes::RBF);
  //svm->setDegree(0);  // for poly
  svm->setGamma(20);  // for poly/rbf/sigmoid
  //svm->setCoef0(0);   // for poly/sigmoid
  svm->setC(7);       // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
  //svm->setNu(0);      // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
  //svm->setP(0);       // for CV_SVM_EPS_SVR

  svm->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 1000, 1E-6));

  svm->train(trainingData, ml::SampleTypes::ROW_SAMPLE, traning_label);

  cv::Mat predicted(testData.rows, 1, CV_32F);
  svm->predict(testData, predicted);
  
  
  
  Mat support_vectors = svm->getSupportVectors();
  return predicted;
}



int main()
{
	
	//ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\reEidted.bmp");
	ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_1.jpg");

	frame.detect_holes();
	frame.show();
	frame.save("C:\\Users\\Administrator\\Desktop\\img\\img.jpg");
	frame.save_g("C:\\Users\\Administrator\\Desktop\\img\\img_g.jpg");
	
	
	int y_size=ats::ats_frame::y_list.size();
	int n_size=ats::ats_frame::n_list.size();
	Mat training_data(y_size+n_size,8,CV_32FC1);


	list<ats::hole>::iterator it;
	int k=0;
	for(it=ats::ats_frame::y_list.begin();it!=ats::ats_frame::y_list.end();it++,k++)
		for(int i=0;i<8;i++)
			training_data.at<float>(k,i)=it->get_m_ft(i+1);
	for(it=ats::ats_frame::n_list.begin();it!=ats::ats_frame::n_list.end();it++,k++)
		for(int i=0;i<8;i++)
			training_data.at<float>(k,i)=it->get_m_ft(i+1);
	
	Mat labels(y_size+n_size, 2, CV_32FC1);
	for(int i=0;i<y_size;i++)
		labels.at<float>(i,0)=1;
	for(int i=y_size;i<y_size+n_size;i++)
		labels.at<float>(i,0)=-1;

	Mat test_data(ats::ats_frame::test_list.size(),8,CV_32FC1);
	k=0;
	for(it=ats::ats_frame::test_list.begin();it!=ats::ats_frame::test_list.end();it++,k++)
		for(int i=0;i<8;i++)
			test_data.at<float>(k,i)=it->get_m_ft(i+1);


	cout<<svm(training_data,labels,test_data)<<endl;


	waitKey();
	return 0;
}