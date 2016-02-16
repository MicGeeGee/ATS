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
  
  svm->save("C:\\Users\\Administrator\\Desktop\\img\\svm_classifier.xml");
  

  return predicted;
}



int main()
{
	
	ats::ats_svm::load("svm_classifier.xml");

	//ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\reEidted.bmp");
	ats::ats_frame frame("E:\\OPENCV_WORKSPACE\\Image_DataSet\\1\\Img_1.jpg");

	frame.detect_holes();
	frame.show();
	frame.save("C:\\Users\\Administrator\\Desktop\\img\\img.jpg");
	frame.save_g("C:\\Users\\Administrator\\Desktop\\img\\img_g.jpg");
	/*
	ats::ats_svm::train<int>(ats::ats_frame::training_data,ats::ats_frame::labels);
	cout<<ats::ats_svm::predict<int>(ats::ats_frame::test_data);
	ats::ats_svm::save("svm_classifier.xml");
	*/


	waitKey();
	return 0;
}