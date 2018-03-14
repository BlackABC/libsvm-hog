#include <opencv2\core\core.hpp>
#include "fhog.h"
#include "getFeature.h"
#include "ClassficationSVM.h"

//int main()
//{
//	//创建特征文件用，要在训练之前执行
//	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_train.txt", "PositiveFeatures.txt", "NegativeFeatures.txt");
//	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_test.txt", "PositiveFeatures_test.txt", "NegativeFeatures_test.txt");
//	ClassficationSVM cfSVM;
//
//	cfSVM.train("car_svm");
//	cfSVM.predict("PositiveFeatures_test.txt", "car_svm");
//	cfSVM.predict("NegativeFeatures_test.txt", "car_svm");
//
//	//准确率
//	int correctNum = 0;
//	int totalNum = cfSVM.judgeRight.size();
//	for (int i = 0; i < totalNum; i++)
//	{
//		if (cfSVM.judgeRight[i] == true)
//		{
//			correctNum++;
//		}
//	}
//	double precent = 1.0*correctNum / totalNum;
//	std::cout << precent << std::endl;
//
//	system("pause");
//
//	return 0;
//}

int main()
{
	//创建特征文件用，要在训练之前执行
	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_train.txt", "PositiveFeatures.txt", "NegativeFeatures.txt");
	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_test.txt", "PositiveFeatures_test.txt", "NegativeFeatures_test.txt");
	ClassficationSVM cfSVM;

	cfSVM.train("people_svm");
	cfSVM.predict("PositiveFeatures_test.txt", "people_svm");
	cfSVM.predict("NegativeFeatures_test.txt", "people_svm");

	//准确率
	int correctNum = 0;
	int totalNum = cfSVM.judgeRight.size();
	for (int i = 0; i < totalNum; i++)
	{
		if (cfSVM.judgeRight[i] == true)
		{
			correctNum++;
		}
	}
	double precent = 1.0*correctNum / totalNum;
	std::cout << precent << std::endl;

	system("pause");

	return 0;
}