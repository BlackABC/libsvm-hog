#include <opencv2\core\core.hpp>
#include "fhog.h"
#include "getFeature.h"
#include "ClassficationSVM.h"

//int main()
//{
//	//���������ļ��ã�Ҫ��ѵ��֮ǰִ��
//	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_train.txt", "PositiveFeatures.txt", "NegativeFeatures.txt");
//	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_test.txt", "PositiveFeatures_test.txt", "NegativeFeatures_test.txt");
//	ClassficationSVM cfSVM;
//
//	cfSVM.train("car_svm");
//	cfSVM.predict("PositiveFeatures_test.txt", "car_svm");
//	cfSVM.predict("NegativeFeatures_test.txt", "car_svm");
//
//	//׼ȷ��
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
	//���������ļ��ã�Ҫ��ѵ��֮ǰִ��
	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_train.txt", "PositiveFeatures.txt", "NegativeFeatures.txt");
	//makeFeatureFile("C:/Users/Administrator/Desktop/VOC2007/ImageSets/Main/car_test.txt", "PositiveFeatures_test.txt", "NegativeFeatures_test.txt");
	ClassficationSVM cfSVM;

	cfSVM.train("people_svm");
	cfSVM.predict("PositiveFeatures_test.txt", "people_svm");
	cfSVM.predict("NegativeFeatures_test.txt", "people_svm");

	//׼ȷ��
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