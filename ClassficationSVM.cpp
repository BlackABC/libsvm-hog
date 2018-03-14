#include "ClassficationSVM.h"
#include <stdio.h>
#include <sstream>
#include <vector>

ClassficationSVM::ClassficationSVM()
{
	setParams();
}

ClassficationSVM::~ClassficationSVM()
{

}

void ClassficationSVM::train(const std::string& modelFileName)
{
	std::cout << "reading positive features..." << std::endl;
	readTrainData("PositiveFeatures.txt");
	std::cout << "reading negative features..." << std::endl;
	readTrainData("NegativeFeatures.txt");
	std::cout << sampleNum << std::endl;

	prob.l = sampleNum;//number of training samples
	prob.x = new svm_node *[prob.l];//features of all the training samples
	prob.y = new double[prob.l];//type of all the training samples

	int index = 0;
	while (!dataList.empty())
	{
		prob.x[index] = dataList.front();
		prob.y[index] = typeList.front();
		dataList.pop_front();
		typeList.pop_front();
		index++;
	}

	std::cout << "start training" << std::endl;
	svm_model *svmModel = svm_train(&prob, &param);

	std::cout << "save model" << std::endl;
	svm_save_model(modelFileName.c_str(), svmModel);
	std::cout << "done!" << std::endl;
}

void ClassficationSVM::predict(const std::string& featureFileName, const std::string& modelFileName)
{
	svm_model *svmModel = svm_load_model(modelFileName.c_str());
	FILE *fp = fopen(featureFileName.c_str(), "rt");

	fseek(fp, 0l, SEEK_END);
	long end = ftell(fp);
	fseek(fp, 0l, SEEK_SET);
	long start = ftell(fp);

	//电脑不给力，读取测试数据100个
	int testNum = 0;

	while (start != end)
	{
		svm_node *input = new svm_node[FEATUREDIM + 1];
		for (int k = 0; k < FEATUREDIM; k++)
		{
			float value = 0;
			fscanf(fp, "%f", &value);
			input[k].index = k + 1;
			input[k].value = value;
		}

		input[FEATUREDIM].index = -1;
		int predictValue = svm_predict(svmModel, input);
		if (featureFileName == "PositiveFeatures_test.txt")
		{
			if (predictValue == 0)
			{
				judgeRight.push_back(false);
			}
			else
			{
				judgeRight.push_back(true);
			}
		}
		else if (featureFileName == "NegativeFeatures_test.txt")
		{
			if (predictValue == 0)
			{
				judgeRight.push_back(true);
			}
			else
			{
				judgeRight.push_back(false);
			}
		}
		start = ftell(fp);

		testNum++;

		if (testNum == 1000)
			break;
	}
	fclose(fp);

	////准确率
	//int correctNum = 0;
	//int totalNum = judgeRight.size();
	//for (int i = 0; i < totalNum; i++)
	//{
	//	if (judgeRight[i] == true)
	//	{
	//		correctNum++;
	//	}
	//}
	//double precent = 1.0*correctNum / totalNum;
	//std::cout << precent << std::endl;
}

void ClassficationSVM::setParams()
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.01;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 200;
	param.C = 500;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.nr_weight = 0;
	param.weight = NULL;
	param.weight_label = NULL;
}

void ClassficationSVM::readTrainData(const std::string& featureFileName)
{
	FILE *fp = fopen(featureFileName.c_str(), "r");

	fseek(fp, 0l, SEEK_END);
	long end = ftell(fp);
	fseek(fp, 0l, SEEK_SET);
	long start = ftell(fp);

	//电脑不给力，控制样本数
	int sampNum = 0;

	while (start != end)
	{
		svm_node* features = new svm_node[FEATUREDIM + 1];
		for (int k = 0; k < FEATUREDIM; k++)
		{
			float value = 0;
			fscanf(fp, "%f", &value);
			features[k].index = k + 1;
			features[k].value = value;
		}
		features[FEATUREDIM].index = -1;

		//negative sample type is 0
		int type = 0;
		//positive sample type is 1
		if (featureFileName == "PositiveFeatures.txt")
			type = 1;

		dataList.push_back(features);
		typeList.push_back(type);
		sampleNum++;
		start = ftell(fp);

		sampNum++;

		if (sampNum == 200 && featureFileName == "PositiveFeatures.txt")
			break;

		if (sampNum == 800 && featureFileName == "NegativeFeatures.txt")
			break;
	}
	fclose(fp);
}