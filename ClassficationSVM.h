#include <iostream>
#include "svm.h"
#include <list>
#include <vector>

#define FEATUREDIM 5239

class ClassficationSVM{
public:
	ClassficationSVM();
	~ClassficationSVM();
	//—µ¡∑
	void train(const std::string& modelFileName);
	//∑÷¿‡‘§≤‚ºÏ≤‚
	void predict(const std::string& featureFileName, const std::string& modelFileName);
public:
	std::vector<bool> judgeRight;
private:
	void setParams();
	void readTrainData(const std::string& featureFileName);
private:
	svm_parameter param;
	svm_problem prob;//all the data for train
	std::list<svm_node*> dataList;//list of features of all the samples
	std::list<double> typeList;//list of type of all the samples
	int sampleNum;
};