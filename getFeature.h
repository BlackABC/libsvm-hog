#include <iostream>
#include <fstream>
#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "fhog.h"

void makeFeatureFile(const std::string& fileName, const std::string& positiveName, const std::string& negativeName);