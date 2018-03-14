#include "getFeature.h"

//positiveName正样本特征文件   PositiveFeatures.txt
//negativeName负样本特征文件   NegativeFeatures.txt
void makeFeatureFile(const std::string& fileName, const std::string& positiveName, const std::string& negativeName)
{
	std::fstream fp;
	fp.open(fileName);
	std::string str;

	std::fstream f1, f2;
	f1.open(positiveName, std::ios::app);
	f2.open(negativeName, std::ios::app);
	//处理逐个图片，先获取图片名
	while (std::getline(fp, str))
	{
		const char *c = str.c_str();
		//解析出图片名和label名
		char picName[7], labelName;
		strncpy(picName, c, 6);
		picName[6] = '\0';
		labelName = c[7];
		std::string strtmp = picName;
		//读入图像
		cv::Mat image = cv::imread("C:/Users/Administrator/Desktop/VOC2007/JPEGImages/" + strtmp + ".jpg");
		//cv::imshow("image", image);
		//归一化图像大小
		cv::Mat normImage;
		cv::resize(image, normImage, cv::Size(300, 300));
		//求特征图
		IplImage z_ipl = normImage;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, 20, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);
		//将计算得到的特征存到文件中，正样本存入PositiveFeatures.txt，负样本存入NegativeFeatures.txt
		if (labelName == '-')//标签是负样本
		{
			for (int i = 0; i < map->sizeX*map->sizeY*map->numFeatures; i++)
			{
				f2 << *map->map << " ";
				*map->map++;
			}
		}
		else//标签是正样本
		{
			for (int i = 0; i < map->sizeX*map->sizeY*map->numFeatures; i++)
			{
				f1 << *map->map << " ";
				*map->map++;
			}
		}
	}

	fp.close();
	f1.close();
	f2.close();
	std::cout << "The train samples are ready!" << std::endl;
}