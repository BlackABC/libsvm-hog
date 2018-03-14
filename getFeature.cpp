#include "getFeature.h"

//positiveName�����������ļ�   PositiveFeatures.txt
//negativeName�����������ļ�   NegativeFeatures.txt
void makeFeatureFile(const std::string& fileName, const std::string& positiveName, const std::string& negativeName)
{
	std::fstream fp;
	fp.open(fileName);
	std::string str;

	std::fstream f1, f2;
	f1.open(positiveName, std::ios::app);
	f2.open(negativeName, std::ios::app);
	//�������ͼƬ���Ȼ�ȡͼƬ��
	while (std::getline(fp, str))
	{
		const char *c = str.c_str();
		//������ͼƬ����label��
		char picName[7], labelName;
		strncpy(picName, c, 6);
		picName[6] = '\0';
		labelName = c[7];
		std::string strtmp = picName;
		//����ͼ��
		cv::Mat image = cv::imread("C:/Users/Administrator/Desktop/VOC2007/JPEGImages/" + strtmp + ".jpg");
		//cv::imshow("image", image);
		//��һ��ͼ���С
		cv::Mat normImage;
		cv::resize(image, normImage, cv::Size(300, 300));
		//������ͼ
		IplImage z_ipl = normImage;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, 20, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);
		//������õ��������浽�ļ��У�����������PositiveFeatures.txt������������NegativeFeatures.txt
		if (labelName == '-')//��ǩ�Ǹ�����
		{
			for (int i = 0; i < map->sizeX*map->sizeY*map->numFeatures; i++)
			{
				f2 << *map->map << " ";
				*map->map++;
			}
		}
		else//��ǩ��������
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