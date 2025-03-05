/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"

#include "opencv2/highgui/highgui.hpp"

#if HAS_ZIPLIB
	#include "zip.h"
#endif

#include <boost/thread.hpp>
#include <filesystem>

using namespace dso;

inline SE3 getfirstpose(const std::string& path)
{
    std::string strGroundtruthFilename = path + "/groundtruth.txt";
    std::ifstream fGroundtruth(strGroundtruthFilename);
    if (!fGroundtruth.is_open())
    {
        throw std::runtime_error("Please ensure that you have the groundtruth file");
    }

    std::string s;
    if (!std::getline(fGroundtruth, s))
    {
        throw std::runtime_error("Failed to read the first line from groundtruth file");
    }

    while (s[0] == '#')
    {
        if (!std::getline(fGroundtruth, s))
        {
            throw std::runtime_error("Failed to read a valid line from groundtruth file");
        }
    }

    std::stringstream ss(s);
    double t;
    SE3 firstpose;

    double tx, ty, tz, qx, qy, qz, qw;
    if (!(ss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
    {
        throw std::runtime_error("Failed to parse the first pose from groundtruth file");
    }

    Eigen::Quaterniond q(qw, qx, qy, qz);
    firstpose = SE3(q, Vec3(tx, ty, tz));

	std::cout << "First pose: " << q.toRotationMatrix() << " " << Vec3(tx, ty, tz) << std::endl;
	return firstpose;
}

inline int getdir (std::string dir, std::vector<std::string> &files, std::vector<std::string> &depthfiles, std::vector<double> &timestamps)
{
	std::string strAssociationFilename = dir + "/associate.txt";
	std::string strAssociationFilename2 = dir + "/rgb.txt";
	std::ifstream fAssociation;
	
	if(std::filesystem::exists(strAssociationFilename2))
	{
		fAssociation.open(strAssociationFilename2.c_str());
		if(!fAssociation)
		{
			printf("please ensure that you have the rgb file\n");
			return -1;
		}
		while(!fAssociation.eof())
		{
			std::string s;
			std::getline(fAssociation,s);
			if(!s.empty())
			{
				std::stringstream ss;
				ss << s;
				double t;
				std::string sRGB;
				ss >> t;
				timestamps.push_back(t);
				ss >> sRGB;
				sRGB = dir + "/" + sRGB;
				files.push_back(sRGB);
			}
		}
	}
	else if (std::filesystem::exists(strAssociationFilename))
	{
		fAssociation.open(strAssociationFilename.c_str());
		if(!fAssociation)
		{
			printf("please ensure that you have the associate file\n");
			return -1;
		}
		while(!fAssociation.eof())
		{
			std::string s;
			std::getline(fAssociation,s);
			if(!s.empty())
			{
				std::stringstream ss;
				ss << s;
				double t;
				std::string sRGB, sD;
				ss >> t;
				timestamps.push_back(t);
				ss >> sRGB;
				sRGB = dir + "/" + sRGB;
				depthfiles.push_back(sRGB);
				ss >> t;
				ss >> sD;
				sD = dir + "/" + sD;
				files.push_back(sD);
			}
		}
	}
	else
	{
		printf("please ensure that you have the associate or rgb file\n");
		return -1;
	}

    return files.size();
}


struct PrepImageItem
{
	int id;
	bool isQueud;
	ImageAndExposure* pt;

	inline PrepImageItem(int _id)
	{
		id=_id;
		isQueud = false;
		pt=0;
	}

	inline void release()
	{
		if(pt!=0) delete pt;
		pt=0;
	}
};




class ImageFolderReader
{
public:
	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->path = path;
		this->calibfile = calibFile;

#if HAS_ZIPLIB
		ziparchive=0;
		databuffer=0;
#endif

		isZipped = (path.length()>4 && path.substr(path.length()-4) == ".zip");





		if(isZipped)
		{
#if HAS_ZIPLIB
			int ziperror=0;
			ziparchive = zip_open(path.c_str(),  ZIP_RDONLY, &ziperror);
			if(ziperror!=0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for(int k=0;k<numEntries;k++)
			{
				const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if(nstr == "." || nstr == "..") continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		else{
			getdir (path, files, depthfiles, timestamps);
			firstpose = getfirstpose(path);
		}

		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

		widthOrg = undistort->getOriginalSize()[0]; // 原始图像大小
		heightOrg = undistort->getOriginalSize()[1];
		width=undistort->getSize()[0]; // 矫正后图像大小
		height=undistort->getSize()[1];
		
		printf("ImageFolderReader: got %d rgb files and %d depth files in %s!\n", (int)files.size(), (int)depthfiles.size(), path.c_str());

	}
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if(ziparchive!=0) zip_close(ziparchive);
		if(databuffer!=0) delete databuffer;
#endif

		delete undistort;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return  undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0]; // 矫正后图像大小
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		return files.size();
	}

	double getTimestamp(int id)
	{
		if(timestamps.size()==0) return id*0.1f;
		if(id >= (int)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}


	void prepImage(int id, bool as8U=false)
	{

	}


	MinimalImageB* getImageRaw(int id)
	{
			return getImageRaw_internal(id,0);
	}

	ImageAndExposure* getImage(int id, bool forceLoadDirectly=false)
	{
		return getImage_internal(id, 0);
	}

	// 2019.11.07 yzk
	ImageAndExposure* getDepthImage(int id, bool forceLoadDirectly=false)
	{
		return getDepthImage_internal(id, 0);
	}
	// 2018.11.07 yzk


	inline float* getPhotometricGamma()
	{
		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}

	SE3 getFirstpose()
	{
		return firstpose;
	}
	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort* undistort;
	SE3 firstpose;
private:


	MinimalImageB* getImageRaw_internal(int id, int unused)
	{
		if(!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if(databuffer==0) databuffer = new char[widthOrg*heightOrg*6+10000];
			zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*6+10000);

			if(readbytes > (long)widthOrg*heightOrg*6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,(long)widthOrg*heightOrg*6+10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg*heightOrg*30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg*heightOrg*30+10000);

				if(readbytes > (long)widthOrg*heightOrg*30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,(long)widthOrg*heightOrg*30+10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

	// 2019.11.07 yzk
	MinimalImage<unsigned short>* getDepthImageRaw_internal(int id, int unused)
	{
			// CHANGE FOR ZIP FILE
		return IOWrap::readDepthImageBW_16U(depthfiles[id]);
	}
	// 2019.11.07 yzk


	ImageAndExposure* getImage_internal(int id, int unused)
	{
		MinimalImageB* minimg = getImageRaw_internal(id, 0);
		ImageAndExposure* ret2 = undistort->undistort<unsigned char>(
				minimg,
				(exposures.size() == 0 ? 1.0f : exposures[id]),
				(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}

	// 2019.11.07 yzk
	ImageAndExposure* getDepthImage_internal(int id, int unused)
	{
		MinimalImage<unsigned short>* minimg = getDepthImageRaw_internal(id, 0);
		/*
		cv::Mat m(minimg->h, minimg->w, CV_16UC1);
		std::cout << "here-1" << std::endl;
		memcpy(m.data, minimg->data, 2 * minimg->h * minimg->w);
		std::cout << "here-2" << std::endl;
		cv::imshow("third depthimage", m);
		std::cout << "here-3" << std::endl;
		*/
		ImageAndExposure* ret2 = undistort->transformDepthImage<unsigned short>(
				minimg,
				(exposures.size() == 0 ? 1.0f : exposures[id]),
				(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		//std::cout << "here-4" << std::endl;
		delete minimg;

        /*
		cv::Mat m2(ret2->h, ret2->w, CV_32FC1);
		memcpy(m2.data, ret2->image, sizeof(float) * ret2->h*ret2->w);
		m2.convertTo(m2, CV_16UC1);
		cv::imshow("fourth depthimage", m2);
		*/

		return ret2;
	}
	// 2019.11.07 yzk

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0,path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
		for(int i=0;i<(int)exposures.size();i++)
		{
			if(exposures[i] == 0)
			{
				// fix!
				float sum=0,num=0;
				if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
				if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

				if(num>0)
					exposures[i] = sum/num;
			}

			if(exposures[i] == 0) exposuresGood=false;
		}


		if((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}




	std::vector<ImageAndExposure*> preloadedImages;
	std::vector<std::string> files;
	std::vector<std::string> depthfiles;
	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

	std::string path;
	std::string calibfile;

	bool isZipped;

#if HAS_ZIPLIB
	zip_t* ziparchive;
	char* databuffer;
#endif
};

