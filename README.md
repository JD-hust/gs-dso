# Direct Sparse Odometry with Continuous 3D Gaussian Maps for Indoor Environments

### 1. Related Papers
**Direct Sparse Odometry with Continuous 3D Gaussian Maps for Indoor Environments**, *Deng J, Lang F, Yuan Z, Yang X*.

 [paper link](https://arxiv.org/abs/2503.03373)

### 2. Installation

		git clone git@github.com:JD-hust/gs-dso.git

#### 2.1 Environment
We run the code on Ubantu 20.04 with CUDA 11.8.

#### 2.2 Required Dependencies

##### 2.2.1 Suitesparse 
Install with

		sudo apt-get install libsuitesparse-dev libboost-all-dev

##### 2.2.2 Eigen3
Eigen 3.2.8, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

##### 2.2.3 OpenCV
OpenCV 2.4.9, Follow [OpenCV Installation](https://opencv.org/releases/page/7/).

##### 2.2.4 Pangolin
Pangolin, Follow [Pangolin Installation](https://github.com/stevenlovegrove/Pangolin).

##### 2.2.5 ziplib
Install with

		sudo apt-get install zlib1g-dev
		cd dso/thirdparty
		tar -zxvf libzip-1.1.1.tar.gz
		cd libzip-1.1.1/
		./configure
		make
		sudo make install
		sudo cp lib/zipconf.h /usr/local/include/zipconf.h
	
##### 2.2.6 Libtorch
Libtorch 2.5.1, follow [Libtorch Installation](https://pytorch.org/cppdocs/installing.html).

#### 2.3 Build

		cd gs-dso
		mkdir build
		cd build
		cmake ..
		make -j4
		
### 3. Usage

#### 3.1 Dataset Format

		<sequence folder name>
			|____________rgb/
			|____________data.ply
			|____________transfroms.json
			|____________groundtruth.txt
			|____________associate.txt/rgb.txt
			
Please adjust your dataset file directory and format as described above.

#### 3.2 Run
To process prior lidar map, run it with the following instruction:

		./opensplat --val-render ../output -i PATH/to/datasets/ -o ../output/ -n 20000 --keep-crs --sh-degree 1

and then run it with the following instruction for localization:

		./gsdso_dataset files=PATH/to/datasets calib=../calib/DATASETS/calib.txt result=../output/ mode=1 preset=0

We provide a pre-trained prior map and an example sequence in the [link](https://drive.google.com/drive/folders/1goUk6_3Pf9eM31T2yYfQpnUxonxfSvzU?usp=drive_link).There is the instruction to run on the sequence:

		./gsdso_dataset files=../example/S2 calib=../calib/ICL/calib.txt result=../example/output/ mode=1 preset=0


For more details on configuration parameters, see [Direct Sparse Odometry](https://github.com/JakobEngel/dso) and [OpenSplat](https://github.com/pierotofy/OpenSplat).

### 4. Acknowledgement
This work is implemented based on [Direct Sparse Odometry](https://github.com/JakobEngel/dso), [RGBD-DSO](https://github.com/HustCK/RGBD-DSO) and [OpenSplat](https://github.com/pierotofy/OpenSplat). Thanks to [J. Engel](https://scholar.google.com/citations?user=ndOMZXMAAAAJ&hl=zh-CN) et al., who open source such excellent code for community.
