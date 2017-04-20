#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "correction.cuh"
#include <stdlib.h>

using namespace std;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


string sample_file = "/home/vitaly/crops_for_quilting_resized/3553547177614f656358663945623164.png_crop_area.png";
string image_file = "/home/vitaly/output/3553547177614f656358663945623164.png_crop_area.png";

cv::Mat sample;
cv::Mat image;

void initData(const string & sfile, const string & imfile){
	sample = cv::imread(sfile.c_str(), cv::IMREAD_GRAYSCALE);
	image = cv::imread(imfile.c_str(), cv::IMREAD_GRAYSCALE);
}


int main(int argc, char **argv) {
	initData(sample_file, image_file);
	correctionGPU(sample, image, 17, 17);
	return 0;
}
