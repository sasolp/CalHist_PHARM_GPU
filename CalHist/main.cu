
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <string>
#include <random>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <opencv.hpp>

#include "math.h"
using namespace cv;
using namespace std;
#include "jstruct.h"
#include "residualMaker.cuh"
#include "168modelMaker.cuh"





double generateGaussianNoise(double mu, double sigma)
{
	static const double epsilon = std::numeric_limits<double>::min();
	static const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static  bool generate = false;
	generate = !generate;

	if (!generate)
		return z1 * sigma + mu;

	double u1, u2;
	do
	{
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}
void createHostKernels(float** hostKernels, int*sizes, int*offsets, int *shifts)
{
	float numbers [64];
	int offset = 0;
	float sum = 0;
	float min = 100000;
	float max = -100000;
	int rows;
	int cols;
	int shift_y;
	int shift_x;
	float **kernel = new float*[8];
	for (size_t i = 0; i < 8; i++)
	{
		kernel[i] = new float[8];
	}

	for (int t = 0; t < Total; t++)
	{
		srand(t +1);
		rows = rand() % 8 + 1;
		cols = rand() % 8 + 1;
		shift_y = rand() % 8 + 1;
		shift_x = rand() % 8 + 1;
		//if (t < 10)
		//printf("\nrows : %d, cols : %d, offset : %d\n", rows, cols, offset);
		sizes[t * 2 + 0] = rows;
		sizes[t * 2 + 1] = cols;
		shifts[t * 2 + 0] = shift_y;
		shifts[t * 2 + 1] = shift_x;
		const int nrolls = rows*cols;  // number of experiments
		offsets[t] = offset;
		/*float **kernel = new float*[rows];
		for (size_t i = 0; i < rows; i++)
		{
			kernel[i] = new float[cols];
		}*/
		sum = 0;
		min = 100000;
		max = -100000;
		for (int i = 0; i<nrolls;) {
			float number = (float)generateGaussianNoise(0, 1);
			numbers[i] = number;
			//if ((number >= -2) && (number < 20))
			{
				if (number > max)
				{
					max = number;
				}
				if (number < min)
				{
					min = number;
				}
				++i;
			}
		}
		min = min - 0.1f;
		for (int i = 0; i<nrolls; ++i) {
			numbers[i] -= min;
			sum += numbers[i] ;
		}
		float sum1 = 0;
		for (int i = 0; i<nrolls; ++i) {
			numbers[i] = (numbers[i]) / sum;
			sum1 += numbers[i];
		}
		int c = 0;
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				(*hostKernels)[offset + i * cols + j] = kernel[i][j] =numbers[c++];
				
			}
		}
		offset += nrolls;
		for (size_t i = 0; i < rows; i++)//h reverse
		{
			for (size_t j = 0; j < cols; j++)
			{
				(*hostKernels)[offset + i * cols + j] = kernel[rows - 1 -i][j];
			}
		}
		offset += nrolls;

		for (size_t i = 0; i < rows; i++)//v reverse
		{
			for (size_t j = 0; j < cols; j++)
			{
				(*hostKernels)[offset + i * cols + j] = kernel[i][cols - 1 - j];
			}
		}
		offset += nrolls;

		for (size_t i = 0; i < rows; i++)//hv reverse
		{
			for (size_t j = 0; j < cols; j++)
			{
				(*hostKernels)[offset + i * cols + j] = numbers[--c];
				
			}
		}
		offset += nrolls;
	}
	for (size_t i = 0; i < rows; i++)
	{
		delete kernel[i];
	}
	delete kernel;
}

void free_hists(PSRM_Features &host_features)
{
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaFree(host_features.hists[i]);
	}
}
int save_features(string file_path, int class_id, int** hists)
{
	int ret_val = 0;
	file_path += "_Features.fea";
	FILE* fp_result = fopen(file_path.c_str(), "wb+");
	if (fp_result)
	{
		std::fprintf(fp_result, "%d\n", class_id);
		for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
		{
			for (size_t j = 0; j < Total * BINS_COUNT; j++)
			{
				std::fprintf(fp_result, "\t%d", hists[i][j]);
			}
			std::fprintf(fp_result, "\n");
		}
		fclose(fp_result);
	}
	else
	{
		ret_val = -1;
	}
	return ret_val;
}
/*extern "C" __declspec(dllimport)
_Ret_maybenull_
void*
__stdcall
LoadLibraryA(
_In_ const char* lpLibFileName
);
typedef int*(FAR __stdcall *FARPROC)();
extern "C" __declspec(dllimport)
FARPROC
__stdcall
GetProcAddress(
_In_ void* hModule,
_In_ const char* lpProcName
);
#define _Post_equals_last_error_     _SAL2_Source_(_Post_equals_last_error_, (),  _Post_satisfies_(_Curr_ != 0))
extern "C" __declspec(dllimport)
_Check_return_ _Post_equals_last_error_
int
__stdcall
GetLastError(
void
);*/
int main(int argc, char*argv[])
{
	FILE* fp_OUTPUT = 0;
	if (argc < 2)
	{
		printf("Please, Enter the List File Path as first argument [and the quality factor as second argument]");
		getchar();
		return -1;
	}
	if (argc >= 2)
	{

		fp_OUTPUT = fopen(argv[2],"a");
	}
	/*void* snoop_lib = LoadLibraryA("C:\\Users\\user\\Documents\\Visual Studio 2013\\Projects\\CalHist_Pharm\\x64\\Release\\JPEGSnoop.dll");
	//void* snoop_lib = LoadLibraryA("JPEGSnoop.dll");
	if (snoop_lib == 0)
	{
		
		printf("\nError : Can't load the Snoop DLL %d\n", GetLastError());
		return - 2;
	}
	GetJPEGQualityEx = (Get_JPEG_Quality_Ex)GetProcAddress(snoop_lib, "GetJPEGQuality");*/
	if (fp_OUTPUT) { fprintf(fp_OUTPUT, "file list is %s\n", argv[1]); fflush(fp_OUTPUT); }printf( "file list is %s\n", argv[1]);
	FILE* fp_list = fopen(argv[1], "r");
	FILE *fp_existance = 0;
	if (!fp_list)
	{
		if (fp_OUTPUT) {fprintf(fp_OUTPUT, "the your Entered List File Path is not exist");fflush(fp_OUTPUT); }printf("the your Entered List File Path is not exist");
		getchar();
		return -2;
	}

	cudaError_t cudaStatus;
	PSRM_Features host_features = {};
	int *hists[COUNT_OF_SUBMODELS];
	for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
	{
		hists[i] = new int[Total * BINS_COUNT];
	}
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaStatus = cudaMalloc(&(host_features.hists[i]), Total * BINS_COUNT * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!", __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
	}
	int class_id = 0;
	int dim1 = 0, dim2 = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	char str_file_path[_MAX_PATH];
	float *kernels = new float[Total * 64 * 4];
	float *dev_kernels;
	float* dev_src = NULL, *host_dev_residuals[KERNELS_COUNT] = {};
	float* host_src[KERNELS_COUNT] = {};
	cudaStream_t streams[STREAM_COUNT + 5];
	cudaStatus = cudaSetDevice(0);
	cudaEvent_t start[STREAM_COUNT], stop[STREAM_COUNT];
	int sizes[Total * 2];
	int shifts[Total * 2];
	int offsets[Total];
	int *dev_sizes;
	int *dev_shifts;
	int *dev_offsets;
	
	int offset = 0;
	/*	for (size_t i = 0; i < 4; i++)
		{
			for (size_t i = 0; i < sizes[0]; i++)//hv reverse
			{
				for (size_t j = 0; j < sizes[1]; j++)
				{
					printf("\t%f", kernels[offset + i * sizes[1] + j]);
				}
				printf("\n");
			}
			printf("\n\n");
			offset += sizes[0] * sizes[1];
		}
	
	printf("\n\n");
	printf("\n\n");*/
	const int MAX_COLS = (const int)(1100);
	const int MAX_ROWS = (const int)(1100);
	const int TILE_HEIGHT = 8;
	const int TILE_WEIGHT = 8;
	mat2D<Mat *>bases(8,8) ;
	int QF = 75, q = 2;
	uint3 blocks_res = { 128, 128, 1 }, threads_res = { 2, 2, 1 };
	for (size_t i = 0; i < STREAM_COUNT + 5; i++)
	{
		cudaStreamCreate(&streams[i]);
		if (i >= STREAM_COUNT)continue;
		cudaEventCreate(&start[i]);
		cudaEventCreate(&stop[i]);
	}
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", __LINE__); fflush(fp_OUTPUT);
		}
	}
	cudaStatus = cudaMalloc((void**)&dev_kernels, Total * 64 * 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!", __LINE__); fflush(fp_OUTPUT);
		}
		return 1;
	}
	

	
	cudaStatus = cudaMalloc((void**)&dev_src, MAX_COLS * MAX_ROWS * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!", __LINE__); fflush(fp_OUTPUT);
		}
		return 1;
	}
	for (size_t i = 0; i < KERNELS_COUNT; i++)
	{
		cudaStatus = cudaHostAlloc((void**)&host_src[i], MAX_COLS * MAX_ROWS * sizeof(float), cudaHostAllocMapped | cudaHostAllocWriteCombined);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}

		cudaStatus = cudaHostGetDevicePointer(&host_dev_residuals[i], host_src[i], 0);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
	}
	
	cudaStatus = cudaMalloc((void**)&dev_shifts, Total * 2 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!(dev_shifts)", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!(dev_shifts)", __LINE__); fflush(fp_OUTPUT);
		}
		return 1;
	}
	cudaStatus = cudaMalloc((void**)&dev_sizes, Total * 2 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!(dev_sizes)", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!(dev_sizes)", __LINE__); fflush(fp_OUTPUT);
		}
		return 1;
	}
	cudaStatus = cudaMalloc((void**)&dev_offsets, Total * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "\nline %d: cudaMalloc failed!(dev_offsets)", __LINE__);
		if (fp_OUTPUT) {
			fprintf(fp_OUTPUT, "\nline %d: cudaMalloc failed!(dev_offsets)", __LINE__); fflush(fp_OUTPUT);
		}
		return 1;
	}
	GetBases(bases);
	//std::fprintf(stderr, "\nline %d: cudaMemcpy failed!", __LINE__);
	while (!feof(fp_list))
	{
		host_features.last_index = 0;
		host_features.submodel_index = 0;
		if (!fgets(str_file_path, 260, fp_list))
		{
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nProcessing is finished"); fflush(fp_OUTPUT); }printf("\nProcessing is finished"); 
			break;
		}
		sscanf(str_file_path, "%d", &class_id);
		strcpy_s(str_file_path, strchr(str_file_path, '\t') + 1);
		if (str_file_path[strlen(str_file_path) - 1] == '\n')
		{
			str_file_path[strlen(str_file_path) - 1] = 0;
		}
		while (str_file_path[0] == ' ' || str_file_path[0] == '\t')strcpy(str_file_path, str_file_path + 1);
		while (str_file_path[strlen(str_file_path) - 1] == ' ' || str_file_path[strlen(str_file_path) - 1] == '\t')str_file_path[strlen(str_file_path) - 1] = 0;
		fp_existance = fopen(string(str_file_path).c_str(), "rb");
		if (!fp_existance)
		{
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nFile Not Found : %s", str_file_path); fflush(fp_OUTPUT); }printf("\nFile Not Found : %s", str_file_path);
			continue;
		}
		fclose(fp_existance);
		if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nclassID = %d, path = %s", class_id, str_file_path); fflush(fp_OUTPUT); }printf("\nclassID = %d, path = %s", class_id, str_file_path);
		string tmp = str_file_path;
		std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
		strcpy(str_file_path, tmp.c_str());
		if (!strstr(str_file_path, ".jpg") && !strstr(str_file_path, ".jpeg"))
		{
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\t Not JPEG"); fflush(fp_OUTPUT); } printf("\t Not JPEG"); 
			continue;
		}
		fp_existance = fopen((string(str_file_path) + string("_Features.fea")).c_str(), "rb");
		if (fp_existance)
		{
			fclose(fp_existance);
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\tProcessed in the Past"); fflush(fp_OUTPUT); }printf("\tProcessed in the Past");
			continue;
		}
		fp_existance = fopen(str_file_path, "rb");
		if (fp_existance)
		{
			char buff[32];
			offset = fread(buff, 1, 32, fp_existance);
			fclose(fp_existance);
			if (offset == 0)
			{
				if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\tEmpty input File"); fflush(fp_OUTPUT); }printf("\tEmpty input File"); 
				continue;
			}
		}
		/*QF = GetJPEGQualityEx(str_file_path);
		printf("\nQuality = %f\n", QF);
		q = GetQ(QF);*/
		Mat img;
		createHostKernels(&kernels, sizes, offsets, shifts);
		cudaStatus = cudaMemcpy(dev_kernels, kernels, Total * 64 * 4 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
		cudaStatus = cudaMemcpy(dev_shifts, shifts, Total * 2 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
		cudaStatus = cudaMemcpy(dev_sizes, sizes, Total * 2 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
		cudaStatus = cudaMemcpy(dev_offsets, offsets, Total * 1 * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			return 1;
		}
		try
		{

			read_jpeg(string(str_file_path).c_str(), &bases, QF, img);
		}
		catch (exception &ex)
		{

		}
		q = GetQ(QF);
		if (!img.data || !img.cols || !img.rows)
		{
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nNULL image, %s", str_file_path); fflush(fp_OUTPUT); }printf("\nNULL image, %s", str_file_path);
			continue;
		}
		if (img.cols > 1024 && img.rows > 1024)
			img = img(Rect(0, 0, 1024, 1024));
		else if (img.cols > 1024)
			img = img(Rect(0, 0, 1024, img.rows));
		else if (img.rows > 1024)
			img = img(Rect(0, 0, img.cols, 1024));
		copyMakeBorder(img, img, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(0, 0, 0, 0));
		//img.convertTo(img, CV_8U);
		//cvNamedWindow("x", CV_WINDOW_KEEPRATIO);
		//imshow("x", img); cvWaitKey();
		dim1 = (int)ceil(sqrt(img.cols / 8.0f));
		blocks_res.x =  dim1;
		threads_res.x = (unsigned int)ceil(img.cols / 8.0f / dim1);
		dim2 = (int)ceil(sqrt(img.rows / 8.0f));
		blocks_res.y = dim2;
		threads_res.y =  (unsigned int)ceil(img.rows / 8.0f / dim2);
		//img.convertTo(img, CV_8U);
		//imshow("1", img); cvWaitKey();
		//img.convertTo(img, CV_32FC1);
		cudaStatus = cudaMemcpy(dev_src, img.data, img.rows * img.cols * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			continue;
		}
		cudaStatus = cudaDeviceSynchronize();
		// do some work on the GPU

		float t1 = (float)clock();
		int i = 0;
		int kernel_index = 1;
		cudaEventRecord(start[i], streams[STREAM_COUNT + 0]);

		for (; i < 2; i++)
		{
			make_res_1st << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 0] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaEventRecord(stop[0], streams[STREAM_COUNT + 0]); cudaEventSynchronize(stop[0]);
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 0]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			//return 1;
		}
		kernel_index = 1; cudaEventRecord(start[1], streams[STREAM_COUNT + 1]);
		for (; i < 4; i++)
		{
			make_res_2x2 << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 1] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaEventRecord(stop[1], streams[STREAM_COUNT + 1]); cudaEventSynchronize(stop[1]);
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 1]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			//return 1;
		}
		kernel_index = 1; cudaEventRecord(start[2], streams[STREAM_COUNT + 2]);
		for (; i < 7; i++)
		{
			make_res_3st << <blocks_res, threads_res, 0, streams[STREAM_COUNT + 2] >> >(dev_src, host_dev_residuals[i], img.cols, img.rows, kernel_index++, TILE_WEIGHT, TILE_HEIGHT);

		}
		cudaEventRecord(stop[2], streams[STREAM_COUNT + 2]); cudaEventSynchronize(stop[2]);
		cudaStatus = cudaStreamSynchronize(streams[STREAM_COUNT + 2]);
		if (cudaStatus != cudaSuccess) {
			std::fprintf(stderr, cudaGetErrorString(cudaStatus), __LINE__);
			if (fp_OUTPUT) {
				fprintf(fp_OUTPUT, cudaGetErrorString(cudaStatus), __LINE__); fflush(fp_OUTPUT);
			}
			//return 1;
		}
		

		
		make_models_1st(host_dev_residuals, dev_kernels, dev_offsets, dev_sizes, dev_shifts, streams, host_features, q, img.cols, img.rows);
		if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\n make_models_1st is done\n"); fflush(fp_OUTPUT); }printf("\n make_models_1st is done\n");
		
		make_models_2x2(host_dev_residuals, dev_kernels, dev_offsets, dev_sizes, dev_shifts, streams, host_features, q, img.cols, img.rows);
		if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\n make_models_2x2 is done\n"); fflush(fp_OUTPUT); }printf("\n make_models_2x2 is done\n");
		
		make_models_3st(host_dev_residuals, dev_kernels, dev_offsets, dev_sizes, dev_shifts, streams, host_features, q, img.cols, img.rows);
		if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\n make_models_3st is done\n"); fflush(fp_OUTPUT); }printf("\n make_models_3st is done\n");
		
		for (size_t i = 0; i < STREAM_COUNT; i++)
		{
			cudaStreamSynchronize(streams[i]);
		}
		/*
		*/
		cudaDeviceSynchronize();
		float t2 = (clock() - t1) / CLOCKS_PER_SEC;
		//printf("\n lastIndex = %d\n", host_features.submodel_index);
		
		for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
		{
			cudaStatus = cudaMemcpy(hists[i], host_features.hists[i], Total * BINS_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				std::fprintf(stderr, "\nline %d: %s\n", __LINE__, cudaGetErrorString(cudaStatus) );
				if (fp_OUTPUT) {
					fprintf(fp_OUTPUT, "\nline %d: %s\n", __LINE__, cudaGetErrorString(cudaStatus)); fflush(fp_OUTPUT);
				}
				break;
			}
			//printf("\n%d", host_features.sub_model_index[i]);
		}
		if (cudaStatus != cudaSuccess)
		{

			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nfeature extracting is not successfully"); fflush(fp_OUTPUT); }printf("\nfeature extracting is not successfully");
			continue;
		}
		/*for (size_t i = 0; i < 1; i++)
		{
			printf("\n\n\n\n");
			for (size_t j = 0; j < Total * BINS_COUNT; j++)
			{
				printf(" %d", hists[i][j]);
			}

		}
		for (size_t i = 6; i < 7; i++)
		{
			printf("\n\n\n\n");
			for (size_t j = 0; j < Total * BINS_COUNT; j++)
			{
				printf(" %d", hists[i][j]);
			}

		}
		printf("\n\n", sum);*/
		
		/*for (size_t i = 0; i < STREAM_COUNT; i++)
		{

		cudaEventElapsedTime(&elapsedTime,
		start[i], stop[i]);
		std::fprintf(stderr, "\nline %d: \nGPU Elapsed Time is %f Second", elapsedTime/1000, t2);
		}*/
		std::fprintf(stderr, "\nCPU Elapsed Time is %f Seconds\n", t2); 
		if (fp_OUTPUT) { fprintf(fp_OUTPUT, "\nCPU Elapsed Time is %f Seconds\n", t2); fflush(fp_OUTPUT); }
		if (save_features(str_file_path, class_id, hists))
		{
			if (fp_OUTPUT) { fprintf(fp_OUTPUT, "Saving in file is not successfully \n"); fflush(fp_OUTPUT); }printf("Saving in file is not successfully \n");
		}
	}
	FreeBases(&bases);
	cudaFree(dev_src);
	cudaFree(dev_kernels);
	cudaFree(dev_offsets);
	cudaFree(dev_sizes);
	cudaFree(dev_shifts);
	//cudaFree(dev_residuals);
	for (size_t i = 0; i < HIST_COUNT; i++)
	{
		cudaFree(host_features.hists[i]);
	}
	for (size_t i = 0; i < COUNT_OF_SUBMODELS; i++)
	{
		delete hists[i];
	}
	for (size_t i = 0; i < KERNELS_COUNT; i++)
	{
		cudaFreeHost(host_src[i]);
	}
	for (size_t i = 0; i < STREAM_COUNT + 5; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	fclose(fp_list);
	delete kernels;
	return 0;
}

