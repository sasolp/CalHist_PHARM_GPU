#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "168ModelMaker.cuh"
#include <cmath>
#include <time.h>
#include <string>
#include <vector>
using namespace std;
	int residual_offset = 0;
	extern uint3 blocks = { 30, 30, 1 };
	extern uint3  threads = { 1, 1, 1 };
void make_models_1st(float ** host_dev_residuals, float* dev_kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height)
{
	residual_offset = 0;
	const int shift_offset = 3;
	host_features.last_index = -1;
	host_features.submodel_index = -1;

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 1;
	strcpy(host_features.name[host_features.last_index], "s1_spam14_R");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_1st::R]);
	//return;
	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 2;
	strcpy(host_features.name[host_features.last_index], "s1_spam14_U");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_1st::U]);


	return;

}
void make_models_3st(float ** host_dev_residuals, float* dev_kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height)
{
	//return;

	const int shift_offset = 5;
	

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 3;
	strcpy(host_features.name[host_features.last_index], "s3_spam14_R");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_3st::R_]);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 4;
	strcpy(host_features.name[host_features.last_index], "s3_spam14_U");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_3st::U_]);


	return;
}
void make_models_2x2(float ** host_dev_residuals, float* dev_kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height)
{
	//return;
	const int shift_offset = 3;

	host_features.index[++host_features.last_index] = 1; host_features.sub_model_index[++host_features.submodel_index] = host_features.last_index;
	host_features.submodel[host_features.last_index] = 5;
	strcpy(host_features.name[host_features.last_index], "s2x2_spam14_H");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_2x2::Dh]);

	host_features.index[++host_features.last_index] = 2;
	host_features.submodel[host_features.last_index] = 6;
	strcpy(host_features.name[host_features.last_index], "s2x2_spam14_V");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_2x2::Dv]);
	host_features.index[++host_features.last_index] = 3;
	host_features.submodel[host_features.last_index] = 7;
	strcpy(host_features.name[host_features.last_index], "s2x2_spam14_DMaj");
	PROJ_HIST_SPAM(host_dev_residuals[RESIDUALS_2x2::Dd]);

	
	return;
}


__global__ void proj_hist_spam(float* first_residual, float* kernels, int*offsets, int* sizes, int* shifts, int* out_hist, int q, int shift_offset, int src_width, int src_height)
{
	//return;
	int hist0 =  0, hist1 = 0;
	float gauss_kernels[4][8][8];
	int kernel_index = blockIdx.x * gridDim.x + blockIdx.y;
	int kernel_offset = offsets[kernel_index];
	int kernel_rows = sizes[kernel_index * 2 + 0];
	int kernel_cols = sizes[kernel_index * 2 + 1];
	int kernel_size = kernel_rows * kernel_cols;
	int shift_y = shifts[kernel_index * 2 + 0];
	int shift_x = shifts[kernel_index * 2 + 1];
	int row_offset = 0;
	float  sums, tmp = 0;
	int img_offset = 0;
	int i, j, w, z, m, n;
	int end_x = src_width - kernel_cols - 1;
	int end_y = src_height - kernel_rows - 1;
	int bin_edge = q;
	//if (blockIdx.x == 0 && blockIdx.y == 0)printf("\n\n");
	for (j = 0; j < kernel_rows; j++)
	{
		row_offset = kernel_offset + j * kernel_cols;
		for (z = 0; z < kernel_cols; z++)
		{
			gauss_kernels[0][j][z] = kernels[row_offset + z];
			//if (blockIdx.x == 0 && blockIdx.y == 0)printf(" %f", gauss_kernels[0][j][z]);
		}
	}
	//if (blockIdx.x == 0 && blockIdx.y == 0)printf("\n\n");
	kernel_offset = kernel_offset +  kernel_size;
	for (j = 0; j < kernel_rows; j++)
	{
		row_offset = kernel_offset + j * kernel_cols;
		for (z = 0; z < kernel_cols; z++)
		{
			gauss_kernels[1][j][z] = kernels[row_offset + z];
			//if (blockIdx.x == 0 && blockIdx.y == 0)printf(" %f", gauss_kernels[1][j][z]);
		}
	}
	//if (blockIdx.x == 0 && blockIdx.y == 0)printf("\n\n");
	kernel_offset = kernel_offset + kernel_size;
	for (j = 0; j < kernel_rows; j++)
	{
		row_offset = kernel_offset + j * kernel_cols;
		for (z = 0; z < kernel_cols; z++)
		{
			gauss_kernels[2][j][z] = kernels[row_offset + z];
			//if (blockIdx.x == 0 && blockIdx.y == 0)printf(" %f", gauss_kernels[2][j][z]);
		}
	}
	//if (blockIdx.x == 0 && blockIdx.y == 0)printf("\n\n");
	kernel_offset = kernel_offset +  kernel_size;
	for (j = 0; j < kernel_rows; j++)
	{
		row_offset = kernel_offset + j * kernel_cols;
		for (z = 0; z < kernel_cols; z++)
		{
			gauss_kernels[3][j][z] = kernels[row_offset + z];
			//if (blockIdx.x == 0 && blockIdx.y == 0)printf(" %f", gauss_kernels[3][j][z]);
		}
	}
	
	w = 0;
	i = shift_y;
	j = shift_x;
	for (; i < end_y; i+= 8)//+3 for 4*4 kernel
	{
		for (; j < end_x; j+= 8)//+3 for 4*4 kernel
		{
			img_offset = i * src_width;
			sums = 0;
			for (m = 0; m < kernel_rows; m++)
			{
				for (n = 0; n < kernel_cols; n++)
				{
					tmp = first_residual[img_offset + m * src_width + j + n];
					sums += (tmp * gauss_kernels[w][m][n]);

				}
			}
			tmp = abs(sums);
			//if (i < 100 && j < 100)printf(" %f", tmp);
			if (tmp >= 0 && tmp < bin_edge)
			{
				hist0 += 1;
			}
			else if (tmp >= bin_edge && tmp < 2 * bin_edge)
			{
				hist1 += 1;

			}
		}
	}
	
	w++;
	i = 1 - shift_y - kernel_rows + shift_offset;
	i = i < 0 ? 0 : i;

	j = shift_x ;
	for (; i < end_y; i += 8)//+3 for 4*4 kernel
	{
		for (; j < end_x; j += 8)//+3 for 4*4 kernel
		{
			img_offset = i * src_width;
			sums = 0;
			for (m = 0; m < kernel_rows; m++)
			{
				for (n = 0; n < kernel_cols; n++)
				{
					tmp = first_residual[img_offset + m * src_width + j + n];
					sums += (tmp * gauss_kernels[w][m][n]);

				}
			}
			tmp = abs(sums);
			if (tmp >= 0 && tmp < bin_edge)
			{
				hist0 += 1;
			}
			else if (tmp >= bin_edge && tmp < 2 * bin_edge)
			{
				hist1 += 1;

			}
		}
	}
	w++;
	i = shift_y;
	j = 1 - shift_x - kernel_cols + shift_offset;
	j = j < 0 ? 0 : j;
	for (; i < end_y; i += 8)//+3 for 4*4 kernel
	{
		for (; j < end_x; j += 8)//+3 for 4*4 kernel
		{
			img_offset = i * src_width;
			sums = 0;
			for (m = 0; m < kernel_rows; m++)
			{
				for (n = 0; n < kernel_cols; n++)
				{
					tmp = first_residual[img_offset + m * src_width + j + n];
					sums += (tmp * gauss_kernels[w][m][n]);

				}
			}
			tmp = abs(sums);
			if (tmp >= 0 && tmp < bin_edge)
			{
				hist0 += 1;
			}
			else if (tmp >= bin_edge && tmp < 2 * bin_edge)
			{
				hist1 += 1;

			}
		}
	}
	w++;
	i = 1 - shift_y - kernel_rows + shift_offset;
	i = i < 0 ? 0 : i;

	j = 1 - shift_x - kernel_cols + shift_offset;
	j = j < 0 ? 0 : j;
	for (; i < end_y; i += 8)//+3 for 4*4 kernel
	{
		for (; j < end_x; j += 8)//+3 for 4*4 kernel
		{
			img_offset = i * src_width;
			sums = 0;
			for (m = 0; m < kernel_rows; m++)
			{
				for (n = 0; n < kernel_cols; n++)
				{
					tmp = first_residual[img_offset + m * src_width + j + n];
					sums += (tmp * gauss_kernels[w][m][n]);

				}
			}
			tmp = abs(sums);
			if (tmp >= 0 && tmp < bin_edge)
			{
				hist0 += 1;
			}
			else if (tmp >= bin_edge && tmp < 2 * bin_edge)
			{
				hist1 += 1;

			}
		}
	}
	out_hist[kernel_index * 2 + 0] = hist0;
	out_hist[kernel_index * 2 + 1] = hist1;
	
}