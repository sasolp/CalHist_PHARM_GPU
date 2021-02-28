#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "residualMaker.cuh"
#include "stdio.h"
#define KERNELS_COUNT 30
__global__ void make_res_1st(float * src, float * dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height)
{
	int tile_top_left =
		(blockIdx.y * tile_height * blockDim.y + threadIdx.y * tile_height) * (src_width/*gridDim.x * blockDim.x * tile_width*/)+
		blockIdx.x * (blockDim.x * tile_width) +
		threadIdx.x * tile_width;
	int row, col;

	row = tile_top_left / src_width + tile_height + 1;
	col = tile_top_left % src_width + tile_width + 1;

	/*if (blockIdx.x > blockDim.x - 2 || blockIdx.y > blockDim.y - 2 || !blockIdx.x || !blockIdx.y)
	{
	if ((!row) || (!col) || (col >= src_width - 1) || row >= (src_height - 1))
	{
	return;
	}
	}*/
	float  ptr_data[11][11];
	float  ptr_results[11][11];
	//return;
	int offset = 0;
	//int end0 = row >= (src_height - 1) ? src_height % 8 : tile_height + 1;
	//int end1 = (col >= src_width - 1) ? src_width % 8 : tile_width + 1;
	int end0 = row >= (src_height - 1) ? src_height - tile_top_left / src_width : tile_height + 1;
	int end1 = (col >= src_width - 1) ? src_width - tile_top_left % src_width : tile_width + 1;
	int tile_top_left0 = tile_top_left;
	//tile_top_left0 +=  + 1;
	switch (kernel_index)
	{
	case 1://left
	case 8://leftdown
		src += src_width;
		break;
	case 2://right
	case 7://rightdown
	case 4://down
		src += src_width + 1;
		break;
	case 3://up
	case 6://RU
		src += 1;
		break;
	}
	for (int i = 0; i < end0; i++)
	{
		//offset = i * tile_width; 
		for (int j = 0; j < end1; j++)
		{
			ptr_data[i][j] = src[tile_top_left0 + j];
		}
		tile_top_left0 += src_width;
	}
	//return; 
	tile_top_left0 = tile_top_left;
	switch (kernel_index)
	{
	case 1://left
		/*for (int i = 0; i < tile_height ; i++)
		{
		for (int j = 0; j < tile_width; j++)
		{
		ptr_results[i][j] = ptr_data[i][j];
		}
		}*/
		for (int i = 0; i < tile_height; i++)
		{
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i][j - 1] = ptr_data[i][j - 1] - ptr_data[i][j];
			}
		}
		break;
	case 2://right

		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i][j] = ptr_data[i][j + 1] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 3://up

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i - 1][j] = ptr_data[i - 1][j] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 4://down

		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i][j] = ptr_data[i + 1][j] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 5://leftup
		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 1][j - 1] = ptr_data[i - 1][j - 1] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 6://rightup
		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i - 1][j] = ptr_data[i - 1][j + 1] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 7://rightdown
		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i][j] = ptr_data[i + 1][j + 1] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 8://leftdown
		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i][j - 1] = ptr_data[i + 1][j - 1] - ptr_data[i][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	}
	tile_top_left0 = tile_top_left;
	for (int i = 0; i < tile_height; i++)
	{
		for (int j = 0; j < tile_width; j++)
		{
			dst[tile_top_left0 + j] = ptr_results[i][j];
		}
		tile_top_left0 += src_width;
	}


}


__global__ void make_res_2x2(float * src, float * dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height)
{
	int tile_top_left =
		(blockIdx.y * tile_height * blockDim.y + threadIdx.y * tile_height) * (src_width/*gridDim.x * blockDim.x * tile_width*/)+
		blockIdx.x * (blockDim.x * tile_width) +
		threadIdx.x * tile_width;
	float  ptr_data[15][15];
	float  ptr_results[15][15];
	int row, col;

	//int end0 = row >= (src_height - 1) ? src_height % 8 : tile_height + 2;
	//int end1 = (col >= src_width - 1) ? src_width % 8 : tile_width + 2;
	row = tile_top_left / src_width + tile_height + 2;
	col = tile_top_left % src_width + tile_width + 2;
	//printf("%d", )
	int end0 = (row >= src_height - 1) ? src_height - tile_top_left / src_width - 1: tile_height + 2;
	int end1 = (col >= src_width - 1) ? src_width - tile_top_left % src_width - 1: tile_width + 2;
	int tile_top_left0 = tile_top_left;
	switch (kernel_index)
	{
	case 4://h
		src += src_width;
		break;
	case 3://v
		src += 1;
		break;
	}
	for (int i = 0; i < end0; i++)
	{
		//offset = i * tile_width;
		for (int j = 0; j < end1; j++)
		{
			ptr_data[i][j] = src[tile_top_left0 + j];
		}
		tile_top_left0 += src_width;
	}
	//return;
	tile_top_left0 = tile_top_left;
	switch (kernel_index)
	{
	case 1://Dh

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 1][j - 1] = ptr_data[i - 1][j - 1] + ptr_data[i - 1][j] - ptr_data[i][j - 1] - ptr_data[i][j];
			}
		}

		break;
	case 2://Dv

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 1][j - 1] = ptr_data[i - 1][j - 1] - ptr_data[i - 1][j] + ptr_data[i][j - 1] - ptr_data[i][j];
			}
		}

		break;
	case 3://Dd

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 1][j - 1] = -ptr_data[i - 1][j - 1] + ptr_data[i - 1][j] + ptr_data[i][j - 1] - ptr_data[i][j];
			}
		}

		break;


	}

	for (int i = 0; i < tile_height; i++)
	{
		for (int j = 0; j < tile_width; j++)
		{
			dst[tile_top_left0 + j] = ptr_results[i][j];
		}
		tile_top_left0 += src_width;
	}
}
__global__ void make_res_3st(float * src, float * dst, int src_width, int src_height, int kernel_index, int tile_width, int tile_height)
{
	int src_size = src_height * src_width;
	int tile_size = tile_width * tile_height;
	int tile_top_left =
		(blockIdx.y * tile_height * blockDim.y + threadIdx.y * tile_height) * (src_width/*gridDim.x * blockDim.x * tile_width*/)+
		blockIdx.x * (blockDim.x * tile_width) +
		threadIdx.x * tile_width;
	float  ptr_data[15][15];
	float  ptr_results[15][15];
	//return;
	int offset = 0;
	/*if (blockIdx.x > blockDim.x - 2 || blockIdx.y > blockDim.y - 2 || !blockIdx.x || !blockIdx.y)
	{
	if ((tile_top_left % src_width < 2) || (tile_top_left % src_width == src_width - 2) || tile_top_left < src_width * 2 || tile_top_left >(src_size - 2*src_width))
	return;
	}*/
	int row, col;

	row = tile_top_left / src_width + tile_height + 3;
	col = tile_top_left % src_width + tile_width + 3;
	int end0 = row >= (src_height - 1) ? src_height - tile_top_left / src_width - 1: tile_height + 3;
	int end1 = (col >= src_width - 1) ? src_width - tile_top_left % src_width - 1 : tile_width + 3;

	int tile_top_left0 = tile_top_left;
	//tile_top_left0 += src_width + 1;
	//tile_top_left0 += src_width + 1;
	//return;
	switch (kernel_index)
	{
	case 1:
		src += src_width * 2;
		break;
	case 2:
		src += src_width * 2 + 1;
		break;
	case 7://rightdown
		src += src_width + 1;
		break;
	case 3://up
		src += 2;
		break;
	case 8://leftdown
		src += src_width;
		break;
	case 4://down
		src += src_width + 2;
		break;
		//case 2://right
	case 6://RU
		src += 1;
		break;
	}
	for (int i = 0; i < end0; i++)
	{
		//offset = i * tile_width;
		for (int j = 0; j < end1; j++)
		{
			ptr_data[i][j] = src[tile_top_left0 + j];
		}
		tile_top_left0 += src_width;
	}
	//return;
	tile_top_left0 = tile_top_left;
	float f1_3 = 1 / 3.0f;
	switch (kernel_index)
	{
	case 1://left

		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 2; j < tile_width + 2; j++)
			{
				ptr_results[i][j - 2] = -ptr_data[i][j - 2] + 3 * ptr_data[i][j - 1] - 3 * ptr_data[i][j] + ptr_data[i][j + 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 2://right
		for (int i = 0; i < tile_height; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i][j - 1] = -ptr_data[i][j + 2] + 3 * ptr_data[i][j + 1] - 3 * ptr_data[i][j] + ptr_data[i][j - 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 3://up


		for (int i = 2; i < tile_height + 2; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i - 2][j] = -ptr_data[i - 2][j] + 3 * ptr_data[i - 1][j] - 3 * ptr_data[i][j] + ptr_data[i + 1][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 4://down


		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 0; j < tile_width; j++)
			{
				ptr_results[i - 1][j] = -ptr_data[i + 2][j] + 3 * ptr_data[i + 1][j] - 3 * ptr_data[i][j] + ptr_data[i - 1][j];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 5://leftup

		for (int i = 2; i < tile_height + 2; i++)
		{
			//offset = i * tile_width;
			for (int j = 2; j < tile_width + 2; j++)
			{
				ptr_results[i - 2][j - 2] = -ptr_data[i - 2][j - 2] + 3 * ptr_data[i - 1][j - 1] - 3 * ptr_data[i][j] + ptr_data[i + 1][j + 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 6://rightup

		for (int i = 2; i < tile_height + 2; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 2][j - 1] = -ptr_data[i - 2][j + 2] + 3 * ptr_data[i - 1][j + 1] - 3 * ptr_data[i][j] + ptr_data[i + 1][j - 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 7://rightdown

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 1; j < tile_width + 1; j++)
			{
				ptr_results[i - 1][j - 1] = -ptr_data[i + 2][j + 2] + 3 * ptr_data[i + 1][j + 1] - 3 * ptr_data[i][j] + ptr_data[i - 1][j - 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	case 8://leftdown

		for (int i = 1; i < tile_height + 1; i++)
		{
			//offset = i * tile_width;
			for (int j = 2; j < tile_width + 2; j++)
			{
				ptr_results[i - 1][j - 2] = -ptr_data[i + 2][j - 2] + 3 * ptr_data[i + 1][j - 1] - 3 * ptr_data[i][j] + ptr_data[i - 1][j + 1];
			}
			tile_top_left0 += src_width;
		}
		break;
	}
	tile_top_left0 = tile_top_left;
	for (int i = 0; i < tile_height; i++)
	{
		for (int j = 0; j < tile_width; j++)
		{
			dst[tile_top_left0 + j] = ptr_results[i][j];
		}
		tile_top_left0 += src_width;
	}
}




