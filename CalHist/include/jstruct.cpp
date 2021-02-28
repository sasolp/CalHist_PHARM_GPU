#define _CRT_SECURE_NO_DEPRECATE
#include <setjmp.h>
#include "jstruct.h"
#include <stdio.h>
#include <stdlib.h>
#include "jpeg\jerror.h"
#include <string>
#include <opencv.hpp>
extern "C"
{
#include "jpeg\jpeglib.h"
#include "jpeg\jpegint.h"
}
#include "mat2D.h"
using namespace cv;
jstruct::jstruct(std::string filePath)
{
	jstruct(filePath, false);
}

jstruct::jstruct(std::string filePath, bool loadSpatial)
{
	this->loadSpatial = loadSpatial;

	jpeg_load(filePath);

	if (loadSpatial) spatial_load(filePath);
}

jstruct::~jstruct()
{
	for (int i=0; i<markers.size(); i++) delete [] markers[i]; markers.clear();
	for (int i=0; i<coef_arrays.size(); i++) delete coef_arrays[i];	coef_arrays.clear();
    for (int i=0; i<quant_tables.size(); i++) delete quant_tables[i]; quant_tables.clear();
	for (int i=0; i<ac_huff_tables.size(); i++) delete ac_huff_tables[i]; ac_huff_tables.clear();
	for (int i=0; i<dc_huff_tables.size(); i++) delete dc_huff_tables[i]; dc_huff_tables.clear();
	for (int i=0; i<comp_info.size(); i++) delete comp_info[i]; comp_info.clear();
	for (int i=0; i<spatial_arrays.size(); i++) delete spatial_arrays[i]; spatial_arrays.clear();
}

void jstruct::jpeg_load(std::string filePath)
{
	jpeg_decompress_struct cinfo;
	jpeg_saved_marker_ptr marker_ptr;
	jpeg_component_info *compptr;
	jvirt_barray_ptr *coef_arrays;
	FILE *infile;
	JDIMENSION blk_x,blk_y;
	JBLOCKARRAY buffer;
	JCOEFPTR bufptr;
	JQUANT_TBL *quant_ptr;
	JHUFF_TBL *huff_ptr;
	int c_width, c_height, ci, i, j, n;

	/* open file */
	if ((infile = fopen(filePath.c_str(), "rb")) == NULL)
	    throw new std::string("Can't open file to read");

	/* set up the normal JPEG error routines, then override error_exit. */
	cinfo.err = jpeg_std_error(new jpeg_error_mgr());

	/* initialize JPEG decompression object */
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);

	/* save contents of markers */
	jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);

	/* read header and coefficients */
	jpeg_read_header(&cinfo, TRUE);

	/* for some reason out_color_components isn't being set by
    jpeg_read_header, so we will infer it from out_color_space: */
	switch (cinfo.out_color_space) {
		case JCS_GRAYSCALE:
			cinfo.out_color_components = 1;
			break;
		case JCS_RGB:
			cinfo.out_color_components = 3;
			break;
		case JCS_YCbCr:
			cinfo.out_color_components = 3;
			break;
		case JCS_CMYK:
			cinfo.out_color_components = 4;
			break;
		case JCS_YCCK:
			cinfo.out_color_components = 4;
			break;
	}

	this->image_width = cinfo.image_width;
	this->image_height = cinfo.image_height;
	this->image_color_space = cinfo.out_color_space;
	this->image_components = cinfo.out_color_components;
	this->jpeg_color_space = cinfo.jpeg_color_space;
	this->jpeg_components = cinfo.num_components;
	this->progressive_mode = cinfo.progressive_mode;
	this->optimize_coding = 0;
    for (ci = 0; ci < this->jpeg_components; ci++)
	{
		struct_comp_info * temp = new struct_comp_info();
		temp->component_id = cinfo.comp_info[ci].component_id;
		temp->h_samp_factor = cinfo.comp_info[ci].h_samp_factor;
		temp->v_samp_factor = cinfo.comp_info[ci].v_samp_factor;
		temp->quant_tbl_no = cinfo.comp_info[ci].quant_tbl_no;
		temp->ac_tbl_no = cinfo.comp_info[ci].ac_tbl_no;
		temp->dc_tbl_no = cinfo.comp_info[ci].dc_tbl_no;
		this->comp_info.push_back(temp);
	}

	marker_ptr = cinfo.marker_list;
	while (marker_ptr != NULL) 
	{
	    if (marker_ptr->marker == JPEG_COM) 
		{
			char* tempMarker= new char[marker_ptr->data_length + 1];
			tempMarker[marker_ptr->data_length] = '\0';
			/* copy comment string to char array */
			for (i = 0; i < (int) marker_ptr->data_length; i++) 
				tempMarker[i] = marker_ptr->data[i];
			this->markers.push_back(tempMarker);
		}
		marker_ptr = marker_ptr->next;
	}

	for (n = 0; n < NUM_QUANT_TBLS; n++) 
	{
		mat2D<int> * tempMat = new mat2D<int>(DCTSIZE, DCTSIZE);
		
	    if (cinfo.quant_tbl_ptrs[n] != NULL) 
		{
			quant_ptr = cinfo.quant_tbl_ptrs[n];
			for (i = 0; i < DCTSIZE; i++) 
				for (j = 0; j < DCTSIZE; j++)
					tempMat->Write(i, j, quant_ptr->quantval[i*DCTSIZE+j]);
			this->quant_tables.push_back(tempMat);
		}
	}
	
	for (n = 0; n < NUM_HUFF_TBLS; n++) 
	{
		struct_huff_tables * tempStruct = new struct_huff_tables();
		if (cinfo.ac_huff_tbl_ptrs[n] != NULL) {
			huff_ptr = cinfo.ac_huff_tbl_ptrs[n];

			for (i = 1; i <= 16; i++) tempStruct->counts.push_back(huff_ptr->bits[i]);
			for (i = 0; i < 256; i++) tempStruct->symbols.push_back(huff_ptr->huffval[i]);
		}
		this->ac_huff_tables.push_back(tempStruct);
	}

	for (n = 0; n < NUM_HUFF_TBLS; n++) 
	{
		struct_huff_tables * tempStruct = new struct_huff_tables();
		if (cinfo.dc_huff_tbl_ptrs[n] != NULL) {
			huff_ptr = cinfo.dc_huff_tbl_ptrs[n];

			for (i = 1; i <= 16; i++) tempStruct->counts.push_back(huff_ptr->bits[i]);
			for (i = 0; i < 256; i++) tempStruct->symbols.push_back(huff_ptr->huffval[i]);
		}
		this->dc_huff_tables.push_back(tempStruct);
	}

	/* creation and population of the DCT coefficient arrays */
	coef_arrays = jpeg_read_coefficients(&cinfo);
	for (ci = 0; ci < cinfo.num_components; ci++) {
		compptr = cinfo.comp_info + ci;
		c_height = compptr->height_in_blocks * DCTSIZE;
		c_width = compptr->width_in_blocks * DCTSIZE;
		mat2D<int> * tempCoeffs = new mat2D<int>(c_height, c_width);

		/* copy coefficients from virtual block arrays */
		for (blk_y = 0; blk_y < compptr->height_in_blocks; blk_y++) 
		{
			buffer = cinfo.mem->access_virt_barray((j_common_ptr) &cinfo, coef_arrays[ci], blk_y, 1, FALSE);
			for (blk_x = 0; blk_x < compptr->width_in_blocks; blk_x++) 
			{
				bufptr = buffer[0][blk_x];
				for (i = 0; i < DCTSIZE; i++)        /* for each row in block */
					for (j = 0; j < DCTSIZE; j++)      /* for each column in block */
						tempCoeffs->Write(i+blk_y*DCTSIZE, j+blk_x*DCTSIZE, bufptr[i*DCTSIZE+j]);
			}
		}
		this->coef_arrays.push_back(tempCoeffs);
	}

	/* done with cinfo */
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	/* close input file */
	fclose(infile);
}

void jstruct::jpeg_write(std::string filePath, bool optimize_coding)
{
	struct jpeg_compress_struct cinfo;
	int c_height,c_width,ci,i,j,n,t;
	FILE *outfile;
	jvirt_barray_ptr *coef_arrays = NULL;
	JDIMENSION blk_x,blk_y;
	JBLOCKARRAY buffer;
	JCOEFPTR bufptr;  
	
	/* open file */
	if ((outfile = fopen(filePath.c_str(), "wb")) == NULL)
	    throw new std::string("Can't open file to write");

	/* set up the normal JPEG error routines, then override error_exit. */
	cinfo.err = jpeg_std_error(new jpeg_error_mgr());

	/* initialize JPEG decompression object */
	jpeg_create_compress(&cinfo);

	/* write the output file */
	jpeg_stdio_dest(&cinfo, outfile);
	/* Set the compression object with our parameters */
	cinfo.image_width = this->image_width;
	cinfo.image_height = this->image_height;
	cinfo.jpeg_height = this->image_height;
	cinfo.jpeg_width = this->image_width;
	cinfo.input_components = this->image_components;
	cinfo.in_color_space = (J_COLOR_SPACE)this->image_color_space;

	jpeg_set_defaults(&cinfo);
	cinfo.optimize_coding = optimize_coding;
	cinfo.num_components = this->jpeg_components;
	cinfo.jpeg_color_space = (J_COLOR_SPACE)this->jpeg_color_space;
	/* set the compression object with default parameters */
	cinfo.min_DCT_h_scaled_size = 8;
	cinfo.min_DCT_v_scaled_size = 8;
	
	/* basic support for writing progressive mode JPEG */
	if (this->progressive_mode) 
		jpeg_simple_progression(&cinfo);

	/* copy component information into cinfo from jpeg_obj*/
	for (ci = 0; ci < cinfo.num_components; ci++)
	{
		cinfo.comp_info[ci].component_id = this->comp_info[ci]->component_id;
		cinfo.comp_info[ci].h_samp_factor = this->comp_info[ci]->h_samp_factor;
		cinfo.comp_info[ci].v_samp_factor = this->comp_info[ci]->v_samp_factor;
		cinfo.comp_info[ci].quant_tbl_no = this->comp_info[ci]->quant_tbl_no;
		cinfo.comp_info[ci].ac_tbl_no = this->comp_info[ci]->ac_tbl_no;
		cinfo.comp_info[ci].dc_tbl_no = this->comp_info[ci]->dc_tbl_no;
	}

	coef_arrays = (jvirt_barray_ptr *)(cinfo.mem->alloc_small) ((j_common_ptr) &cinfo, JPOOL_IMAGE, sizeof(jvirt_barray_ptr) * cinfo.num_components);
    
	/* request virtual block arrays */
	for (ci = 0; ci < cinfo.num_components; ci++)
	{
		int block_height = this->coef_arrays[ci]->rows / DCTSIZE;
		int block_width = this->coef_arrays[ci]->cols / DCTSIZE;
		cinfo.comp_info[ci].height_in_blocks = block_height;
		cinfo.comp_info[ci].width_in_blocks = block_width;
  
		coef_arrays[ci] = (cinfo.mem->request_virt_barray)(
			(j_common_ptr) &cinfo, JPOOL_IMAGE, TRUE,
			(JDIMENSION)jround_up((long) cinfo.comp_info[ci].width_in_blocks,
			                       (long) cinfo.comp_info[ci].h_samp_factor),
			(JDIMENSION)jround_up((long) cinfo.comp_info[ci].height_in_blocks,
								   (long) cinfo.comp_info[ci].v_samp_factor),
			(JDIMENSION)cinfo.comp_info[ci].v_samp_factor);
	}

	/* realize virtual block arrays */
	jpeg_write_coefficients(&cinfo,coef_arrays);

	/* populate the array with the DCT coefficients */
	for (ci = 0; ci < cinfo.num_components; ci++)
	{
		/* Get a pointer to the mx coefficient array */
    
		c_height = this->coef_arrays[ci]->rows;
		c_width = this->coef_arrays[ci]->cols;

		/* Copy coefficients to virtual block arrays */
		for (blk_y = 0; blk_y < cinfo.comp_info[ci].height_in_blocks; blk_y++)
		{
			buffer = (cinfo.mem->access_virt_barray)((j_common_ptr) &cinfo, coef_arrays[ci], blk_y, 1, TRUE);

			for (blk_x = 0; blk_x < cinfo.comp_info[ci].width_in_blocks; blk_x++)
			{
				bufptr = buffer[0][blk_x];
				for (i = 0; i < DCTSIZE; i++)        /* for each row in block */
					for (j = 0; j < DCTSIZE; j++)      /* for each column in block */
						bufptr[i*DCTSIZE+j] = (JCOEF)this->coef_arrays[ci]->Read(i+blk_y*DCTSIZE, j+blk_x*DCTSIZE);
			}
		}
	}

	/* get the quantization tables */
	for (n = 0; n < (int)this->quant_tables.size(); n++)
	{
		if (cinfo.quant_tbl_ptrs[n] == NULL)
			cinfo.quant_tbl_ptrs[n] = jpeg_alloc_quant_table((j_common_ptr) &cinfo);

		/* Fill the table */
		for (i = 0; i < DCTSIZE; i++) 
			for (j = 0; j < DCTSIZE; j++) 
			{
				t = this->quant_tables[n]->Read(i, j);
				if (t<1 || t>65535)
					throw new std::string("Quantization table entries not in range 1..65535");

				cinfo.quant_tbl_ptrs[n]->quantval[i*DCTSIZE+j] = (UINT16) t;
			}
	}

	/* set remaining quantization table slots to null */
	for (; n < NUM_QUANT_TBLS; n++)
		cinfo.quant_tbl_ptrs[n] = NULL;

	/* Get the AC and DC huffman tables but check for optimized coding first*/
	if (cinfo.optimize_coding == FALSE)
	{
		if (!this->ac_huff_tables.empty())
		{
			for (n = 0; n < (int)this->ac_huff_tables.size(); n++)
			{
				if (cinfo.ac_huff_tbl_ptrs[n] == NULL)
					cinfo.ac_huff_tbl_ptrs[n] = jpeg_alloc_huff_table((j_common_ptr) &cinfo);
				else
				{
					for (i = 1; i <= 16; i++)
						cinfo.ac_huff_tbl_ptrs[n]->bits[i] = (UINT8) this->ac_huff_tables[n]->counts[i-1];
					for (i = 0; i < 256; i++)
						cinfo.ac_huff_tbl_ptrs[n]->huffval[i] = (UINT8) this->ac_huff_tables[n]->symbols[i];
				}
			}
			for (; n < NUM_HUFF_TBLS; n++) cinfo.ac_huff_tbl_ptrs[n] = NULL;
		}
    
		if (!this->dc_huff_tables.empty())
		{
			for (n = 0; n < (int)this->dc_huff_tables.size(); n++)
			{
				if (cinfo.dc_huff_tbl_ptrs[n] == NULL)
					cinfo.dc_huff_tbl_ptrs[n] = jpeg_alloc_huff_table((j_common_ptr) &cinfo);
				else
				{
					for (i = 1; i <= 16; i++)
						cinfo.dc_huff_tbl_ptrs[n]->bits[i] = (unsigned char) this->dc_huff_tables[n]->counts[i-1];
					for (i = 0; i < 256; i++)
						cinfo.dc_huff_tbl_ptrs[n]->huffval[i] = (unsigned char) this->dc_huff_tables[n]->symbols[i];
				}
			}
			for (; n < NUM_HUFF_TBLS; n++) cinfo.dc_huff_tbl_ptrs[n] = NULL;
		}
	}

	/* copy markers */
	for (i = 0; i < (int)this->markers.size(); i++)
	{
		JOCTET * tempMarker = (JOCTET *)this->markers[i];
		int strlen;
		for (strlen=0; tempMarker[strlen]!='\0'; strlen++);
		jpeg_write_marker(&cinfo, JPEG_COM, tempMarker, strlen);   
	}

	/* done with cinfo */
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	/* close the file */
	fclose(outfile);
}

void jstruct::spatial_load(std::string filePath)
{
	/* open file */
	FILE * infile;
	if ((infile = fopen(filePath.c_str(), "rb")) == NULL)
	    throw new std::string("Can't open file to read");

	struct jpeg_decompress_struct cinfo;
	cinfo.err = jpeg_std_error(new jpeg_error_mgr());
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	jpeg_read_header(&cinfo, true);
    jpeg_start_decompress(&cinfo);

	bool grayscale = (cinfo.out_color_space == JCS_GRAYSCALE);
	int colors = 3;	if (grayscale) colors = 1;
	for (int i=0; i < (int)colors; i++)
	this->spatial_arrays.push_back(new mat2D<int>(cinfo.output_height, cinfo.output_width));

	int row_stride = cinfo.output_width * cinfo.output_components ;
	JSAMPARRAY pJpegBuffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
	for (int row=0; row < (int)cinfo.output_height; row++) 
	{
		jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
		for (int col=0; col < (int)cinfo.output_width; col++)
		{
			for (int clr = 0; clr < colors; clr++)
			{
				unsigned int val = (unsigned int)pJpegBuffer[0][colors * col + clr]; 
				spatial_arrays[clr]->Write(row, col, (int)val);
			}
		}
	}
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);
}

const int table0[64] = //(the luminance table):
{ 16, 11, 10, 16, 24, 40, 51, 61,
12, 12, 14, 19, 26, 58, 60, 55,
14, 13, 16, 24, 40, 57, 69, 56,
14, 17, 22, 29, 51, 87, 80, 62,
18, 22, 37, 56, 68, 109, 103, 77,
24, 35, 55, 64, 81, 104, 113, 92,
49, 64, 78, 87, 103, 121, 120, 101,
72, 92, 95, 98, 112, 100, 103, 99 };

const int table1[64] = //(the chrominance table):
{ 17, 18, 24, 47, 99, 99, 99, 99,
18, 21, 26, 66, 99, 99, 99, 99,
24, 26, 56, 99, 99, 99, 99, 99,
47, 66, 99, 99, 99, 99, 99, 99,
99, 99, 99, 99, 99, 99, 99, 99,
99, 99, 99, 99, 99, 99, 99, 99,
99, 99, 99, 99, 99, 99, 99, 99,
99, 99, 99, 99, 99, 99, 99, 99 };

float GetQ(int QF)
{
	if (QF < 1) return 20.0f;
	else if (QF > 99) return 1.0f;
	else return ((((float)65) / 4) - (((float)3) / 20) * ((float)QF));
}
double Find_QF(mat2D <int> *qtable2D, bool tnum)
{
	int *t;
	double QF;


	int *qtable = NULL;
	double QualitySum = 0;
	double quality;

	t = (int *)((tnum) ? table1 : table0);

	try
	{
		if (qtable2D == NULL)
		{

		}


		qtable = qtable2D->vect.data();

		for (int i = 0; i<64; i++)
		{
			double QualityValue = ((double)qtable[i] * 100.0 - 50.0) / (double)t[i];
			QualitySum += QualityValue;
		}
		quality = QualitySum / 64.0;


		QF = round((200.0 - quality) / 2.0);

		if (QF <= 0)  QF = 1;
		if (QF>100) QF = 100;
	}

	catch (int ErrorCode)
	{
		QF = 0;
	}


	return QF;
}

void FreeBases(mat2D<Mat *>*bases)
{
	for (int mode_r = 0; mode_r < 8; mode_r++)
	{
		for (int mode_c = 0; mode_c < 8; mode_c++)
		{
			delete bases->Read(mode_r, mode_c);
		}
	}
}

void GetBases(mat2D<Mat *>&bases)
{
	float sqrt2 = sqrt(2.0f);
	//bases = mat2D<Mat *>(8, 8);
	for (int mode_r = 0; mode_r < 8; mode_r++)
	{
		for (int mode_c = 0; mode_c < 8; mode_c++)
		{
			Mat * basis = new Mat(8, 8, CV_32F);
			for (int m = 0; m < 8; m++)
				for (int n = 0; n < 8; n++)
				{
					float wr = 1; if (mode_r == 0) wr = 1 / sqrt2;
					float wc = 1; if (mode_c == 0) wc = 1 / sqrt2;
					float val = ((wr*wc) / 4) * cos(3.141592654f*mode_r*(2 * m + 1) / 16) * cos(3.141592654f*mode_c*(2 * n + 1) / 16);
					((float*)basis->data)[m*basis->cols + n] = val;
				}
			bases.Write(mode_r, mode_c, basis);
		}
	}
}
void JPEG2spatial(mat2D<int> * coeffs, mat2D<int> * quant_table, mat2D<Mat *>*bases, int q, Mat& spatialImage)
{
	spatialImage = Mat(coeffs->rows, coeffs->cols, CV_32F);

	for (int mode_r = 0; mode_r<8; mode_r++)
		for (int mode_c = 0; mode_c<8; mode_c++)
		{
			int Q = quant_table->Read(mode_r, mode_c);
			//mat2D<float> * basis = bases->Read(mode_r, mode_c);
			Mat * basis = bases->Read(mode_r, mode_c);
			// use basis to compute spatial image
			for (int r = 0; r<coeffs->rows; r += 8)
			{
				for (int c = 0; c<coeffs->cols; c += 8)
				{
					float DCTval = (float)coeffs->Read(mode_r + r, mode_c + c) * Q /*/ q*/;
					for (int m = 0; m<8; m++)
						for (int n = 0; n<8; n++)
						{
							((float*)spatialImage.data)[(r + m)* spatialImage.cols + c + n] += DCTval * ((float*)basis->data)[m*basis->cols + n];

						}
				}
			}
		}
	for (int r = 0; r < coeffs->rows; r++)
	{
		for (int c = 0; c < coeffs->cols; c++)
		{
			((float*)spatialImage.data)[r* spatialImage.cols + c] += 128;
		}
	}
}
void read_jpeg(string path, mat2D<Mat *>*bases, int &QF, Mat &img_jpeg)
{
	//printf("\nxx: %s\n", path.c_str());
	jstruct * JPEGimageStruct = new jstruct(path, false);
	QF = Find_QF(JPEGimageStruct->quant_tables[0], false);
	printf("\nQF = %d \n", QF);
	int q = GetQ(QF);
	//if (JPEGimageStruct->coef_arrays.size() != 1) { throw new std::string("Error: Only grayscale images can be embedded."); }
	if ((JPEGimageStruct->coef_arrays[0]->rows % 8 != 0) || (JPEGimageStruct->coef_arrays[0]->rows % 8 != 0))
	{
		throw new std::string("Error: Image dimensions must be divisible by 8.");
	}

	JPEG2spatial(JPEGimageStruct->coef_arrays[0], JPEGimageStruct->quant_tables[0], bases, q, img_jpeg);
	//imshow("", img_jpeg); cvWaitKey();
	delete JPEGimageStruct;
}