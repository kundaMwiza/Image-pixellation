#include <omp.h>
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>


#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include <vector_functions.h>

#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"
#include <thrust/reduce.h>
																					

#define FAILURE 0
#define SUCCESS !FAILURE

// replace with your user name
#define USER_NAME "acp18mlk"														

// structure for the pixels - 3 unsigned characters 			
struct pixels {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

/* structure for the image data
stores width, height, maxcolour, average pixel values 
and pixel data in struct pixels * imgpix
*/
struct img {
	char rmode[6];
	unsigned int w;
	unsigned int h;
	unsigned int maxc;
	unsigned int r_a;
	unsigned int g_a;
	unsigned int b_a;
	struct pixels * imgpix;
};

/*
Structure of arrays 
*/

struct img_cuda {
	unsigned char * r;
	unsigned char * g;
	unsigned char * b;
};

void print_help();
int process_command_line(int argc, char *argv[]);
void write_file(struct img * img_data, FILE * fc);
void write_file_cuda(struct img_cuda * img_data, FILE *fc, unsigned int width, unsigned int height);
int read_file(struct img * img_data, FILE * fp);
void mosaic_filter_cpu(struct img * img_data, struct img* output, int c);
void mosaic_filter_MP(struct img * img_data, struct img* output, int c);
void CUDAv1(struct img_cuda *h_output, unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, uchar4 * d_image, int c, int width, int height);
void CUDAv2(struct img_cuda *h_output, unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, uchar4 * d_image, int c, int width, int height);
void checkCUDAError(const char *msg);


typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef enum FORMAT {PLAINTEXT, BINARY} FORMAT;
unsigned int c = 0;
MODE execution_mode = CPU;
char * input_name;
char * output_name;
char * option_param;
FORMAT img_format = BINARY;


__constant__ unsigned int device_c;
__constant__ unsigned int device_width;
__constant__ unsigned int device_height;
__constant__ unsigned int mosaics_width;

/*--------------------------------One thread per block--------------------------------*/

__global__ void CUDAv1(unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, uchar4 * d_image) {
	
	// shared variables
	__shared__ int c, width, height;

	c = device_c;
	width = device_width;
	height = device_height;

	/* 
	map each thread to the start position of a mosaic cell 
	thread 0 in block 0 -> (0,0)
	thread 1 in block 0 -> (0,c)
	thread 2 in block 0 -> (0,2c)
	Note: only 64/c mosaic in a block
	*/

	int x = blockIdx.x*64*c + threadIdx.x*c;
	int y = blockIdx.y*c;
	int location;

	// rgb pixel to load data from d_image
	uchar4 pixel;
	// float4 pixel to accumulate the average in a mosaic cell
	float4 average = make_float4(0, 0, 0, 0);

	/* 
	parameters to specify the limits of a mosaic cell
	lowh -> start position of a mosaic cell in the y axis
	highh -> end position of a mosaic cell in the y axis
	loww, highw defined similarly for the x axis
	*/

	int highh, lowh, highw, loww, mosaic_size;

	highh = y + c;
	lowh = y;
	if (highh > height) {
		highh = height;
	}
	
	highw = x + c;
	loww = x;
	if (highw > width) {
		highw = width;
	}

	//size of mosaic cell
	mosaic_size = (highh - lowh)*(highw - loww);

	/*
	check if the start position of a threads mosaic cell is out of bounds 
	inner for loop processes a mosaic cell
	*/

	if (x < width) {
		for (int i = lowh; i < highh; i++) {
			for (int j = loww; j < highw; j++) {
				location = (i * width + j);
				pixel = d_image[location];
				average.x += pixel.x;
				average.y += pixel.y;
				average.z += pixel.z;
			}
		}


		//calculate average of mosaic cell;
		average.x /= (float)(mosaic_size);
		average.y /= (float)(mosaic_size);
		average.z /= (float)(mosaic_size);

		// write average of mosaic cell to output arrays r_d, g_d, b_d
		for (int i = loww; i < highw; i++) {
			for (int j = lowh; j < highh; j++) {
				location = (j * width + i);
				r_d[location] = (unsigned char)average.x;
				g_d[location] = (unsigned char)average.y;
				b_d[location] = (unsigned char)average.z;
			}
		}
	}
}


/*--------------------------------c threads per mosaic c > 64--------------------------------*/
__global__ void CUDAv2( uchar4 * d_image, float4 * mosaic_reduce) {

	// declare a shared memory array for block level reduction
	extern __shared__ float4 block_data[];
	
	// shared variables
	__shared__ int c, width, height, size, cells_width;

	c = device_c;
	width = device_width;
	height = device_height;
	cells_width = mosaics_width;

	// rgb pixel to load data from d_image
	uchar4 pixel;
	// float4 pixel to accumulate the average in a mosaic cell
	float4 avg_pixel = make_float4(0, 0, 0, 0);

	/*
	
	(*)
	(x,y) maps a thread to its position within the first row of a mosaic cell e.g. 
	thread 0 in block 0 -> mapped to the 0th index in the first row of the first c x c mosaic cell 
	thread 1 in block 0 -> mapped to the 1st index in the first row of the first c x c mosaic cell 
	...
	thread 0 in block 1 -> mapped to the 0th index in the first row of the second c x c mosaic cell
	thread 1 in block 1 -> mapped to the 1st index in the first row of the second c x c mosaic cell

	location -> position of pixel in d_image
	location2 -> position of mosaic cell in 1 dimensional array
	
	*/
	int x, y, location, location2, divider, greater_than;

	divider = c / 1024;
	greater_than = (c > 1024) ? 1 : 0;

	/*

	(**)
	given c > 1024, each mosaic cell is split into c**2/1024**2 sections.
	each section is assigned a blockIdx.z, therefore, blocks contributing to the same mosaic cell
	are uniquely identified by blockidx.x and blockidx.y 

	divider -> used to offset blocks contributing to the same mosaic cell in the y axis.
	For example, if c = 2048, there are 4 blocks and blockidx.z = 2 and 3 will be positioned on the 
	second "row" of the mosaic cell. (blockidx.z /divider) * 1024 offsets the starting y axis value for these blocks.
	similarly, first "row" blocks of a mosaic cell are offset in the x axis by (blockidx.z%divider)*1024

	if c < 1024 each thread within a block is simply mapped using the scheme given above (*)

	*/

	if (greater_than == 1) {
		x = c * blockIdx.x + threadIdx.x + (blockIdx.z % divider) * 1024;
		y = c * blockIdx.y + (blockIdx.z / divider) * 1024;
	}
	else {
		x = c * blockIdx.x + threadIdx.x;
		y = blockIdx.y * c;
	}

	/*
	The code below is used to compute the size of a mosaic cell 
	each control flow gives the size of a mosaic cell based on its location in the image
	*/

	if ((blockIdx.x == gridDim.x - 1) && (blockIdx.y == gridDim.y - 1))
	{
		size = (width - c * blockIdx.x)*(height - c * blockIdx.y);
	}
	else if (blockIdx.x == gridDim.x - 1) {
		size = (width - c * blockIdx.x)*c;
	}
	else if (blockIdx.y == gridDim.y - 1) {
		size = (height - c * blockIdx.y)*c;
	}
	else {
		size = c * c;
	}

	/*
	since each thread processes a column of a mosaic cell, the "upper" limit of the mosaic cell 
	needs to be defined to avoid reading out of bounds
	if c > 1024, the upper limit is either the image size or the y axis value generated in (**), offset by 1024
	if c <= 1024, the upper limit is either the image size or the y axis value generated in (**), offset by c
	*/

	int upper_lim;
	upper_lim = (c > 1024) ? y + 1024 : y + c;
	upper_lim = (upper_lim > height) ? height : upper_lim;

	/*
	check if thread is out of bounds (width wise)
	*/
	if (x < width) {
		// each thread processes a column of a mosaic cell
		for (int i = y; i < upper_lim; i++) {
			location = i * width + x;
			pixel = d_image[location];
			avg_pixel.x += pixel.x;
			avg_pixel.y += pixel.y;
			avg_pixel.z += pixel.z;

		}
	}

	// synchronize threads for block level reduction
	block_data[threadIdx.x] = avg_pixel;
	__syncthreads();

	// block level reduction using shared memory
	int stride_val = blockDim.x / 2;
	float4 pixel_temp;
	float4 pixel_receive;
	for (int stride = stride_val; stride > 0; stride >>= 1) {
		if (threadIdx.x < stride) {
			pixel_temp = block_data[threadIdx.x + stride];
			pixel_receive = block_data[threadIdx.x];
			pixel_receive.x += pixel_temp.x;
			pixel_receive.y += pixel_temp.y;
			pixel_receive.z += pixel_temp.z;
			block_data[threadIdx.x] = pixel_receive;
		}
		__syncthreads();
	}

	/*
	thread 0 writes the block reduced value to a device array for writing by CUDAv2_scatter
	location2 is the position of a mosaic cell within a global array
	if c > 1024, multiple blocks will write to location2 -> location2 uniquely identifies 
	a mosaic cell
	*/
	if (threadIdx.x == 0) {
		location2 = cells_width * blockIdx.y + blockIdx.x;
		atomicAdd(&(mosaic_reduce[location2].x), pixel_receive.x/(float)size);
		atomicAdd(&(mosaic_reduce[location2].y), pixel_receive.y/ (float)size);
		atomicAdd(&(mosaic_reduce[location2].z), pixel_receive.z/ (float)size);
	}
}
__global__ void CUDAv2_scatter(unsigned char* r_d, unsigned char* g_d, unsigned char* b_d,  float4 *  mosaic_reduce) {
	
	/*
	Function to write output image
	Indexing specifications simiilar to CUDAv2
	*/
	__shared__ int c, width, height, cells_width;
	__shared__ float4 write_pixel;

	c = device_c;
	width = device_width;
	height = device_height;
	cells_width = mosaics_width;
	
	int x, y, location, location2, divider, greater_than, upper_lim;
	
	divider = c / 1024;
	greater_than = (c > 1024) ? 1 : 0;

	if (greater_than == 1) {
		x = c * blockIdx.x + threadIdx.x + (blockIdx.z % divider) * 1024;
		y = c * blockIdx.y + (blockIdx.z / divider) * 1024;
	}
	else {
		x = c * blockIdx.x + threadIdx.x;
		y = blockIdx.y * c;
	}

	location2 = cells_width * blockIdx.y + blockIdx.x;
	upper_lim = (c > 1024) ? y + 1024 : y + c;
	upper_lim = (upper_lim > height) ? height : upper_lim;

	/* check if thread is within bounds before writing a "column" of a mosaic cell 
	to the output image
	*/

	if (x < width) {
		write_pixel = mosaic_reduce[location2];
		float4 pixel2 = write_pixel;
		uchar4 pixel1;
		pixel1.x = (unsigned char) pixel2.x;
		pixel1.y = (unsigned char) pixel2.y;
		pixel1.z = (unsigned char) pixel2.z;

		for (int i = y; i < upper_lim; i++) {
			location = i * width + x;
			r_d[location] = pixel1.x;
			g_d[location] = pixel1.y;
			b_d[location] = pixel1.z;
		}
	}
}

/*--------------------------------c threads per mosaic c < 64--------------------------------*/

__global__ void CUDAv2a(uchar4 *d_image, float4 * mosaic_reduce) {
	
	// shared variables
	__shared__ int c, width, height, cells_width;

	c = device_c;
	width = device_width;
	height = device_height;
	cells_width = mosaics_width;
	
	
	// rgb pixel to load data from d_image
	uchar4 pixel;
	// float4 pixel to accumulate the average in a mosaic cell
	float3 avg_pixel = make_float3(0, 0, 0);


	int x, y, location, location2;

	/*
	similar to CUDAv2 however mosaic cells are "squashed" together in block
	sizes of 64. 64*blockIdx.x offsets a thread to the correct mosaic cell in the x axis
	blockIdx.y*c offsets a thread to the correct mosaic cell in the y axis
	*/

	x = 64 * blockIdx.x + threadIdx.x;
	y = blockIdx.y * c;

	unsigned mosaic_loww, mosaic_highw, mosaic_lowh, mosaic_highh;

	/*
	parameters to specify the limits of a mosaic cell within a block
	mosaic_loww -> start position of a mosaic cell in the x axis
	mosaic_highw -> end position of a mosaic cell in the x axis
	mosaic_low, mosaic_highh defined similarly for the y axis
	*/

	mosaic_loww = blockIdx.x * blockDim.x + c * (threadIdx.x / c);
	mosaic_highw = blockIdx.x * blockDim.x + c * ((threadIdx.x / c) + 1);
	mosaic_lowh = blockIdx.y*c;
	mosaic_highh = blockIdx.y*c + c;

	/*
	check if end positions (width or height) are greater than the image sizes
	correct to the image width or height respectively
	*/

	mosaic_highw = (mosaic_highw > width) ? width : mosaic_highw;
	mosaic_highh = (mosaic_highh > height) ? height : mosaic_highh;

	// size of a mosaic cell
	int size = (mosaic_highh - mosaic_lowh)*(mosaic_highw - mosaic_loww);
	int upper_lim;

	/*since each thread processes a column of a mosaic cell, the "upper" limit of the mosaic cell
	  needs to be defined to avoid reading out of bounds. The upper limit is either the image size 
	  or the y axis value offset by c
	*/
	upper_lim = y + c;
	upper_lim = (upper_lim > height) ? height : upper_lim;

	/*
	check if thread is out of bounds (width wise)
	*/

	if (x < width) {
		// each thread processes a column of a mosaic cell
		for (int i = y; i < upper_lim; i++) {
			location = i * width + x;
			pixel = d_image[location];
			avg_pixel.x += pixel.x;
			avg_pixel.y += pixel.y;
			avg_pixel.z += pixel.z;
		}
	}

	/*
	location2 uniquely identifies the location of a mosaic cell within an 
	array of size(number of mosaic cells) 
	*/

	location2 = cells_width * blockIdx.y + x / c;

	// within block warp level reduction -> width = c, stride = c/2

	for (int stride = c / 2; stride > 0; stride >>= 1) {
		avg_pixel.x += __shfl_down_sync(0xFFFFFFFF, avg_pixel.x, stride, c);
		avg_pixel.y += __shfl_down_sync(0xFFFFFFFF, avg_pixel.y, stride, c);
		avg_pixel.z += __shfl_down_sync(0xFFFFFFFF, avg_pixel.z, stride, c);
	}

	// only threads with the warp reduced values write to an array storing mosaic cell averages 

	if (x < width && x % c == 0) {
		atomicAdd(&(mosaic_reduce[location2].x), avg_pixel.x/(float)size);
		atomicAdd(&(mosaic_reduce[location2].y), avg_pixel.y/(float)size);
		atomicAdd(&(mosaic_reduce[location2].z), avg_pixel.z/(float)size);
	}
}
__global__ void CUDAv2a_scatter(unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, float4 * mosaic_reduce) {
	
	/*
	Function to write output image
	Indexing specifications simiilar to CUDAv2
	*/

	extern __shared__ float4 avg_vals[];
	__shared__ int c, width, height, cells_width;

	c = device_c;
	width = device_width;
	height = device_height;
	cells_width = mosaics_width;

	int x, y, location, location2, location3;

	x = 64 * blockIdx.x + threadIdx.x;
	y = blockIdx.y * c;

	int upper_lim;

	location3 = threadIdx.x / c;
	location2 = cells_width * blockIdx.y + x / c;
	upper_lim = y + c;
	upper_lim = (upper_lim > height) ? height : upper_lim;

	/* check if thread is within bounds before writing a "column" of a mosaic cell
	to the output image
	*/

	if (x < width) {
		uchar4 pixel1;
		avg_vals[location3] = mosaic_reduce[location2];
		float4 pixel2 = avg_vals[location3];
		pixel1.x = (unsigned char)pixel2.x;
		pixel1.y = (unsigned char)pixel2.y;
		pixel1.z = (unsigned char)pixel2.z;
		for (int i = y; i < upper_lim; i++) {
			location = i * width + x;
			r_d[location] = pixel1.x;
			g_d[location] = pixel1.y;
			b_d[location] = pixel1.z;
		}
	}
}


int main(int argc, char *argv[]) {
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	
	// host and device arrays for the image
	uchar4 * h_image, *d_image;

	// output arrays for each colour -> used to avoid bad global access patterns when writing to global memory
	unsigned char *r_d, *g_d, *b_d;

	// host structure to copy above arrays back to the host 
	struct img_cuda *h_output = NULL;

	// first pointer to image data -> for reading the data
	struct img *n01 = NULL;		
	// second pointer to image data -> for writing the output image
	struct img *n02 = NULL;															
	
	FILE * fp = NULL;
	FILE * fc = NULL;
	
	// open file for reading
	fp = fopen(input_name, "rb");
	
	// test if able to open input file in binary mode																				
	if (!fp) {
		fprintf(stderr, "Error: unable to open file '%s'\n", input_name);
		return(1);
	}
	
	//TODO: read input image file (either binary or plain text PPM) 

	// allocate memory for the first pointer
	n01 = (struct img*)malloc(sizeof(struct img));									
	// test if able to allocate memory 
	if (n01 == NULL) {
		fprintf(stderr, "Error: unable to allocate memory for the image structure.\n");
		return(1);
	}
	// test if reading image data from file is successful
	if (read_file(n01, fp) == 1){
		return 1;
	}

	// initialise variables for the pixel averages (OPENMP and CUDA)
	n01->r_a = 0;																	
	n01->g_a = 0;
	n01->b_a = 0; 

	int image_width = n01->w;
	int image_height = n01->h;

	// allocate memory for the first pointer
	n02 = (struct img*)malloc(sizeof(struct img));
	n02->imgpix = (struct pixels*)malloc(sizeof(struct pixels)*image_width*image_height);
	// test if able to allocate memory 
	if (n02 == NULL) {
		fprintf(stderr, "Error: unable to allocate memory for the image structure.\n");
		return(1);
	}

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			//TODO: starting timing here
			clock_t begin, end;
			float seconds;
			begin = clock();

			// execute mosaic function for the CPU variant
			mosaic_filter_cpu(n01, n02, c);		

			//TODO: calculate the average colour value
			unsigned char r_a = (unsigned char)(n01->r_a);
			unsigned char g_a = (unsigned char)(n01->g_a);
			unsigned char b_a = (unsigned char)(n01->b_a);

			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d \n", r_a, g_a, b_a);

			//TODO: end timing here
			end = clock();
			seconds = (end - begin) / (float)CLOCKS_PER_SEC;
			printf("CPU mode execution time took %fms\n", seconds*1000);
			break;
		}
		case (OPENMP) : {
			//TODO: starting timing here
			double begin, end, seconds;
			begin = omp_get_wtime();

			// execute mosaic function for the OpenMP variant
			mosaic_filter_MP(n01, n02, c);		

			//TODO: calculate the average colour value
			unsigned char r_a = (unsigned char)(n01->r_a);
			unsigned char g_a = (unsigned char)(n01->g_a);
			unsigned char b_a = (unsigned char)(n01->b_a);

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d green = %d blue = %d \n", r_a, g_a, b_a);

			//TODO: end timing here
			end = omp_get_wtime();
			seconds = (end - begin);
			printf("OPENMP mode execution time took %f ms\n", seconds*1000);
			break;
		}
		case (CUDA) : {
			// compute the amount of memory required to store the entire image
			int image_size = sizeof(uchar4)*image_width*image_height;			

			// compute the amount of memory required to store 1 colourway for the image e.g red
			int image_size_arr = sizeof(unsigned char)*image_width*image_height;

			// allocate memory for the output image structure from the GPU
			h_output = (struct img_cuda*)malloc(sizeof(struct img_cuda));
			cudaMallocHost((void **)&(h_output->r), image_size_arr);
			cudaMallocHost((void **)&(h_output->g), image_size_arr);
			cudaMallocHost((void **)&(h_output->b), image_size_arr);

			// check if able to allocate memory for the output image
			checkCUDAError("CUDA malloc");

			// allocate memory for the device arrays, one for each colourway
			cudaMalloc((void **)&r_d, sizeof(unsigned char)*image_width*image_height);
			cudaMalloc((void **)&g_d, sizeof(unsigned char)*image_width*image_height);
			cudaMalloc((void **)&b_d, sizeof(unsigned char)*image_width*image_height);

			// allocate a temporary host image array to move data from an array of pixel structures (see above) to an array of uchar4 
			h_image = (uchar4 *)malloc(image_size);
			// check if able to allocate memory for the temporary variable
			if (h_image == NULL) {
				fprintf(stderr, "Error: unable to allocate memory for the temporary image structure.\n");
				return(1);
			}

			// allocate memory for the image on the device
			cudaMalloc((void **)&d_image, image_size);
			// check if able to allocate memory for the output image 
			checkCUDAError("CUDA malloc");


			// map struct pixels * ptr1 -> uchar4 * ptr2
			for (unsigned int i = 0; i < n01->h; i++) {
				for (unsigned int j = 0; j < n01->w; j++) {
					h_image[i*n01->w + j].x = n01->imgpix[i*n01->w + j].r;
					h_image[i*n01->w + j].y = n01->imgpix[i*n01->w + j].g;
					h_image[i*n01->w + j].z = n01->imgpix[i*n01->w + j].b;
				}
			}			

			// copy data from the host to the device
			cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_width, &image_width, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_height, &image_height, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_c, &c, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

			// check if able to allocate memory for the output image
			checkCUDAError("CUDA memcpy host to device");

			// free unneeded pointers
			free(n01->imgpix);
			free(n01);
			free(n02->imgpix);
			free(n02);
			free(h_image);

			// if c < 8, use CUDAv1 (more efficient), otherwise use CUDAv2
			if (c < 8) {
				CUDAv1(h_output, r_d, g_d, b_d, d_image, c, image_width, image_height);

			}
			else {
				CUDAv2(h_output, r_d, g_d, b_d, d_image, c, image_width, image_height);
			}

			break;
		}
		case (ALL) : {
															
			/*--------------------------------CPU--------------------------------*/

			clock_t begin_cpu, end_cpu;
			float seconds_cpu;
			begin_cpu = clock();
			mosaic_filter_cpu(n01, n02, c);

			//TODO: calculate the average colour value
			unsigned char r_a = (unsigned char)(n01->r_a);
			unsigned char g_a = (unsigned char)(n01->g_a);
			unsigned char b_a = (unsigned char)(n01->b_a);

			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d \n", r_a, g_a, b_a);

			//TODO: end timing here
			end_cpu = clock();
			seconds_cpu = (end_cpu - begin_cpu) / (float)CLOCKS_PER_SEC;
			printf("CPU mode execution time took %fms\n", seconds_cpu * 1000);			


			/*--------------------------------OpenMP--------------------------------*/

			//TODO: starting timing here
			double begin_MP, end_MP, seconds_MP;
			begin_MP = omp_get_wtime();
			mosaic_filter_MP(n01, n02, c);
			//TODO: calculate the average colour value
			r_a = (unsigned char)(n01->r_a);
			g_a = (unsigned char)(n01->g_a);
			b_a = (unsigned char)(n01->b_a);
			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d green = %d blue = %d \n", r_a, g_a, b_a);

			//TODO: end timing here
			end_MP = omp_get_wtime();
			seconds_MP = (end_MP - begin_MP);
			printf("OPENMP mode execution time took %f ms\n", seconds_MP * 1000);

			/*--------------------------------CUDA--------------------------------*/
			
			// compute the amount of memory required to store the entire image
			int image_size = sizeof(uchar4)*image_width*image_height;

			// compute the amount of memory required to store 1 colourway for the image e.g red
			int image_size_arr = sizeof(unsigned char)*image_width*image_height;

			// allocate memory for the output image structure from the GPU
			h_output = (struct img_cuda*)malloc(sizeof(struct img_cuda));
			cudaMallocHost((void **)&(h_output->r), image_size_arr);
			cudaMallocHost((void **)&(h_output->g), image_size_arr);
			cudaMallocHost((void **)&(h_output->b), image_size_arr);

			// check if able to allocate memory for the output image
			checkCUDAError("CUDA malloc");

			// allocate memory for the device arrays, one for each colourway
			cudaMalloc((void **)&r_d, sizeof(unsigned char)*image_width*image_height);
			cudaMalloc((void **)&g_d, sizeof(unsigned char)*image_width*image_height);
			cudaMalloc((void **)&b_d, sizeof(unsigned char)*image_width*image_height);

			// allocate a temporary host image array to move data from an array of pixel structures (see above) to an array of uchar4 
			h_image = (uchar4 *)malloc(image_size);
			// check if able to allocate memory for the temporary variable
			if (h_image == NULL) {
				fprintf(stderr, "Error: unable to allocate memory for the temporary image structure.\n");
				return(1);
			}

			// allocate memory for the image on the device
			cudaMalloc((void **)&d_image, image_size);
			// check if able to allocate memory for the output image 
			checkCUDAError("CUDA malloc");


			// map struct pixels * ptr1 -> uchar4 * ptr2
			for (unsigned int i = 0; i < n01->h; i++) {
				for (unsigned int j = 0; j < n01->w; j++) {
					h_image[i*n01->w + j].x = n01->imgpix[i*n01->w + j].r;
					h_image[i*n01->w + j].y = n01->imgpix[i*n01->w + j].g;
					h_image[i*n01->w + j].z = n01->imgpix[i*n01->w + j].b;
				}
			}

			// copy data from the host to the device
			cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_width, &image_width, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_height, &image_height, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(device_c, &c, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

			// check if able to allocate memory for the output image
			checkCUDAError("CUDA memcpy host to device");

			// free unneeded pointers
			free(n01->imgpix);
			free(n01);
			free(n02->imgpix);
			free(n02);
			free(h_image);

			// if c < 8, use CUDAv1 (more efficient), otherwise use CUDAv2
			if (c < 8) {
				CUDAv1(h_output, r_d, g_d, b_d, d_image, c, image_width, image_height);

			}
			else {
				CUDAv2(h_output, r_d, g_d, b_d, d_image, c, image_width, image_height);
			}

			break;
		}
	}

	// save the output image file (from last executed mode)

	fc = fopen(output_name, "wb");

	// test if able to open output file in binary mode																				
	if (!fc) {
		fprintf(stderr, "Error: unable to open file '%s'\n", output_name);
		return(1);
	}

	// write output image file
	if ((execution_mode == CUDA) || execution_mode == ALL) {
		write_file_cuda(h_output, fc, image_width, image_height);
	}
	else {
		write_file(n02, fc);
	}
	
	fclose(fp);
	fclose(fc);	

	// free memory
	if (execution_mode == ALL || execution_mode == CUDA) {
		cudaFreeHost(h_output->r);
		cudaFreeHost(h_output->g);
		cudaFreeHost(h_output->b);
		free(h_output);
		cudaFree(r_d);
		cudaFree(g_d);
		cudaFree(b_d);
		cudaFree(d_image);	
	}
	else{
		free(n01->imgpix);
		free(n01);
		free(n02->imgpix);
		free(n02);
	}

	
	return 0;
}

void print_help(){
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		   "\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		   "\t               ALL. The mode specifies which version of the simulation\n"
		   "\t               code should execute. ALL should execute each mode in\n"
		   "\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		   "\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		   "\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		printf("%d", argc);
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	//first argument is always the executable name

	//read in the non optional command line arguments
	
	c = (unsigned int)atoi(argv[1]);
	
	double cprime = log2(c);
																						// check if c is an integer greater than zero and c = 2^n
	if ((c <= 0) || (ceil(cprime) != cprime)) {
		fprintf(stderr, "Error: the value of c must be a positive integer power of 2 and must be specified first.\n");
		return(FAILURE);
	}	
	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0) {
		execution_mode = CPU;
	}
	else if (strcmp(argv[2], "OPENMP") == 0) {
		execution_mode = OPENMP;
	}
	else if (strcmp(argv[2], "CUDA") == 0) {
		execution_mode = CUDA;
	}
	else if (strcmp(argv[2], "ALL") == 0)
	{
		execution_mode = ALL;
	}
	else {
		fprintf(stderr, "The read mode must be: CPU, OPENMP, CUDA or ALL and must be specified after the value of c.\n");
		return(FAILURE);
	}
	//TODO: read in the input image name
	input_name = argv[4];
	if ((strcmp(argv[3], "-i") != 0) || (strstr(input_name, ".ppm") == NULL)){
		fprintf(stderr, "Error: must specify input file with ppm format in the form -i file.ppm after specifying the read mode.\n");
		return(FAILURE);
	}

	//TODO: read in the output image name
	output_name = argv[6];
	if ((strcmp(argv[5], "-o") != 0) || (strstr(output_name, ".ppm") == NULL)){
		fprintf(stderr, "Error: must specify output file with ppm format in the form -o file.ppm after specifying the input file.\n");
		return(FAILURE);
	}

	//TODO: read in any optional part 3 arguments
	if (argc >= 8){
		if (argc == 9){
				if ((strcmp(argv[7],"-f") == 0)){
					if (((strcmp(argv[8], "PPM_BINARY") != 0) && (strcmp(argv[8], "PPM_PLAIN_TEXT") != 0))){
						fprintf(stderr, "Error: must specify optional parameter in the form -f PPM_BINARY or -f PPM_PLAIN_TEXT after specifying the output file.\n");
						return(FAILURE);
					}
					else{
						img_format = strcmp(argv[8], "PPM_BINARY") == 0? BINARY : PLAINTEXT;
					}
				}
			}
		else{
			fprintf(stderr, "Error: incorrect specification of program arguments. Correct usage is..\n");
			print_help();
			return(FAILURE);
		}
	}

	return SUCCESS;
}

int read_file(struct img * img_data, FILE* fp) {

	int stop = 0;
	char temp, comment;
	int no_matches;
																						// read magic number, image width, height and maximum colour
	while (stop < 4) {
		comment = getc(fp);																// check if line begins with a comment
		if (comment == '#') {
			while (comment != '\n') {
				comment = getc(fp);
			}
		}
		else {
			ungetc(comment, fp);
			if (stop == 0) {															// read in magic number
				fscanf(fp, "%c%c%c", &(img_data->rmode[0]), &(img_data->rmode[1]), &(img_data->rmode[2]));
				if ((img_data->rmode[0] != 'P') || ((img_data->rmode[1] != '3') && (img_data->rmode[1] != '6')) || (img_data->rmode[2] != '\n')) {
					fprintf(stderr, "Error: badly formatted ppm header. The first non comment string must be P6 or P3 and must follow a newline character.\n");
					return(1);
				}
				else {
					stop++;
				}
			}
			else if (stop == 1) {														// read in image width
				no_matches = fscanf(fp, "%d%c", &(img_data->w), &temp);
				if (no_matches < 2 || temp != '\n') {
					fprintf(stderr, "Error: badly formatted ppm header. Missing width or width parameter does not follow a newline character.\n");
					return(1);
				}
				else {
					stop++;
				}
			}
			else if (stop == 2) {														// read in image height
				no_matches = fscanf(fp, "%d%c", &(img_data->h), &temp);
				if (no_matches < 2 || temp != '\n') {
					fprintf(stderr, "Error: badly formatted ppm header. Missing height or height parameter does not follow a newline character.\n");
					return(1);
				}
				else {
					stop++;
				}
			}
			else if (stop == 3) {														// read in image maximum colour
				no_matches = fscanf(fp, "%d%c", &(img_data->maxc), &temp);
				if (no_matches < 2 || temp != '\n') {
					fprintf(stderr, "Error: badly formatted ppm header. Missing maxcolour or maxcolour of image does not follow a newline character.\n");
					return(1);
				}
				else {
					stop++;
				}
			}
		}
	}

	comment = getc(fp);
	if (comment == '#') {																// check if line after maxcolour begins with a comment
		fprintf(stderr, "Error: badly formatted ppm header. Cannot have comment after maximum colour specification\n");
		return(1);
	}
	ungetc(comment, fp);

	if ((img_data->w <= 0) || (img_data->h <= 0)) {										// check if image dimensions are erroneous
		fprintf(stderr, "Error: cannot have an image with dimension w %d h %d\n", img_data->w, img_data->h);
		return(1);
	}

	if ((c > img_data->w) || (c > img_data->h)) {										// check if c is greater than the width/ height of image
		fprintf(stderr, "Error: cannot have c greater than the image dimensions w %d h %d\n", img_data->w, img_data->h);
		return(1);
	}

	if (img_data->maxc != 255) {														// check if max colour is not 255
		fprintf(stderr, "Error: image maximum colour value cannot be greater than 255\n");
		return(1);
	}
																						// allocate memory for the image data
	img_data->imgpix = (struct pixels*)malloc(sizeof(struct pixels)*img_data->w*img_data->h);

	if(img_data->imgpix == NULL) {
		fprintf(stderr, "Error: unable to allocate memory for the requested image in contiguous memory.\n");
		return(1);
	}
																						// read plain text ppm image
	if (img_data->rmode[0] == 'P' && img_data->rmode[1] == '3') {
		unsigned int i, j, r, g, b, counts;
		char tab_nl;
		for (i = 0; i < img_data->h; i++) {
			for (j = 0; j < img_data->w; j++) {
				if (j == (img_data->w - 1)) {
					counts = fscanf(fp, "%d %d %d%c", &r, &g, &b, &tab_nl);				// end of row pixel must follow a newline character
					if (counts < 4 && ((tab_nl != '\n') && (feof(fp) == 0))) {
						fprintf(stderr, "Error: either:(1) the last three pixels in a row of an image must be of the form r g b\\n.\n(2) Cannot have more/less pixels in a row than the width of an image.\n(3) The number of pixels in the image do not match the image dimensions.\n");
						return(1);
					}
					img_data->imgpix[i*img_data->w + j].r = (unsigned char)r;
					img_data->imgpix[i*img_data->w + j].g = (unsigned char)g;
					img_data->imgpix[i*img_data->w + j].b = (unsigned char)b;

				}
				else {
					counts = fscanf(fp, "%d %d %d%c", &r, &g, &b, &tab_nl);				// pixel values must be of the form  r g b\\t if not end of row pixel 
					if (counts < 4 || tab_nl != '\t') {
						fprintf(stderr, "Error: either: (1) Plain text ppm pixels must be of the form r g b\\t with r g b\\n only allowable at the end of the row of an image.\n(2) Cannot have more/less pixels in a row than the width of an image.\n(3) The number of pixels in the image do not match the image dimensions. \n");
						return(1);
					}
					img_data->imgpix[i*img_data->w + j].r = (unsigned char)r;
					img_data->imgpix[i*img_data->w + j].g = (unsigned char)g;
					img_data->imgpix[i*img_data->w + j].b = (unsigned char)b;
				}
			}
		}
	}

	else if (img_data->rmode[0] == 'P' && img_data->rmode[1] == '6') {					// read binary format image
		int pixels_read = (int) fread(img_data->imgpix, sizeof(struct pixels), img_data->h*img_data->w, fp);
		if (pixels_read != img_data->h*img_data->w) {									// number of pixels read by fread must match the number of pixels in the image
			fprintf(stderr, "Error: The number of pixels in the image do not match the image dimensions.\n");
			return(1);
		}
	}

	return 0;
}
// function to write image to ppm file
void write_file(struct img * img_data, FILE *fc) {								
	// add correct magic number
	switch (img_format) {
		case(PLAINTEXT): {
			fprintf(fc, "P3\n");
			break;
		}
		case(BINARY): {
			fprintf(fc, "P6\n");
			break;
		}
	}
	// add image width/ height/ maximum colour
	fprintf(fc, "%d\n", img_data->w);
	fprintf(fc, "%d\n", img_data->h);
	fprintf(fc, "%d\n", img_data->maxc);

	// add pixel data in plain text or binary form
	switch (img_format) {
		case(PLAINTEXT): {
			unsigned int i, j;
			for (i = 0; i < img_data->h; i++) {
				for (j = 0; j < img_data->w; j++) {
					if (j == (img_data->w - 1)) {
						fprintf(fc, "%d %d %d\n", img_data->imgpix[i*img_data->w + j].r, img_data->imgpix[i*img_data->w + j].g, img_data->imgpix[i*img_data->w + j].b);
					}
					else {
						fprintf(fc, "%d %d %d\t", img_data->imgpix[i*img_data->w + j].r, img_data->imgpix[i*img_data->w + j].g, img_data->imgpix[i*img_data->w + j].b);
					}
				}
			}
			break;
		}
		case(BINARY): {
			fwrite(img_data->imgpix, sizeof(struct pixels), (img_data->w)*(img_data->h), fc);
			break;
		}
	}
}

void write_file_cuda(struct img_cuda * img_data, FILE *fc, unsigned int width, unsigned int height) {
	// add correct magic number
	switch (img_format) {
	case(PLAINTEXT): {
		fprintf(fc, "P3\n");
		break;
	}
	case(BINARY): {
		fprintf(fc, "P6\n");
		break;
	}
	}
	// add image width/ height/ maximum colour
	fprintf(fc, "%d\n", width);
	fprintf(fc, "%d\n", height);
	fprintf(fc, "%d\n", 255);

	// add pixel data in plain text or binary form
	switch (img_format) {
	case(PLAINTEXT): {
		unsigned int i, j;
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				if (j == (width - 1)) {
					fprintf(fc, "%d %d %d\n", img_data->r[i * width + j], img_data->g[i * width + j], img_data->b[i * width + j]);
				}
				else {
					fprintf(fc, "%d %d %d\t", img_data->r[i * width + j], img_data->g[i * width + j], img_data->b[i * width + j]);
				}
			}
		}
		break;
	}
	case(BINARY): {
		for (unsigned int i = 0; i < height; i++) {
			for (unsigned int j = 0; j < width; j++)
			{
				fwrite(&(img_data->r[i*width + j]), sizeof(unsigned char), 1, fc);
				fwrite(&(img_data->g[i*width + j]), sizeof(unsigned char), 1, fc);
				fwrite(&(img_data->b[i*width + j]), sizeof(unsigned char), 1, fc);
			}
		}
		break;
	}
	}
}

// mosaic pixellation function - CPU
void mosaic_filter_cpu(struct img * img_data, struct img* output, int c) {
	unsigned int i, j, k, m;
	unsigned int upperh = (unsigned int) ceil(img_data->h / (float)c);					// number of mosaic cells row wise
	unsigned int upperw = (unsigned int) ceil(img_data->w / (float)c);					// number of mosaic cells column wise
	unsigned int imgw = img_data->w;
	unsigned int imgh = img_data->h;
	unsigned int loww, highw, lowh, highh;												// variables for the mosaic cell specification (assume index starts from 1)
																						// starting row, ending row, starting column, ending column indices	
	unsigned long long r_a = 0;																// local variables for the average values
	unsigned long long g_a = 0;
	unsigned long long b_a = 0;

	for (i = 0; i < upperh; i++) {														// row index of mosaic cell 
		
		highh = i * c + c;																// set ending row and starting row indices in array
		lowh = i * c;																	// check if ending row index is greater than image height
		if (highh > imgh) {
			highh = imgh;
		}

		for (j = 0; j < upperw; j++) {													// column index of mosaic cell

			highw = j * c + c;															// set ending column and starting column indices in array
			loww = j * c;
																						// check if ending column index is greater than image width
			if (highw > imgw) {
				highw = imgw;
			}

			unsigned int r = 0;															// mosaic cell average value variables
			unsigned int g = 0;
			unsigned int b = 0;
			unsigned char r_temp, g_temp, b_temp;


			for (k = lowh; k < highh; k++) {
				for (m = loww; m < highw; m++) {
					r += img_data->imgpix[k*imgw + m].r;
					g += img_data->imgpix[k*imgw + m].g;
					b += img_data->imgpix[k*imgw + m].b;
				}
			}

			r_a += r;
			g_a += g;
			b_a += b;

			r_temp = (unsigned char)(r / ((float)((highh - lowh)*(highw - loww))));		// calculate average mosaic cell value for r, g, b
			g_temp = (unsigned char)(g / ((float)((highh - lowh)*(highw - loww))));
			b_temp = (unsigned char)(b / ((float)((highh - lowh)*(highw - loww))));

			for (k = lowh; k < highh; k++) {											// update mosaic cell values
				for (m = loww; m < highw; m++) {
					output->imgpix[k*imgw + m].r = r_temp;
					output->imgpix[k*imgw + m].g = g_temp;
					output->imgpix[k*imgw + m].b = b_temp;
				}
			}
		}
	}

	float image_size = (float) imgw * imgh;
	img_data->r_a = (unsigned char) (r_a/image_size);																// write to structure variables for the averages
	img_data->g_a = (unsigned char) (g_a/image_size);
	img_data->b_a = (unsigned char) (b_a/image_size);
}
// mosaic pixellation function - OPENMP				
void mosaic_filter_MP(struct img * img_data, struct img* output, int c) {

#pragma warning(push)
#pragma warning(disable: 4101)
	// get number of threads available
	unsigned int NUM_THREADS = omp_get_max_threads();	
	int i, j, k, m;
	int upperh = (int) ceil(img_data->h / (float)c);
	int upperw = (int) ceil(img_data->w / (float)c);
	int imgw = img_data->w;
	int imgh = img_data->h;
	int loww, highw, lowh, highh;
	unsigned char r_temp, g_temp, b_temp;
	unsigned long long r_a = 0;
	unsigned long long g_a = 0;
	unsigned long long b_a = 0;
	unsigned int r = 0;
	unsigned int g = 0;
	unsigned int b = 0;

#pragma warning(pop)
																						/* Unreferenced local variables warning
																			when building with OPENMP - OPENMP bug
																						*/


																				
																						// declare parallel region, one omp for with reduction variables
#pragma omp parallel for default(shared) num_threads(NUM_THREADS) private(i, j, k, m, r_temp, g_temp, b_temp, highh, lowh, highw, loww, r, g, b) reduction(+: r_a, g_a, b_a)
	for (i = 0; i < upperh; i++) {														//each thread executes a row of mosaic cells

		highh = i * c + c;
		lowh = i * c;
		if (highh > imgh) {
			highh = imgh;
		 }

		for (j = 0; j < upperw; j++) {

			r = 0;
			g = 0; 
			b = 0;

			highw = j * c + c;
			loww = j * c;
			if (highw > imgw) {
				highw = imgw;
			}

			for (k = lowh; k < highh; k++) {
				for (m = loww; m < highw; m++) { 
					r += img_data->imgpix[k*imgw + m].r;
					g += img_data->imgpix[k*imgw + m].g;
					b += img_data->imgpix[k*imgw + m].b;
				}
			}

			r_a += r;
			g_a += g;
			b_a += b;

			r_temp = (unsigned char)(r / ((float)((highh - lowh)*(highw - loww))));
			g_temp = (unsigned char)(g / ((float)((highh - lowh)*(highw - loww))));
			b_temp = (unsigned char)(b / ((float)((highh - lowh)*(highw - loww))));


			for (k = lowh; k < highh; k++) {
				for (m = loww; m < highw; m++) {
					output->imgpix[k*imgw + m].r = r_temp;
					output->imgpix[k*imgw + m].g = g_temp;
					output->imgpix[k*imgw + m].b = b_temp;
				}
			}
		}
	}
	float image_size = (float) imgw * imgh;
	img_data->r_a = (unsigned char)(r_a / image_size);																// write to structure variables for the averages
	img_data->g_a = (unsigned char)(g_a / image_size);
	img_data->b_a = (unsigned char)(b_a / image_size);
}



//one thread per mosaic 
void CUDAv1(struct img_cuda *h_output, unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, uchar4 * d_image, int c, int width, int height) {
	
	// variables to store times
	float ms, ms2;

	// image average variables
	unsigned long long r, g, b;

	// create CUDA event timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// x = number of mosaic cells along the width of the image
	unsigned int  x = (unsigned int) ceil(width / (float)c);
	
	// y = number of mosaic cells along the height of the image
	unsigned int  y = (unsigned int) ceil(height / (float)c);
	
	// number of blocks after "squashing" mosaic cells in larger blocks
	int num_x = (int) ceil(x / (float)64);

	// copy number of mosaic cells along the width to the device
	cudaMemcpyToSymbol(mosaics_width, &x, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

	// check memcpy to device
	checkCUDAError("CUDA memcpy host to device");

	// number of blocks per grid
	dim3 blocksPerGrid(num_x, (unsigned int) ceil(height / (float)c), 1);
	// number of threads per block
	dim3 threadsPerBlock(64, 1, 1);

	// start timing
	cudaEventRecord(start, 0);
	CUDAv1 << <blocksPerGrid, threadsPerBlock >> > (r_d, g_d, b_d, d_image);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// stop timing and synchronize
	cudaEventElapsedTime(&ms, start, stop);
	
	// check kernel launch failures
	checkCUDAError("CUDA Kernel Launch");

	// copy output arrays back to host
	cudaMemcpy(h_output->r, r_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output->g, g_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output->b, b_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

	// check memcpy failures
	checkCUDAError("CUDA memcpy Device to host");

	// start timer for thrust reduce -> global average calculation 
	cudaEventRecord(start, 0);
	r = thrust::reduce(h_output->r, h_output->r + width * height, (unsigned long long) 0);
	g = thrust::reduce(h_output->g, h_output->g + width * height, (unsigned long long) 0);
	b = thrust::reduce(h_output->b, h_output->b + width * height, (unsigned long long) 0);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// stop timer for thrust reduce
	cudaEventElapsedTime(&ms2, start, stop);

	float imagesize =  (float) width * height;
	printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", (unsigned char)(r / imagesize), (unsigned char)(g / imagesize), (unsigned char)(b / imagesize));
	printf("CUDA mode execution time took %f ms\n", ms+ms2);
}


// c threads per mosaic
void CUDAv2(struct img_cuda *h_output, unsigned char* r_d, unsigned char* g_d, unsigned char* b_d, uchar4 * d_image, int c, int width, int height) {

	// mosaic_reduce : device array to store mosaic cell averages
	float4 * mosaic_reduce;

	// declare time measuring variables
	cudaEvent_t start, stop;
	float ms, ms2;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// if c < 64 , "squash" mosaics into larger blocks else use c threads per block
	if (c < 64) {

		// factor -> number of mosaic cell that can fit within a block of size 64
		int factor = 64 / c;
		// statically define the number of threads per block 
		int tpb = 64;

		// define the number of mosaic cells along the width/ along height of the image
		int x = (int) ceil(width / (float)c);
		int y = (int) ceil(height / (float)c);

		// intialise variables for the average of the image
		unsigned long long r = 0;
		unsigned long long g = 0;
		unsigned long long b = 0;

		// allocate memory for a device array to store mosaic cell averages
		cudaMalloc((void **)&mosaic_reduce, sizeof(float4) * x*y);

		// check if memory is successfully allocated
		checkCUDAError("CUDA malloc");

		// copy variables to the host
		cudaMemcpyToSymbol(mosaics_width, &x, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

		// check if memcpy failed
		checkCUDAError("CUDA memcpy host to device");

		// squash number of mosaics along the width by factor
		x = (int) ceil(x / (float)factor);

		// blocks per grid, threads per block
		dim3 blocksPerGrid(x, y, 1);
		dim3 threadsPerBlock(tpb, 1, 1);

		// start timing kernel launch
		cudaEventRecord(start, 0);
		CUDAv2a << <blocksPerGrid, threadsPerBlock >> > (d_image, mosaic_reduce);
		CUDAv2a_scatter << <blocksPerGrid, threadsPerBlock, factor*sizeof(float4) >> > (r_d, g_d, b_d, mosaic_reduce);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);

		// check if kernel launched successfully 
		checkCUDAError("CUDA kernel launch");

		// memcpy data back to host

		cudaMemcpy(h_output->r, r_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output->g, g_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output->b, b_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

		// check if data is successfully copied 
		checkCUDAError("CUDA memcpy from device");

		// start timing thrust reduce -> image average calculation
		cudaEventRecord(start, 0);
		r = thrust::reduce(h_output->r, h_output->r + width * height, (unsigned long long) 0);
		g = thrust::reduce(h_output->g, h_output->g + width * height, (unsigned long long) 0);
		b = thrust::reduce(h_output->b, h_output->b + width * height, (unsigned long long) 0);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms2, start, stop);

		float imagesize = (float) width * height;
		printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", (unsigned char)(r/imagesize), (unsigned char)(g /imagesize), (unsigned char)(b /imagesize));
		printf("CUDA mode execution time took %f ms\n", ms+ms2);
		cudaFree(mosaic_reduce);
	}
	else {
		// check if c > 1024, if > 1024, split mosaic cell
		int z_axis = (c > 1024) ? ((c*c) / (1024 * 1024)) : 1;
		// if c > 1024, threads per block = 1024, else c 
		int tpb = (c > 1024) ? 1024 : c;

		// compute number of mosaic cells along the width of the image
		int x = (int) ceil(width / (float)c);
		// compute number of mosaic cells along the height of the image
		int y = (int) ceil(height / (float)c);

		// initialise image average variables
		unsigned long long r = 0;
		unsigned long long g = 0;
		unsigned long long b = 0;

		// allocate memory for a device array to store mosaic cell averages
		cudaMalloc((void **)&mosaic_reduce, sizeof(float4) * x*y);

		// check if cuda malloc failed 
		checkCUDAError("CUDA malloc");

		// copy variables to device
		cudaMemcpyToSymbol(mosaics_width, &x, sizeof(int), 0, cudaMemcpyHostToDevice);

		// check if memcpy failed
		checkCUDAError("CUDA memcpy host to device");


		dim3 blocksPerGrid(x, y, z_axis);
		dim3 threadsPerBlock(tpb, 1, 1);

		// start timing kernel launch
		cudaEventRecord(start, 0);
		CUDAv2 << <blocksPerGrid, threadsPerBlock, tpb * sizeof(float4) >> > (d_image, mosaic_reduce);
		CUDAv2_scatter << <blocksPerGrid, threadsPerBlock >> > (r_d, g_d, b_d, mosaic_reduce);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);

		// check if kernel launch failed
		checkCUDAError("CUDA kernel launch");

		// memcpy data from device to host

		cudaMemcpy(h_output->r, r_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output->g, g_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output->b, b_d, sizeof(unsigned char)*height*width, cudaMemcpyDeviceToHost);

		// check if memcpy failed
		checkCUDAError("CUDA memcpy from device");

		// start timing thrust reduce -> global image average
		cudaEventRecord(start, 0);
		r = thrust::reduce(h_output->r, h_output->r + width * height, (unsigned long long) 0);
		g = thrust::reduce(h_output->g, h_output->g + width * height, (unsigned long long) 0);
		b = thrust::reduce(h_output->b, h_output->b + width * height, (unsigned long long) 0);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms2, start, stop);

		float imagesize = (float) width * height;
		printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", (unsigned char)(r / imagesize), (unsigned char)(g / imagesize), (unsigned char)(b / imagesize));
		printf("CUDA mode execution time took %f ms\n", ms+ms2);
		cudaFree(mosaic_reduce);
	}
}


																						
void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}