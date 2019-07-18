#pragma once

#include "time.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "math_constants.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include "Classifier.h"

using namespace cv;
using namespace std;

__host__ __device__ unsigned int XY_TO_INT(int x, int y);

__host__ __device__ int INT_TO_X(unsigned int v);

__host__ __device__ int INT_TO_Y(unsigned int v);

__global__ void patchmatch_single(float *a1, float *b1, float * constraint, unsigned int *ann, float *annd, int * params);

__global__ void patchmatch(float * a, float * b, float *a1, float *b1, float * constraint, unsigned int *ann, float *annd, int * params);

__global__ void init_Ann_kernel(unsigned int * ann, int * params);

__global__ void feature_distance(float *a1, float *b1, float *annd, int * params);

__global__ void upSample_kernel(unsigned int * ann, unsigned int * ann_tmp, int * params, int aw_half, int ah_half);

__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params);

__global__ void avg_vote_bds_a(unsigned int * ann, float * pin, float * pout, float* pw, int * params, float wCohen);

__global__ void avg_vote_bds_b(unsigned int * bnn, float * pin, float * pout, float* pw, int * params, float wComplete);

__global__ void avg_vote_bds(float * pout, float* pw, int * params);

__host__ void norm(float* &dst, const float* src, float* smooth, Dim dim);

__host__ void norm1(float* &dst, const float* src, float* smooth, Dim dim);

__host__ Mat reconstruct_flow(Mat a, Mat b, unsigned int * ann, int patch_w);

__host__ Mat reconstruct_bds(Mat a, Mat b, unsigned int* ann, unsigned int * bnn, int patch_w, double wCohen, double wComplete);

__host__ Mat reconstruct_avg(Mat a, Mat b, unsigned int * ann, int patch_w);

