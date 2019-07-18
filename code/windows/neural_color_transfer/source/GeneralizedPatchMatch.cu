

#include "GeneralizedPatchmatch.cuh"
#include "curand_kernel.h"
#include "fstream"
#include "device_functions.h"
#include "ColorTransfer\Config.h"

__host__ __device__ int clamp(int x, int x_max, int x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 12 bits, max = 4095
	return (((y) << 12) | (x));
}

__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 12) - 1);
}

__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 12)&((1 << 12) - 1);
}

__host__ __device__ int cuMax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}

__host__ __device__ int cuMin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ float MycuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	 return curand_uniform(&state);

}

__device__ void InitcuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(i, 0, 0, &state);

}

__host__ Mat reconstruct_avg(Mat a, Mat b, unsigned int * ann, int patch_w) {

	Mat c;
	a.copyTo(c);
	int leftSize = -patch_w / 2;
	int rightSize = patch_w + leftSize - 1;
	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++)
		{
		
			float point_num = 0, *dist_tmp;
			
			dist_tmp = new float[3];

			dist_tmp[0] = 0;
			dist_tmp[1] = 0;
			dist_tmp[2] = 0;

			for (int dx = leftSize; dx <= rightSize; dx++) {
				for (int dy = leftSize; dy <= rightSize; dy++)
				{
					if ((ax + dx) < a.cols && (ax + dx) >= 0 && (ay + dy) < a.rows && (ay + dy) >= 0)
					{
						unsigned int vp = ann[(ay + dy)*a.cols + ax + dx];
						int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);
				
						int newdx = -dx, newdy = -dy;
						
						if ((xp + newdx) < b.cols && (xp + newdx) >= 0 && (yp + newdy) < b.rows && (yp + newdy) >= 0)//a patch that contain this pixel
						{
							const Vec3b& bv = b.at<Vec3b>(yp + newdy, xp + newdx);
							dist_tmp[0] += bv[0];
							dist_tmp[1] += bv[1];
							dist_tmp[2] += bv[2];

							point_num++;
						}
					}

				}

			}

			Vec3b& cv = c.at<Vec3b>(ay, ax);
			cv[0] = dist_tmp[0] / point_num;
			cv[1] = dist_tmp[1] / point_num;
			cv[2] = dist_tmp[2] / point_num;
			
			delete[] dist_tmp;
		}
	}
	return c;
}

__host__ Mat reconstruct_bds(Mat a, Mat b, unsigned int* ann, unsigned int * bnn, int patch_w, double wCohen, double wComplete) {

	Mat aRes = Mat::zeros(a.size(), CV_32SC3);
	Mat bRes = Mat::zeros(a.size(), CV_32SC3);

	Mat aWgt = Mat::zeros(a.size(), CV_32SC1);
	Mat bWgt = Mat::zeros(a.size(), CV_32SC1);

	Mat cRes = Mat::zeros(a.size(), CV_8UC3);

	int leftSize = -patch_w / 2;
	int rightSize = patch_w + leftSize - 1;

	double wa = wCohen / double(a.cols * a.rows);
	double wb = wComplete / double(b.cols * b.rows);
	int col_tmp[3];

	for (int ay = 0; ay < a.rows; ay++)
	{
		for (int ax = 0; ax < a.cols; ax++)
		{
			int wgt_cnt = 0;
			double dist_sum = 0;
			col_tmp[0] = 0;
			col_tmp[1] = 0;
			col_tmp[2] = 0;

			for (int dx = leftSize; dx <= rightSize; dx++)
			{
				for (int dy = leftSize; dy <= rightSize; dy++)
				{
					if ((ax + dx) < a.cols && (ax + dx) >= 0 && (ay + dy) < a.rows && (ay + dy) >= 0)
					{
						unsigned int vp = ann[(ay + dy)*a.cols + ax + dx];
						int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);

						int newdx = -dx, newdy = -dy;
						if ((xp + newdx) < b.cols && (xp + newdx) >= 0 && (yp + newdy) < b.rows && (yp + newdy) >= 0)//a patch that contain this pixel
						{
							const Vec3b& bv = b.at<Vec3b>(yp + newdy, xp + newdx);
							col_tmp[0] += bv[0];
							col_tmp[1] += bv[1];
							col_tmp[2] += bv[2];

							wgt_cnt++;
						}
					}

				}

			}


			Vec3i& ar = aRes.at<Vec3i>(ay, ax);
			ar[0] += col_tmp[0];
			ar[1] += col_tmp[1];
			ar[2] += col_tmp[2];
			aWgt.at<int>(ay, ax) += wgt_cnt;
		}
	}

	for (int by = 0; by < b.rows; by++)
	{
		for (int bx = 0; bx < b.cols; bx++)
		{
			unsigned int vp = bnn[by * b.cols + bx];
			int xp = INT_TO_X(vp);
			int yp = INT_TO_Y(vp);
			int bid = by * b.cols + bx;

			for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++)
			{
				for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
				{
					if ((bx + dx) < b.cols && (bx + dx) >= 0 && (by + dy) < b.rows && (by + dy) >= 0)
					{
						if ((xp + dx) < a.cols && (xp + dx) >= 0 && (yp + dy) < a.rows && (yp + dy) >= 0)
						{
							int ax = xp + dx;
							int ay = yp + dy;

							const Vec3b& bv = b.at<Vec3b>(by + dy, bx + dx);
							Vec3i& br = bRes.at<Vec3i>(ay, ax);
							br[0] += bv[0];
							br[1] += bv[1];
							br[2] += bv[2];

							bWgt.at<int>(ay, ax) += 1;
						}
					}
				}
			}
		}
	}

	for (int ay = 0; ay < a.rows; ay++)
	{
		for (int ax = 0; ax < a.cols; ax++)
		{
			const Vec3i& av = aRes.at<Vec3i>(ay, ax);
			const Vec3i& bv = bRes.at<Vec3i>(ay, ax);

			const double aw = aWgt.at<int>(ay, ax) * wa;
			const double bw = bWgt.at<int>(ay, ax) * wb;

			Vec3b& c = cRes.at<Vec3b>(ay, ax);

			c[0] = double(av[0] * wa + bv[0] * wb) / double(aw + bw);
			c[1] = double(av[1] * wa + bv[1] * wb) / double(aw + bw);
			c[2] = double(av[2] * wa + bv[2] * wb) / double(aw + bw);
		}
	}
	return cRes;
}

__host__ void norm(float* &dst, const float* src, float* smooth, Dim dim){

	int count = dim.channel*dim.height*dim.width;
	const float* x = src;
	float* x2;
	cudaMalloc(&x2, count*sizeof(float));
	caffe_gpu_mul(count, x, x, x2);

	//caculate dis
	float*sum;
	float* ones;
	cudaMalloc(&sum, dim.height*dim.width*sizeof(float));
	cudaMalloc(&ones, dim.channel*sizeof(float));
	caffe_gpu_set(dim.channel, 1.0f, ones);
	caffe_gpu_gemv(CblasTrans, dim.channel, dim.height*dim.width, 1.0f, x2, ones, 0.0f, sum);

	float *dis;
	cudaMalloc(&dis, dim.height*dim.width*sizeof(float));
	caffe_gpu_powx(dim.height*dim.width, sum, 0.5f, dis);

	if (smooth != NULL)
	{
		// HMM@NOTE: change sum to dis
		cudaMemcpy(smooth, dis, dim.height*dim.width*sizeof(float), cudaMemcpyDeviceToDevice);
		int index;
		float minv, maxv;
		cublasIsamin(Caffe::cublas_handle(), dim.height*dim.width, dis, 1, &index);
		cudaMemcpy(&minv, dis + index - 1, sizeof(float), cudaMemcpyDeviceToHost);
		cublasIsamax(Caffe::cublas_handle(), dim.height*dim.width, dis, 1, &index);
		cudaMemcpy(&maxv, dis + index - 1, sizeof(float), cudaMemcpyDeviceToHost);

		//printf("HMM: Response Min = %f, Max = %f\n", minv, maxv);

		caffe_gpu_add_scalar(dim.height*dim.width, -minv, smooth);
		caffe_gpu_scal(dim.height*dim.width, 1.0f / (maxv - minv), smooth);
	}


	//norm	
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, dim.channel, dim.width*dim.height, 1, 1.0f, ones, dis, 0.0f, x2);
	caffe_gpu_div(count, src, x2, dst);

	cudaFree(x2);
	cudaFree(ones);
	cudaFree(dis);
	cudaFree(sum);
}

__host__ void norm1(float* &dst, const float* src, float* smooth, Dim dim){

	int count = dim.channel*dim.height*dim.width;
	const float* x = src;
	float* x2;
	cudaMalloc(&x2, count*sizeof(float));
	caffe_gpu_mul(count, x, x, x2);

	//caculate dis
	float*sum;
	float* ones;
	cudaMalloc(&sum, dim.height*dim.width*sizeof(float));
	cudaMalloc(&ones, dim.channel*sizeof(float));
	caffe_gpu_set(dim.channel, 1.0f, ones);
	caffe_gpu_gemv(CblasTrans, dim.channel, dim.height*dim.width, 1.0f, x2, ones, 0.0f, sum);

	float *dis;
	cudaMalloc(&dis, dim.height*dim.width*sizeof(float));
	caffe_gpu_powx(dim.height*dim.width, sum, 0.5f, dis);

	if (smooth != NULL)
	{
		// HMM@NOTE: change sum to dis
		cudaMemcpy(smooth, dis, dim.height*dim.width*sizeof(float), cudaMemcpyDeviceToDevice);
		int index;
		float minv, maxv;
		cublasIsamin(Caffe::cublas_handle(), dim.height*dim.width, dis, 1, &index);
		cudaMemcpy(&minv, dis + index - 1, sizeof(float), cudaMemcpyDeviceToHost);
		cublasIsamax(Caffe::cublas_handle(), dim.height*dim.width, dis, 1, &index);
		cudaMemcpy(&maxv, dis + index - 1, sizeof(float), cudaMemcpyDeviceToHost);

		caffe_gpu_add_scalar(dim.height*dim.width, -minv, smooth);
		caffe_gpu_scal(dim.height*dim.width, 1.0f / (maxv - minv), smooth);
	}


	cudaFree(x2);
	cudaFree(ones);
	cudaFree(dis);
	cudaFree(sum);

	int max_id;
	cublasIsamax(Caffe::cublas_handle(), count, src, 1, &max_id);

	float value;
	cudaMemcpy(&value, src + (max_id - 1), sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(dst, src, count*sizeof(float), cudaMemcpyDeviceToDevice);
	caffe_gpu_scal(count, 1.0f / value, dst);

}

__host__ Mat reconstruct_flow(Mat a, Mat b, unsigned int * ann, int patch_w) {
	Mat flow;
	a.copyTo(flow);
	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++)
		{
			unsigned int v = ann[ay*a.cols + ax];
			int xbest = INT_TO_X(v);
			int ybest = INT_TO_Y(v);

			flow.at<Vec3b>(ay, ax).val[0] = (uchar)(255 * ((float)xbest / b.cols));
			flow.at<Vec3b>(ay, ax).val[1] = 0;
			flow.at<Vec3b>(ay, ax).val[2] = (uchar)(255 * ((float)ybest / b.rows));
		}
	}
	return flow;
}

__host__ __device__ float dist_compute_single(float * a1, float * b1, float weight, int channels, int a_rows, int a_cols, int b_rows, int b_cols, 
	int ax, int ay, int bx, int by, int patch_w, float cutoff = INT_MAX) 
{
	//this is the average number of all matched pixel
	//suppose patch_w is an odd number
	float pixel_sum = 0, pixel_no = 0, pixel_dist = 0;//number of pixels realy counted
	float pixel_sum1 = 0;
	int a_slice = a_rows*a_cols, b_slice = b_rows*b_cols;
	int a_pitch = a_cols, b_pitch = b_cols;
	float dp_tmp;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			int newdx = dx, newdy = dy;
			
			if ((ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + newdy) < b_rows && (by + newdy) >= 0 && (bx + newdx) < b_cols && (bx + newdx) >= 0)
				//the pixel in a should exist and pixel in b should exist
			{

				for (int dc = 0; dc < channels; dc++)
				{
					dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b1[dc * b_slice + (by + newdy) * b_pitch + (bx + newdx)];
					pixel_sum1 -= dp_tmp;
				}

				pixel_no += 1;
			}
		}

	}

	if (pixel_no == 0)
	{
		pixel_dist = 1;
	}
	else
	{
		pixel_dist = (pixel_sum + weight*pixel_sum1) / pixel_no;
	}
	if (pixel_dist >= cutoff) 
	{ 
		return cutoff; 
	}
	else 
	{
		return pixel_dist;
	}
}

__host__ __device__ float dist_compute(float * a, float * b, float * a1, float * b1, float weight, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int br, int patch_w, float cutoff = INT_MAX) {//this is the average number of all matched pixel
																																																		  //suppose patch_w is an odd number
	float pixel_sum = 0, pixel_no = 0, pixel_dist = 0;//number of pixels realy counted
	float pixel_sum1 = 0;
	int a_slice = a_rows*a_cols, b_slice = b_rows*b_cols;
	int a_pitch = a_cols, b_pitch = b_cols;
	float dp_tmp;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			int newdx = dx, newdy = dy;
			
			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + newdy) < b_rows && (by + newdy) >= 0 && (bx + newdx) < b_cols && (bx + newdx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{
				if (channels<=3)
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b[dc * b_slice + (by + newdy) * b_pitch + (bx + newdx)];
						pixel_sum += dp_tmp * dp_tmp;
						dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b1[dc * b_slice + (by + newdy) * b_pitch + (bx + newdx)];
						pixel_sum1 += dp_tmp * dp_tmp;

					}
				}
				else
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b[dc * b_slice + (by + newdy) * b_pitch + (bx + newdx)];
						pixel_sum -= dp_tmp;
						dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b1[dc * b_slice + (by + newdy) * b_pitch + (bx + newdx)];
						pixel_sum1 -= dp_tmp;
					}
				}
				

				pixel_no += 1;
			}
		}
	}

	pixel_dist = (pixel_sum + weight*pixel_sum1) / pixel_no;
	if (pixel_dist >= cutoff) { return cutoff; }
	else {
		return pixel_dist;
	}
}

__host__ __device__ float dist_constraint(float * constraint, int ax, int ay, int bx, int by, int aw, int ah, int bw, int bh) {
	float ix = constraint[0 * aw*ah + ay*aw + ax];
	float iy = constraint[1 * aw*ah + ay*aw + ax];
	
	if (ix>=bw || iy >=bh)
	{
		return 0;
	}
	
	float dx, dy;
	dx = (ix - bx) / bw;
	dy = (iy - by) / bh;
	return dx*dx + dy*dy;
}

__host__ __device__ float dist_single(float *a1, float *b1, float * constraint, float weight, int channels, 
	int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int xp, int yp, int flag_constraint, int patch_w, float constraint_weight, float cutoff = INT_MAX) 
{
	if (flag_constraint == 1)
	{
		return dist_compute_single(a1, b1, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, cutoff) + 
			constraint_weight * dist_constraint(constraint, ax, ay, xp, yp, a_cols, a_rows, b_cols, b_rows);
	}
	else
	{
		return dist_compute_single(a1, b1, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, cutoff);
	}
}

__host__ __device__ float dist(float * a, float * b, float *a1, float *b1, float * constraint, float weight, int channels, 
	int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int xp, int yp, int flag_constraint, int patch_w, float constraint_weight, float cutoff = INT_MAX) 
{

	if (flag_constraint == 1)
	{
		return dist_compute(a, b, a1, b1, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, cutoff) + constraint_weight * dist_constraint(constraint, ax, ay, xp, yp, a_cols, a_rows, b_cols, b_rows);
	}
	else
	{
		return dist_compute(a, b, a1, b1, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, cutoff);
	}

}

__device__ void improve_guess_single(float *a1, float *b1, float * constraint, float weight, int channels, 
	int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w, int flag_constraint, float constraint_weight, float rr) 
{
	float d;
	d = dist_single(a1, b1, constraint, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, flag_constraint, patch_w, constraint_weight, dbest);
	if (d + rr < dbest) {
		xbest = xp;
		ybest = yp;
		dbest = d;
	}
}

__device__ void improve_guess(float * a, float * b, float *a1, float *b1, float * constraint, float weight, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w, int flag_constraint, float constraint_weight, float rr) {
	float d;
	d = dist(a, b, a1, b1, constraint, weight, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, flag_constraint, patch_w, constraint_weight, dbest);
	if (d + rr < dbest) {
		xbest = xp;
		ybest = yp;
		dbest = d;
	}
}

__global__ void init_Ann_kernel(unsigned int * ann, int * params) {

	//just use 7 of 9 parameters
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {

		int bx = min(int(float(ax) / float(aw - 1) * (bw - 1)), bw - 1);
		int by = min(int(float(ay) / float(ah - 1) * (bh - 1)), bh - 1);
		ann[ay*aw + ax] = XY_TO_INT(bx, by);
	}
}

__global__ void upSample_kernel(unsigned int * ann, unsigned int * ann_tmp, int * params, int aw_half,int ah_half) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	
	
	float aw_ratio = (float)aw / (float)aw_half;
	float ah_ratio = (float)ah / (float)ah_half;
	int ax_half = (ax+0.5) / aw_ratio;
	int ay_half = (ay+0.5) / ah_ratio;
	ax_half = clamp(ax_half, aw_half - 1, 0);
	ay_half = clamp(ay_half, ah_half - 1, 0);
	

	if (ax < aw&&ay < ah) {
		unsigned int v_half = ann[ay_half*aw_half + ax_half];
		int bx_half = INT_TO_X(v_half);
		int by_half = INT_TO_Y(v_half);

		int bx = ax + (bx_half - ax_half)*aw_ratio + 0.5;
		int by = ay + (by_half - ay_half)*ah_ratio + 0.5;

		bx = clamp(bx, bw-1, 0);
		by = clamp(by, bh-1, 0);

		ann_tmp[ay*aw + ax] = XY_TO_INT(bx, by);
	}

}

__global__ void upSample_flow(unsigned int * ann, unsigned int * ann_tmp, int * params, int aw_half, int ah_half)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	float aw_ratio = (float)aw / (float)aw_half;
	float ah_ratio = (float)ah / (float)ah_half;
	int ax_full = ax * aw_ratio + 0.5;
	int ay_full = ay * ah_ratio + 0.5;

	if (ax_full < aw && ay_full < ah) 
	{
		unsigned int v_half = ann[ay * aw_half + ax];
		int bx_half = INT_TO_X(v_half);
		int by_half = INT_TO_Y(v_half);

		int bx = ax_full + (bx_half - ax) * aw_ratio + 0.5;
		int by = ay_full + (by_half - ay) * ah_ratio + 0.5;

		bx = clamp(bx, bw - 1, 0);
		by = clamp(by, bh - 1, 0);

		ann_tmp[ay * aw_half + ax] = XY_TO_INT(bx, by);
	}

}

__global__ void downSample_kernel(unsigned int * ann, unsigned int * ann_tmp, int * params, int aw, int ah) {

	int ax_half = blockIdx.x*blockDim.x + threadIdx.x;
	int ay_half = blockIdx.y*blockDim.y + threadIdx.y;

	int ah_half = params[1];
	int aw_half = params[2];
	int bh_half = params[3];
	int bw_half = params[4];

	int aw_ratio = (float)aw / (float)aw_half + 0.5;
	int ah_ratio = (float)ah / (float)ah_half + 0.5;
	int ax0 = ax_half * aw_ratio;
	int ay0 = ay_half * ah_ratio;
	int ax1 = (ax_half + 1) * aw_ratio;
	int ay1 = (ay_half + 1) * ah_ratio;
	ax0 = clamp(ax0, aw - 1, 0);
	ay0 = clamp(ay0, ah - 1, 0);
	ax1 = clamp(ax1, aw - 1, 0);
	ay1 = clamp(ay1, ah - 1, 0);

	int y = ay0;
	int x = 0;
	float fx = 0.f;
	float fy = 0.f;
	int cnt = 0;

	for (; y < ay1; ++y)
	{
		for (x = ax0; x < ax1; ++x)
		{
			unsigned int v = ann[y *aw + x];
			int bx = INT_TO_X(v);
			int by = INT_TO_Y(v);
			
			fx += (bx - x) / aw_ratio;
			fy += (by - y) / ah_ratio;
			cnt++;
		}
	}

	if (cnt > 0)
	{
		int bx_half = ax_half + fx / (float)cnt + 0.5;
		int by_half = ay_half + fy / (float)cnt + 0.5;

		bx_half = clamp(bx_half, bw_half - 1, 0);
		by_half = clamp(by_half, bh_half - 1, 0);

		ann_tmp[ay_half * aw_half + ax_half] = XY_TO_INT(bx_half, by_half);
	}
}

__global__ void setIndexData(unsigned int* data, int id, int aw, int ah)
{
	int ax = blockIdx.x * blockDim.x + threadIdx.x;
	int ay = blockIdx.y * blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {
		data[ay * aw + ax] = id;
	}
}

__global__ void patchmatch_single(float *a1, float *b1, float * constraint, unsigned int *ann, float *annd, int * params) 
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];
	int flag_constraint = params[8];
	float constraint_weight = params[9];
	float energy_weight = params[10];

	if (ax < a_cols && ay < a_rows) {

		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, xbest, ybest;

		int newjumpx, newjumpy;

		int xmin, xmax, ymin, ymax, rmin, rmax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		annd[ay*a_cols + ax] = dist_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, flag_constraint, patch_w, constraint_weight);

		for (int iter = 0; iter < pm_iters; iter++) {

			/* Current (best) guess. */
			v = ann[ay*a_cols + ax];
			xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			dbest = annd[ay*a_cols + ax];

			/* In each iteration, improve the NNF, by jumping flooding. */
			for (int jump = 8; jump > 0; jump /= 2) {

				/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
				if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
				{
					vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

					newjumpx = jump;
					newjumpy = 0;
					
					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;//the propagated match from vp, the center of the patch, which should be in the image

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, 0);
					}
				}
				ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
				annd[ay*a_cols + ax] = dbest;

				if ((ax + jump) < a_cols)//right
				{
					vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

					newjumpx = -jump;
					newjumpy = 0;
					
					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}



				if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
				{
					vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b

					newjumpx = 0;
					newjumpy = jump;
					
					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{

						//improve guess
						improve_guess_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay + jump) < a_rows)//down
				{
					vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b

					newjumpx = 0;
					newjumpy = -jump;
					
					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}


				//__syncthreads();

			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > cuMax(b_cols, b_rows)) {
				rs_start = cuMax(b_cols, b_rows);
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
				ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
				
				xp = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
				yp = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);
				
				//improve guess
				improve_guess_single(a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, FLT_MIN);

			}

			ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*a_cols + ax] = dbest;
			
			
			// HMM@HACK
			//__syncthreads();
		}
	}
}

__global__ void feature_distance(float *a1, float *b1, float *annd, int * params)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	
	if (ax < a_cols && ay < a_rows) {
		int c = 0;
		int id = ay*a_cols + ax;
		int slice = a_cols * a_rows;
		annd[id] = 0;

		for (; c < ch; ++c)
		{
			annd[id] -= a1[c * slice + id] * b1[c * slice + id];
		}

	}
}

__global__ void patchmatch(float * a, float * b, float *a1, float *b1, float * constraint, unsigned int *ann, float *annd, int * params) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];
	int flag_constraint = params[8];
	float constraint_weight = params[9];
	float energy_weight = params[10];

	if (ax < a_cols && ay < a_rows) {
	
		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, rbest = 0, xbest, ybest;

		int newjumpx, newjumpy;

		int xmin, xmax, ymin, ymax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		annd[ay*a_cols + ax] = dist(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, flag_constraint, patch_w, constraint_weight);

		for (int iter = 0; iter < pm_iters; iter++) {

			/* Current (best) guess. */
			v = ann[ay*a_cols + ax];
			xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			dbest = annd[ay*a_cols + ax];

			/* In each iteration, improve the NNF, by jumping flooding. */
			for (int jump = 8; jump > 0; jump /= 2) {

				/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
				if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
				{
					vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

					newjumpx = jump;
					newjumpy = 0;

					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;//the propagated match from vp, the center of the patch, which should be in the image

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight,0);
					}
				}
				ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
				annd[ay*a_cols + ax] = dbest;

				if ((ax + jump) < a_cols)//right
				{
					vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

					newjumpx = -jump;
					newjumpy = 0;

					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight,0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				

				if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
				{
					vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b

					newjumpx = 0;
					newjumpy = jump;

					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight,0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				

				if ((ay + jump) < a_rows)//down
				{
					vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b

					newjumpx = 0;
					newjumpy = -jump;

					xp = INT_TO_X(vp) + newjumpx, yp = INT_TO_Y(vp) + newjumpy;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				
				//__syncthreads();

			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > cuMax(b_cols, b_rows)) {
				rs_start = cuMax(b_cols, b_rows);
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
				ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
				xp = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
				yp = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);
				
				//improve guess
				improve_guess(a, b, a1, b1, constraint, energy_weight, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, flag_constraint, constraint_weight,FLT_MIN);

			}

			ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*a_cols + ax] = dbest;
			
			// HMM@HACK
			//__syncthreads();
		}
	}
}

__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params) {

	int ax = blockIdx.x * blockDim.x + threadIdx.x;
	int ay = blockIdx.y * blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	int count = 0;

	if (ax < aw&&ay < ah)
	{
		//set zero for all the channels at (ax,ay)
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] = 0;
		}

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{

				if ((ax + dx) < aw && (ax + dx) >= 0 && (ay + dy) < ah && (ay + dy) >= 0)
				{
					unsigned int vp = ann[(ay + dy)*aw + ax + dx];
					int newdx = -dx, newdy = -dy;

					int xp = INT_TO_X(vp);
					int yp = INT_TO_Y(vp);

					if ((xp + newdx) < bw && (xp + newdx) >= 0 && (yp + newdy) < bh && (yp + newdy) >= 0)
					{
						count++;
						for (int dc = 0; dc < ch; dc++)
						{
							pc[dc*slice_a + ay*pitch_a + ax] += pb[dc*slice_b + (yp + newdy)*pitch_b + xp + newdx];
						}
					}
				}

			}
		}

		//count average value
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] /= count;
		}

	}
}

__global__ void avg_vote_bds_a(unsigned int * ann, float * pin, float * pout, float* pw, int * params, float wCohen)
{
	int ax = blockIdx.x * blockDim.x + threadIdx.x;
	int ay = blockIdx.y * blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	double wa = wCohen / double(aw * ah);
	
	if (ax < aw && ay < ah)
	{
		int aid = ay * aw + ax;

		//set zero for all the channels at (ax,ay)
		for (int i = 0; i < ch; i++)
		{
			pout[i * slice_a + aid] = 0;
		}

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) 
		{
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{

				if ((ax + dx) < aw && (ax + dx) >= 0 && (ay + dy) < ah && (ay + dy) >= 0)
				{
					unsigned int vp = ann[(ay + dy)*aw + ax + dx];
					int xp = INT_TO_X(vp) - dx;
					int yp = INT_TO_Y(vp) - dy;
					
					if (xp < bw && xp >= 0 && yp < bh && yp >= 0)
					{
						pw[aid] += wa;
						for (int dc = 0; dc < ch; dc++)
						{
							pout[dc * slice_a + aid] += pin[dc*slice_b + yp * pitch_b + xp] * wa;
						}
					}
				}
			}
		}
	}
}

__global__ void avg_vote_bds_b(unsigned int * bnn, float * pin, float * pout, float* pw, int * params, float wComplete)
{
	int bx = blockIdx.x * blockDim.x + threadIdx.x;
	int by = blockIdx.y * blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int slice_b = bh * bw;

	double wb = wComplete / double(bw * bh);

	if (bx < bw && by < bh)
	{
		unsigned int vp = bnn[by * bw + bx];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++)
		{
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{
				int xb = bx + dx;
				int yb = by + dy;
				if (xb < bw && xb>= 0 && yb < bh && yb >= 0)
				{
					int xa = xp + dx;
					int ya = yp + dy;
					
					if (xa < aw && xa >= 0 && ya < ah && ya >= 0)
					{
						int aid = ya * aw + xa;
						int bid = yb * bw + xb;

						atomicAdd(&(pw[aid]), wb);
						for (int dc = 0; dc < ch; dc++)
						{
							atomicAdd(&(pout[dc * slice_a + aid]), wb * pin[dc * slice_b + bid]);
						}
					}
				}
			}
		}
	}
}

__global__ void avg_vote_bds(float * pout, float* pw, int * params)
{
	int ax = blockIdx.x * blockDim.x + threadIdx.x;
	int ay = blockIdx.y * blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];

	int slice_a = ah * aw;

	if (ax < aw && ay < ah)
	{
		int aid = ay * aw + ax;
		if (pw[aid] > 0)
		{
			for (int dc = 0; dc < ch; dc++)
			{
				pout[dc * slice_a + aid] /= pw[aid];
			}
		}
	}
}
