#ifndef SPARSE_SOLVER_H
#define SPARSE_SOLVER_H

#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"
#include "mkl_blas.h"

//#include "engine.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;


#define USE_OPENMP					1
#if USE_OPENMP
#define PARAM_CORES					4
#else
#define PARAM_CORES					1
#endif

void sparse_check(int rowNum, int* rowIndex, int* columns, bool isOneBased = true);

void sparse_ATA(int rowNum, int eleNum, int& nonZeroNum, double* A, int* rowIndex, int* columns, double* CA, int* CrowIndex, int* Ccolumns);

void sparse_ATb(int rowNum, int eleNum, double* b, double* CA, int* CrowIndex, int* Ccolumns, double* Cb);

void copy_data(Mat& resMat, double* resData0, double* resData1, double* resData2, int width, int height);

void copy_data(Mat& aresMat, Mat& bresMat, double* resData, int channel, int width, int height);

void solve_direct_cpu(Mat& aRes, Mat& bRes,
	int nonZeroNum, int eleNum, int oneBased,
	double* A, int* rowIndex, int* columns,
	double* Ba0, double* Xa0,
	double* Ba1, double* Xa1,
	double* Ba2, double* Xa2,
	double* Bb0, double* Xb0,
	double* Bb1, double* Xb1,
	double* Bb2, double* Xb2);

void solve_direct_cpu(Mat& aRes, Mat& bRes,
	int nonZeroNum, int eleNum, int oneBased,
	double* A0, int* rowIndex, int* columns,
	double* Bab0, double* Xab0, int channel);

void solve_direct_cpu(Mat& res,
	int nonZeroNum, int eleNum, int oneBased,
	double* A0, int* rowIndex, int* columns,
	double* Bab0, double* Xab0, int channel);



#endif