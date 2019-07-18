#include "SparseSolver_CPU.h"
#include "mkl.h"
#include "mkl_dss.h"
#include <omp.h>

void sparse_check(int rowNum, int* rowIndex, int* columns, bool isOneBased)
{
	sparse_struct* handle = new sparse_struct;
	sparse_matrix_checker_init(handle);
	handle->n = rowNum;
	handle->csr_ia = rowIndex;
	handle->csr_ja = columns;
	handle->indexing = isOneBased ? MKL_ONE_BASED : MKL_ZERO_BASED;
	handle->matrix_format = MKL_CSR;
	handle->matrix_structure = MKL_GENERAL_STRUCTURE;
	handle->message_level = MKL_PRINT;
	handle->print_style = MKL_C_STYLE;
	sparse_matrix_checker(handle);
	printf("CSR matrix checker result: %d, %d, %d\n", handle->check_result[0], handle->check_result[1], handle->check_result[2]);
	delete handle;
}

void sparse_ATA(int rowNum, int eleNum, int& nonZeroNum, double* A, int* rowIndex, int* columns, double* CA, int* CrowIndex, int* Ccolumns)
{
	char transa = 'T';

	// A^TA
	int sort = 2; // sort result only
	int info = 0;
	int request = 0;

	mkl_dcsrmultcsr(&transa, &request, &sort,
		&rowNum, &eleNum, &eleNum,
		CA, Ccolumns, CrowIndex,
		CA, Ccolumns, CrowIndex,
		A, columns, rowIndex,
		&nonZeroNum, &info);

#ifdef ENABLE_VIS
	sparse_check(eleNum, rowIndex, columns);
#endif
}

void sparse_ATb(int rowNum, int eleNum, double* b, double* CA, int* CrowIndex, int* Ccolumns, double* Cb)
{
	char transa = 'T';
	double alpha1 = 1.0;
	double beta = 0.0;
	char matdescra[6] = "G  F ";

	mkl_dcsrmv(&transa, &rowNum, &eleNum, &alpha1, matdescra, CA, Ccolumns, CrowIndex, &(CrowIndex[1]), Cb, &beta, b);
}

void copy_data(Mat& resMat, double* resData0, double* resData1, double* resData2, int width, int height)
{
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int id = y * width + x;
			resMat.at<Vec3d>(y, x) =
				Vec3d(
				resData0[id],
				resData1[id],
				resData2[id]);
		}
	}
}

void copy_data(Mat& aresMat, Mat& bresMat, double* resData, int channel, int width, int height)
{
	int sz = width * height;

#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int id = y * width + x;
			aresMat.at<Vec3d>(y, x)[channel] = resData[id];

			id += sz;
			bresMat.at<Vec3d>(y, x)[channel] = resData[id];
		}
	}
}

void copy_data(Mat& resMat, double* resData, int channel, int width, int height)
{
	int sz = width * height;

#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int id = y * width + x;
			resMat.at<Vec3b>(y, x)[channel] = max(min(int(resData[id] * 255.0), 255), 0);
		}
	}
}

void solve_direct_cpu(Mat& aRes, Mat& bRes,
	int nonZeroNum, int eleNum, int oneBased,
	double* A, int* rowIndex, int* columns,
	double* Ba0, double* Xa0,
	double* Ba1, double* Xa1,
	double* Ba2, double* Xa2,
	double* Bb0, double* Xb0,
	double* Bb1, double* Xb1,
	double* Bb2, double* Xb2)
{
	int maxThreads = omp_get_max_threads();
	printf("MKL Max Threads = %d\n", maxThreads / 2);

	mkl_set_dynamic(0);
	mkl_set_num_threads(maxThreads / 2);

	int mtype = 2; /* real and symmetric */
	/* Internal solver memory pointer pt, */
	/* 32-bit: int pt[64]; 64-bit: long int pt[64] */
	/* or void *pt[64] should be OK on both architectures */
	void *pt[64];
	/* Pardiso control parameters. */
	int iparm[64];
	int maxfct, mnum, phase, error, msglvl;
	/* Auxiliary variables. */
	int i;
	double ddum; /* Double dummy */
	int idum; /* Integer dummy. */
	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		iparm[i] = 0;
	}
	iparm[0] = 1;         /* No solver default */
	iparm[1] = 3;         /* Fill-in reordering from METIS with OpenMP*/
	iparm[3] = 22;        /* No iterative-direct algorithm */
	iparm[4] = 0;         /* No user fill-in reducing permutation */
	iparm[5] = 0;         /* Write solution into x */
	iparm[6] = 0;         /* Not in use */
	iparm[7] = 1;         /* Max numbers of iterative refinement steps */
	iparm[8] = 0;         /* Not in use */
	iparm[9] = 8;         /* Perturb the pivot elements with 1E-8 (symmetric) */
	iparm[10] = 0;        /* Use symmetric permutation and scaling MPS */
	iparm[11] = 0;        /* Conjugate transposed/transpose solve */
	iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-on (default for symmetric) */
	iparm[13] = 0;        /* Output: Number of perturbed pivots */
	iparm[14] = 0;        /* Not in use */
	iparm[15] = 0;        /* Not in use */
	iparm[16] = 0;        /* Not in use */
	iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
	iparm[18] = -1;       /* Output: Mflops for LU factorization */
	iparm[19] = 0;        /* Output: Numbers of CG Iterations */
	iparm[20] = 1;        /* Pivoting for symmetric indefinite matrices. */
	iparm[27] = 0;        /* Double Precision */
	iparm[34] = 0;        /* One-based indexing */

	maxfct = 1;           /* Maximum number of numerical factorizations. */
	mnum = 1;             /* Which factorization to use. */
	msglvl = 0;           /* Do not print statistical information in file */
	error = 0;            /* Initialize error flag */
	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only */
	/* necessary for the FIRST call of the PARDISO solver. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		pt[i] = 0;
	}
	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */

	sparse_check(eleNum, rowIndex, columns);

	phase = 11;
	int nrhs = 1;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during symbolic factorization: %d", error);
		return;
	}
	printf("\nReordering completed ... ");
	printf("\nNumber of nonzeros in factors = %d", iparm[17]);
	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	phase = 22;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during numerical factorization: %d", error);
		//exit(2);
	}
	printf("\nFactorization completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	phase = 33;
	iparm[7] = 2; /* Max numbers of iterative refinement steps. */

	if (Ba0)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Ba0, Xa0, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (Ba1)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Ba1, Xa1, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (Ba2)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Ba2, Xa2, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (Bb0)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Bb0, Xb0, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (Bb1)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Bb1, Xb1, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (Bb2)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Bb2, Xb2, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (error != 0)
	{
		printf("\nERROR during solution: %d", error);
	}
	printf("\nSolve completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Termination and release of memory. */
	/* -------------------------------------------------------------------- */
	phase = -1; /* Release internal memory. */
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	////////////////////////////////////////////////////////////////////
	copy_data(aRes, Xa0, Xa1, Xa2, aRes.cols, aRes.rows);
	copy_data(bRes, Xb0, Xb1, Xb2, bRes.cols, bRes.rows);
}

void solve_direct_cpu(Mat& aRes, Mat& bRes,
	int nonZeroNum, int eleNum, int oneBased,
	double* A0, int* rowIndex, int* columns,
	double* Bab0, double* Xab0, int channel)
{
	int maxThreads = omp_get_max_threads();
	printf("MKL Max Threads = %d\n", maxThreads / 2);

	mkl_set_dynamic(0);
	mkl_set_num_threads(maxThreads / 2);

	int mtype = 2; /* real and symmetric */
	/* Internal solver memory pointer pt, */
	/* 32-bit: int pt[64]; 64-bit: long int pt[64] */
	/* or void *pt[64] should be OK on both architectures */
	void *pt[64];
	/* Pardiso control parameters. */
	int iparm[64];
	int maxfct, mnum, phase, error, msglvl;
	/* Auxiliary variables. */
	int i;
	double ddum; /* Double dummy */
	int idum; /* Integer dummy. */
	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		iparm[i] = 0;
	}
	iparm[0] = 1;         /* No solver default */
	iparm[1] = 3;         /* Fill-in reordering from METIS with OpenMP*/
	iparm[3] = 22;        /* No iterative-direct algorithm */
	iparm[4] = 0;         /* No user fill-in reducing permutation */
	iparm[5] = 0;         /* Write solution into x */
	iparm[6] = 0;         /* Not in use */
	iparm[7] = 1;         /* Max numbers of iterative refinement steps */
	iparm[8] = 0;         /* Not in use */
	iparm[9] = 8;         /* Perturb the pivot elements with 1E-8 (symmetric) */
	iparm[10] = 0;        /* Use symmetric permutation and scaling MPS */
	iparm[11] = 0;        /* Conjugate transposed/transpose solve */
	iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-on (default for symmetric) */
	iparm[13] = 0;        /* Output: Number of perturbed pivots */
	iparm[14] = 0;        /* Not in use */
	iparm[15] = 0;        /* Not in use */
	iparm[16] = 0;        /* Not in use */
	iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
	iparm[18] = -1;       /* Output: Mflops for LU factorization */
	iparm[19] = 0;        /* Output: Numbers of CG Iterations */
	iparm[20] = 1;        /* Pivoting for symmetric indefinite matrices. */
	iparm[27] = 0;        /* Double Precision */
	iparm[34] = 0;        /* One-based indexing */

	maxfct = 1;           /* Maximum number of numerical factorizations. */
	mnum = 1;             /* Which factorization to use. */
	msglvl = 0;           /* Do not print statistical information in file */
	error = 0;            /* Initialize error flag */
	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only */
	/* necessary for the FIRST call of the PARDISO solver. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		pt[i] = 0;
	}
	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */

	sparse_check(eleNum, rowIndex, columns);

	phase = 11;
	int nrhs = 1;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during symbolic factorization: %d", error);
		return;
	}
	printf("\nReordering completed ... ");
	printf("\nNumber of nonzeros in factors = %d", iparm[17]);
	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	phase = 22;
	printf("\nHMM equation no. is %d\n", iparm[29]);
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during numerical factorization: %d", error);
		printf("\nERROR equation no. is %d\n", iparm[29]);
		//exit(2);
	}
	printf("\nFactorization completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	phase = 33;
	iparm[7] = 2; /* Max numbers of iterative refinement steps. */

	if (Bab0)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Bab0, Xab0, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (error != 0)
	{
		printf("\nERROR during solution: %d", error);
	}
	printf("\nSolve completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Termination and release of memory. */
	/* -------------------------------------------------------------------- */
	phase = -1; /* Release internal memory. */
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	////////////////////////////////////////////////////////////////////
	copy_data(aRes, bRes, Xab0, channel, aRes.cols, aRes.rows);
}

void solve_direct_cpu(Mat& res,
	int nonZeroNum, int eleNum, int oneBased,
	double* A0, int* rowIndex, int* columns,
	double* Bab0, double* Xab0, int channel)
{
	int maxThreads = omp_get_max_threads();
	printf("MKL Max Threads = %d\n", maxThreads / 2);

	mkl_set_dynamic(0);
	mkl_set_num_threads(maxThreads / 2);

	int mtype = 2; /* real and symmetric */
	/* Internal solver memory pointer pt, */
	/* 32-bit: int pt[64]; 64-bit: long int pt[64] */
	/* or void *pt[64] should be OK on both architectures */
	void *pt[64];
	/* Pardiso control parameters. */
	int iparm[64];
	int maxfct, mnum, phase, error, msglvl;
	/* Auxiliary variables. */
	int i;
	double ddum; /* Double dummy */
	int idum; /* Integer dummy. */
	/* -------------------------------------------------------------------- */
	/* .. Setup Pardiso control parameters. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		iparm[i] = 0;
	}
	iparm[0] = 1;         /* No solver default */
	iparm[1] = 3;         /* Fill-in reordering from METIS with OpenMP*/
	iparm[3] = 22;        /* No iterative-direct algorithm */
	iparm[4] = 0;         /* No user fill-in reducing permutation */
	iparm[5] = 0;         /* Write solution into x */
	iparm[6] = 0;         /* Not in use */
	iparm[7] = 1;         /* Max numbers of iterative refinement steps */
	iparm[8] = 0;         /* Not in use */
	iparm[9] = 8;         /* Perturb the pivot elements with 1E-8 (symmetric) */
	iparm[10] = 0;        /* Use symmetric permutation and scaling MPS */
	iparm[11] = 0;        /* Conjugate transposed/transpose solve */
	iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-on (default for symmetric) */
	iparm[13] = 0;        /* Output: Number of perturbed pivots */
	iparm[14] = 0;        /* Not in use */
	iparm[15] = 0;        /* Not in use */
	iparm[16] = 0;        /* Not in use */
	iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
	iparm[18] = -1;       /* Output: Mflops for LU factorization */
	iparm[19] = 0;        /* Output: Numbers of CG Iterations */
	iparm[20] = 1;        /* Pivoting for symmetric indefinite matrices. */
	iparm[27] = 0;        /* Double Precision */
	iparm[34] = 0;        /* One-based indexing */

	maxfct = 1;           /* Maximum number of numerical factorizations. */
	mnum = 1;             /* Which factorization to use. */
	msglvl = 0;           /* Do not print statistical information in file */
	error = 0;            /* Initialize error flag */
	/* -------------------------------------------------------------------- */
	/* .. Initialize the internal solver memory pointer. This is only */
	/* necessary for the FIRST call of the PARDISO solver. */
	/* -------------------------------------------------------------------- */
	for (i = 0; i < 64; i++)
	{
		pt[i] = 0;
	}
	/* -------------------------------------------------------------------- */
	/* .. Reordering and Symbolic Factorization. This step also allocates */
	/* all memory that is necessary for the factorization. */
	/* -------------------------------------------------------------------- */

	sparse_check(eleNum, rowIndex, columns);

	phase = 11;
	int nrhs = 1;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during symbolic factorization: %d", error);
		return;
	}
	printf("\nReordering completed ... ");
	printf("\nNumber of nonzeros in factors = %d", iparm[17]);
	printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

	/* -------------------------------------------------------------------- */
	/* .. Numerical factorization. */
	/* -------------------------------------------------------------------- */
	phase = 22;
	printf("\nHMM equation no. is %d\n", iparm[29]);
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0)
	{
		printf("\nERROR during numerical factorization: %d", error);
		printf("\nERROR equation no. is %d\n", iparm[29]);
		//exit(2);
	}
	printf("\nFactorization completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Back substitution and iterative refinement. */
	/* -------------------------------------------------------------------- */
	phase = 33;
	iparm[7] = 2; /* Max numbers of iterative refinement steps. */

	if (Bab0)
	{
		PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, Bab0, Xab0, &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
		}
		printf("\nSolve completed ... ");
	}

	if (error != 0)
	{
		printf("\nERROR during solution: %d", error);
	}
	printf("\nSolve completed ... ");

	/* -------------------------------------------------------------------- */
	/* .. Termination and release of memory. */
	/* -------------------------------------------------------------------- */
	phase = -1; /* Release internal memory. */
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &eleNum, A0, rowIndex, columns, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);

	////////////////////////////////////////////////////////////////////
	copy_data(res, Xab0, channel, res.cols, res.rows);
}
