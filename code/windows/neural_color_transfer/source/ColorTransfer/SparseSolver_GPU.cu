#include "SparseSolver_GPU.cuh"

void solve_ls_cg_gpu(int size, int constraints, double* A, int* columns, int* rowindex, double* x, double* b, int nonzeros, double tolerance, int maxitrs)
{
	/* Create CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	/* Description of the A matrix*/
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);

	int* dACol;
	int* dARow;
	int* dATACol;
	int* dATARow;
	double* dA;
	double* dATA;
	double* db;
	double* dATb;
	int nnzA = nonzeros;
	int nA = size;
	int ncA = constraints;
	int nnzATA = 0;
	int nATA = size;

	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&dACol,  nnzA * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&dARow,  (ncA + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&dA,     nnzA * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&db,     ncA * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&dATb,   nA * sizeof(double)));

	cudaMemcpy(dACol, columns,   nnzA * sizeof(int),      cudaMemcpyHostToDevice);
	cudaMemcpy(dARow, rowindex,  (ncA + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dA,    A,         nnzA * sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(db,    b,         ncA * sizeof(double),    cudaMemcpyHostToDevice);

	/* Compute ATA */
	cusparseMatDescr_t aDescr = 0;
	cusparseStatus = cusparseCreateMatDescr(&aDescr);

	checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(aDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(aDescr, CUSPARSE_INDEX_BASE_ONE);

	int *nnzTotalDevHostPtr = &nnzATA;
	cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
	cudaMalloc((void**)&dATARow, sizeof(int) * (nATA + 1));

	cusparseXcsrgemmNnz(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nA, nA, ncA,
		aDescr, nnzA, dARow, dACol, aDescr, nnzA, dARow, dACol, descr, dATARow, nnzTotalDevHostPtr);

	if (NULL != nnzTotalDevHostPtr)
	{ 
		nnzATA = *nnzTotalDevHostPtr;
	}
	else
	{
		cudaMemcpy(&nnzATA, dATARow + nA, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&nATA,   dATARow + nA, sizeof(int), cudaMemcpyDeviceToHost);
		nnzATA -= nATA;
	}

	cudaMalloc((void**)&dATACol, sizeof(int)* nnzATA);
	cudaMalloc((void**)&dATA,    sizeof(double)* nnzATA);

	cusparseDcsrgemm(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nA, nA, ncA,
		aDescr, nnzA, dA, dARow, dACol, aDescr, nnzA, dA, dARow, dACol, descr, dATA, dATARow, dATACol);

	cusparseDestroyMatDescr(aDescr);
	
	/* Compute ATb */
	double alpha = 1.0;
	double beta = 0.0;

	cusparseStatus = cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, ncA, nA, nnzA, &alpha, descr, dA, dARow, dACol, db, &beta, dATb);

	/* Release useless memory */
	cudaFree(dA);
	cudaFree(db);
	cudaFree(dACol);
	cudaFree(dARow);

	double* dx;
	double* dP;
	double* dAx;

	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&dx, nATA * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&dP, nATA * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&dAx, nATA * sizeof(double)));

	cudaMemcpy(dx, x, nATA * sizeof(double), cudaMemcpyHostToDevice);

	/* Conjugate gradient without preconditioning.
	------------------------------------------
	Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */
	printf("Convergence of conjugate gradient without preconditioning: \n");

	int k = 0;
	double r0 = 0, r1 = 0, dot = 0;
	double alpham1 = -1.0;
	double vb = 0.0;
	double va = 0.0, na = 0.0;

	alpha = 1.0;
	beta = 0.0;

	cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, nATA, nATA, nnzATA, &alpha, descr, dATA, dATARow, dATACol, dx, &beta, dAx);
	cublasDaxpy(cublasHandle, nATA, &alpham1, dAx, 1, dATb, 1);
	cublasStatus = cublasDdot(cublasHandle, nATA, dATb, 1, dATb, 1, &r1);

	k = 1;
	while (r1 > tolerance * tolerance && k <= maxitrs)
	{
		if (k > 1)
		{
			vb = r1 / r0;
			cublasStatus = cublasDscal(cublasHandle, nATA, &vb, dP, 1);
			cublasStatus = cublasDaxpy(cublasHandle, nATA, &alpha, dATb, 1, dP, 1);
		}
		else
		{
			cublasStatus = cublasDcopy(cublasHandle, nATA, dATb, 1, dP, 1);
		}

		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, nATA, nATA, nnzATA, &alpha, descr, dATA, dATARow, dATACol, dP, &beta, dAx);
		cublasDdot(cublasHandle, nATA, dP, 1, dAx, 1, &dot);

		va = r1 / dot;
		cublasDaxpy(cublasHandle, nATA, &va, dP, 1, dx, 1);

		na = -va;
		cublasDaxpy(cublasHandle, nATA, &na, dAx, 1, dATb, 1);

		r0 = r1;
		cublasDdot(cublasHandle, nATA, dATb, 1, dATb, 1, &r1);
		cudaThreadSynchronize();
		k++;
	}

	//printf("  iteration = %3d, residual = %e \n", k, sqrt(r1));

	cudaMemcpy(x, dx, nATA * sizeof(double), cudaMemcpyDeviceToHost);

	/* check result */
	if (0)
	{
		double err = 0.0;
		for (int i = 0; i < nATA; i++)
		{
			double rsum = 0.0;
			for (int j = rowindex[i]; j < rowindex[i + 1]; j++)
			{
				rsum += A[j - 1] * x[columns[j - 1] - 1];
			}

			double diff = fabs(rsum - b[i]);

			if (diff > err)
			{
				err = diff;
			}
		}
		printf("  Convergence Test: %s \n", (k <= maxitrs) ? "OK" : "FAIL");
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	cusparseDestroyMatDescr(descr);

	cudaFree(dATACol);
	cudaFree(dATARow);
	cudaFree(dATA);
	cudaFree(dx);
	cudaFree(dATb);
	cudaFree(dP);
	cudaFree(dAx);
}