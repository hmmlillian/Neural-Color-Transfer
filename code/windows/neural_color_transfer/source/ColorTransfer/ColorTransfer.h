#ifndef COLOR_TRANSFER_H
#define COLOR_TRANSFER_H

#include "Config.h"
#include "SparseSolver_CPU.h"
#include "SparseSolver_GPU.cuh"

using namespace std;
using namespace cv;

struct NN
{
	int id;
	double w;
	NN()
	{
		id = -1;
		w = 0.f;
	}
	NN(int index, double weight)
	{
		id = index;
		w = weight;
	}
};

struct ClusterPixel
{
	int id;
	Vec3d col;
};

struct Long3
{
	long data[3];
	Long3()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
	}
	Long3(long a, long b, long c)
	{
		data[0] = a;
		data[1] = b;
		data[2] = c;
	}
};


class ColorTransfer
{
public:
	ColorTransfer(const Config& config, const Mat& cntRgb) : m_config(config)
	{ 
		m_cntLab = Mat::zeros(cntRgb.size(), CV_8UC3);
		m_cntLabD = Mat::zeros(cntRgb.size(), CV_64FC3);
		cvtColor(cntRgb, m_cntLab, CV_BGR2Lab);
		m_cntLab.convertTo(m_cntLabD, CV_64F, 1.0 / 255.0);


		m_width       = m_cntLab.cols; 
		m_height      = m_cntLab.rows;
		m_gradX       = Mat::zeros(m_cntLab.size(), CV_64FC1);
		m_gradY       = Mat::zeros(m_cntLab.size(), CV_64FC1);
		m_a           = Mat::zeros(m_cntLab.size(), CV_64FC3);
		m_b           = Mat::zeros(m_cntLab.size(), CV_64FC3);
		m_resD        = Mat::zeros(m_cntLab.size(), CV_64FC3);
		m_resInitD    = Mat::zeros(m_cntLab.size(), CV_64FC3);
		m_resNonlocalD = Mat::zeros(m_cntLab.size(), CV_64FC3);
		m_res         = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_resInit     = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_resNonlocal = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_weight      = Mat::zeros(m_cntLab.size(), CV_64FC1);


#ifdef ENABLE_VIS
		m_aVis = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_bVis = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_aVisInit  = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_bVisInit  = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_aNonlocal = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_bNonlocal = Mat::zeros(m_cntLab.size(), CV_8UC3);
		m_errMap    = Mat::zeros(m_cntLab.size(), CV_8UC3);
#endif
	}
	~ColorTransfer(){}

	const Mat& getRes()         { return m_res;         }
	const Mat& getResInit()     { return m_resInit;     }
	const Mat& getResNonlocal() { return m_resNonlocal; }
	
	vector<Long3>& getCntTable()  { return m_cntTable;  }
	vector<Long3>& getCntTable2() { return m_cntTable2; }
	vector<Long3>& getStlTable()  { return m_stlTable;  }
	vector<Long3>& getStlTable2() { return m_stlTable2; }

	const Mat& getAVis()      { return m_aVis;      }
	const Mat& getAVisInit()  { return m_aVisInit;  }
	const Mat& getANonlocal() { return m_aNonlocal; }
	const Mat& getBVis()      { return m_bVis;      }
	const Mat& getBVisInit()  { return m_bVisInit;  }
	const Mat& getBNonlocal() { return m_bNonlocal; }
	const Mat& getErrorMap()  { return m_errMap;    }
	const Mat& getPatchVis()  { return m_patchVis;  }

	void transfer_color_downsample(float* errData, const Mat& cntRgb, const Mat& cntLab, const Mat& stlLab,
		int patchSize, int layer, int downWidth, int downHeight, int samples, const Mat& cntImg, const Mat& stlImg, const Mat& respMask = Mat());

	void build_accumTable_downsample(const Mat& dimg, vector<Long3>& table, vector<Long3>& table2);

	void upsample_color_coefficients_bilinear(vector<double>& roughness, Mat& aMat, Mat& bMat, const Mat& daMat, const Mat& dbMat, const int dwidth, const int dheight, const int width, const int height);

	void clusterFeastures(Mat& dvisMat, float* features, int width, int height, int channel);

	void findKnns(Mat& visMat, const Mat& cntRgb, int samples);

private:
	vector<Long3> m_cntTable;
	vector<Long3> m_cntTable2;
	vector<Long3> m_stlTable;
	vector<Long3> m_stlTable2;


	vector<vector<int>>    m_knn;
	vector<vector<double>> m_knnd;
	vector<vector<NN>>     m_knnid;

	vector<int>   m_labels;
	int           m_labelNum;
	int           m_labelWidth, m_labelHeight;

	const Config& m_config;

	Mat  m_cntLabD;
	Mat  m_cntLab;
	Mat  m_gradX;
	Mat  m_gradY;
	Mat  m_a;
	Mat  m_b;
	Mat  m_resD;
	Mat  m_res;
	Mat  m_resInitD;
	Mat  m_resInit;
	Mat  m_resNonlocal;
	Mat  m_resNonlocalD;
	Mat  m_weight;


	int           m_width;
	int           m_height;
	int           m_downWidth;
	int           m_downHeight;

	Mat           m_aVis;
	Mat           m_bVis;
	Mat           m_aVisInit;
	Mat           m_bVisInit;
	Mat           m_aNonlocal;
	Mat           m_bNonlocal;
	Mat           m_errMap;
	Mat           m_patchVis;

	////////////////////////////////////////////////////////////
	// Visualization helper functions
	void visualizeClusterRandom(Mat& visMat, const vector<vector<int>>& indices, int width, int height);

#ifdef ENABLE_VIS
	void getHeat(float v, float vmin, float vmax, uchar& r, uchar& g, uchar& b);
#endif


	////////////////////////////////////////////////////////////
	// Clustering helper functions
	void sortMergeComputeWeight(const vector<vector<vector<int>>>& nns, const vector<vector<vector<double>>>& nnds);

	void convertDist2Weight();

	void findSubKNNs(vector<vector<int>>& nns, vector<vector<double>>& nnds, const vector<ClusterPixel>& subCluster, int width, int height);

	void insertClusterPixel(vector<ClusterPixel>& subClusterPixels, int& count, const Mat& cntRgb, int x, int y, int samples);

	void getClusters(Mat& visMat, vector<vector<ClusterPixel>>& subClusterPixels, const Mat& cntRgb, int samples);

	////////////////////////////////////////////////////////////
	// Sparse solver functions
	void compute_gradientMat(double lamda, double alpha);

	void compute_gradientMat(Mat& gradX, Mat& gradY, const Mat& cntLabD, double lamda, double alpha);

	void solve_nonlocal_downsample_gpu_gradient(Mat& aMat, Mat& bMat,
		const Mat& src, const Mat& ref,
		const Mat& cntRgb, int layer, float lambda, float alpha, float dWeight);
	
	void solve_WLS_roughness_cpu(const vector<double>& roughness, Mat& aMat, Mat& bMat, double alpha, double lamda);
};


#endif