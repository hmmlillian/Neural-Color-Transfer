#include "stdafx.h"
#include "ColorTransfer.h"

#include "Flann/nanoflann.hpp"
#include "Flann/flann_base.hpp"

#include "mkl.h"
#include "mkl_dss.h"

using namespace nanoflann;

struct PointColor
{
	vector<Vec3d> pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline double kdtree_distance(const double *p1, const int idx_p2, int /*size*/) const
	{
		const double d0 = p1[0] - pts[idx_p2][0];
		const double d1 = p1[1] - pts[idx_p2][1];
		const double d2 = p1[2] - pts[idx_p2][2];
		return max(sqrt(d0 * d0 + d1 * d1 + d2 * d2), 0.0);
		//return int(sqrt(float(d0 * d0 + d1 * d1 + d2 * d2)) + 0.5f);
	}

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline double kdtree_get_pt(const int idx, int dim) const
	{
		return pts[idx][dim];
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

bool cmpDist(const NN& n1, const NN& n2) { if (n1.w == n2.w) return n1.id < n2.id;  return n1.w < n2.w; }

Long3 getValue(const vector<Long3>& table, int sx, int ex, int sy, int ey, int width)
{
	Long3 val(0, 0, 0);
	for (int y = sy; y < ey; ++y)
	{
		int id0 = y * width + sx;
		int id1 = y * width + ex;
		val.data[0] += table[id1].data[0] - table[id0].data[0];
		val.data[1] += table[id1].data[1] - table[id0].data[1];
		val.data[2] += table[id1].data[2] - table[id0].data[2];
	}
	return val;
}

void ColorTransfer::sortMergeComputeWeight(const vector<vector<vector<int>>>& nns, const vector<vector<vector<double>>>& nnds)
{
	int num = nns.size();
	int k = m_config.m_kNum;
	int sz = nns[0].size();

	m_knnid.clear();
	m_knnid.resize(sz);


#pragma omp parallel for
	for (int i = 0; i < sz; ++i)
	{
		for (int j = 0; j < num; ++j)
		{
			const vector<int>& nnVec = nns[j][i];
			
			if (nnVec.size())
			{
				int kNum = nnVec.size();
				for (int ki = 0; ki < kNum; ++ki)
				{
					m_knnid[i].push_back(NN(nnVec[ki], nnds[j][i][ki]));
				}
			}
		}

		sort(m_knnid[i].begin(), m_knnid[i].end(), cmpDist);
		
		int lastId = -1;
		int lastPos = 0;
		int kNum = m_knnid[i].size();
		double wSum = 0.0;
		for (int ki = 0; ki < kNum; ++ki)
		{
			if (m_knnid[i][ki].id != lastId)
			{
				lastId = m_knnid[i][ki].id;
				m_knnid[i][lastPos].id = lastId;
				m_knnid[i][lastPos].w = exp(1.0 - m_knnid[i][ki].w / 3.0);
				wSum += m_knnid[i][lastPos].w;

				lastPos++;
				if (lastPos >= k) break;
			}
		}

		assert(lastPos == k);
		m_knnid[i].resize(k);
	}
}

void ColorTransfer::convertDist2Weight()
{
	vector<vector<double>>& nnds = m_knnd;
	int num = nnds.size();
	for (int i = 0; i < num; ++i)
	{
		int kNum = m_knnd[i].size();
		double wSum = 0.0;
		for (int j = 0; j < kNum; ++j)
		{
			m_knnd[i][j] = exp(1.0 - m_knnd[i][j] / 3.0);
			wSum += m_knnd[i][j];
		}

		if (wSum > 0.0)
		{
			for (int j = 0; j < kNum; ++j)
			{
				m_knnd[i][j] /= wSum;
			}
		}
	}
}

void ColorTransfer::findSubKNNs(vector<vector<int>>& nns, vector<vector<double>>& nnds, const vector<ClusterPixel>& subCluster, int width, int height)
{
	int num = width * height;
	nns.resize(num);
	nnds.resize(num);

	int k = m_config.m_kNum;
	int sz = subCluster.size();

	PointColor colorPnts;
	colorPnts.pts.resize(sz);
#pragma omp parallel for
	for (int i = 0; i < sz; ++i)
	{
		colorPnts.pts[i] = subCluster[i].col;
	}

	// construct a kd-tree index:
	typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointColor>, PointColor, 3 /* dim */, int> my_kd_tree_t;

	my_kd_tree_t index(3 /*dim*/, colorPnts, KDTreeSingleIndexAdaptorParams(k /* max leaf */));
	index.buildIndex();

	// ----------------------------------------------------------------
	// knnSearch():  Perform a search for the N closest points
	// ----------------------------------------------------------------

#pragma omp parallel for
	for (int s = 0; s < sz; ++s)
	{
		int id = subCluster[s].id;
		double x = int(id % width) / (double)width;
		double y = int(id / width) / (double)height;
		const double query_pt[6] = { colorPnts.pts[s][0], colorPnts.pts[s][1], colorPnts.pts[s][2], x, y, 0.f };

		size_t numResults = k + 1;
		nns[id].clear();
		nnds[id].clear();
		nns[id].resize(numResults, -1);
		nnds[id].resize(numResults, MAX_VAL);
		numResults = index.knnSearch(&query_pt[0], numResults, &(nns[id][0]), &nnds[id][0]);
		nns[id].resize(numResults);
		nnds[id].resize(numResults);

		// convert to real index
		int ni = 0;
		for (int t = 0; t < numResults; ++t)
		{
			int nid = subCluster[nns[id][t]].id;
			if (nid != id && ni < k)
			{
				nns[id][ni] = nid;
				nnds[id][ni] = nnds[id][t];
				ni++;
			}
		}

		nns[id].resize(ni);
		nnds[id].resize(ni);
	}

#ifdef ENABLE_VIS
	vector<int> usedTimes(num, 0);
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < nns[i].size(); ++j)
		{
			usedTimes[nns[i][j]] += 1;
		}
	}

	int usedMax = 0;
	int idMax = 0;
	for (int i = 0; i < num; ++i)
	{
		if (usedMax < usedTimes[i])
		{
			usedMax = usedTimes[i];
			idMax = i;
		}
	}

	printf("Max of used times = %d, id = [%d, %d]\n", usedMax, idMax / width, idMax % width);
#endif
}

#ifdef ENABLE_VIS
void ColorTransfer::visualizeClusterRandom(Mat& visMat, const vector<vector<int>>& indices, int width, int height)
{
	int colNum = indices.size();
	int maxNum = 0;

	visMat = Mat::zeros(height, width, CV_8UC3);

	for (int i = 0; i < colNum; ++i)
	{
		int r = rand() % 256;
		int g = rand() % 256;
		int b = rand() % 256;
		Vec3b col(r, g, b);

		for (int j = 0; j < indices[i].size(); ++j)
		{
			int px = indices[i][j] % width;
			int py = indices[i][j] / width;
			visMat.at<Vec3b>(py, px) = col;
		}

		if (indices[i].size() > maxNum)
		{
			maxNum = indices[i].size();
		}
	}

	printf("Max Num = %d\n", maxNum);
}
#endif

void ColorTransfer::insertClusterPixel(vector<ClusterPixel>& subClusterPixels, int& count, const Mat& cntLab, int x, int y, int samples)
{
	int sx = x * samples;
	int sy = y * samples;
	int ex = min(sx + samples, cntLab.cols);
	int ey = min(sy + samples, cntLab.rows);

	for (int yi = sy; yi < ey; ++yi)
	{
		for (int xi = sx; xi < ex; ++xi)
		{
			subClusterPixels[count].col = cntLab.at<Vec3d>(yi, xi);
			subClusterPixels[count].id = yi * cntLab.cols + xi;
			count++;
		}
	}
}

void ColorTransfer::getClusters(Mat& visMat, vector<vector<ClusterPixel>>& subClusterPixels,
	const Mat& cntLab, int samples)
{
	int height = cntLab.rows;
	int width  = cntLab.cols;

	vector<int> count(m_labelNum, 0);
	vector<Mat> clusterMat(m_labelNum);
	subClusterPixels.resize(m_labelNum);
	for (int l = 0; l < m_labelNum; ++l)
	{
		subClusterPixels[l].resize(height * width);
		clusterMat[l] = Mat::zeros(m_labelHeight, m_labelWidth, CV_8UC1);
	}

#pragma omp parallel for
	for (int y = 0; y < m_labelHeight; ++y)
	{
		for (int x = 0; x < m_labelWidth; ++x)
		{
			int id = y * m_labelWidth + x;
			int id0 = m_labels[id];

			clusterMat[id0].at<uchar>(y, x) = 255;

			if (x < m_labelWidth - 1 && id0 != m_labels[id + 1])
			{
				clusterMat[id0].at<uchar>(y, x + 1) = 255;
			}
			if (x > 0 && id0 != m_labels[id - 1])
			{
				clusterMat[id0].at<uchar>(y, x - 1) = 255;
			}
			if (y < m_labelHeight - 1 && id0 != m_labels[id + m_labelWidth])
			{
				clusterMat[id0].at<uchar>(y + 1, x) = 255;
			}
			if (y > 0 && id0 != m_labels[id - m_labelWidth])
			{
				clusterMat[id0].at<uchar>(y - 1, x) = 255;
			}
		}
	}

	for (int y = 0; y < m_labelHeight; ++y)
	{
		for (int x = 0; x < m_labelWidth; ++x)
		{
			for (int l = 0; l < m_labelNum; ++l)
			{
				if (clusterMat[l].at<uchar>(y, x))
				{
					insertClusterPixel(subClusterPixels[l], count[l], cntLab, x, y, samples);
				}
			}
		}
	}

	for (int l = 0; l < m_labelNum; ++l)
	{
		subClusterPixels[l].resize(count[l]);
	}

#ifdef ENABLE_VIS
	for (int l = 0; l < m_labelNum; ++l)
	{
		visMat = Mat::zeros(height, width, CV_8UC3);
		for (int i = 0; i < subClusterPixels[l].size(); ++i)
		{
			int x = subClusterPixels[l][i].id % width;
			int y = subClusterPixels[l][i].id / width;
			visMat.at<Vec3b>(y, x) = Vec3b(subClusterPixels[l][i].col[0], subClusterPixels[l][i].col[1], subClusterPixels[l][i].col[2]);
		}
	}
#endif
}

void ColorTransfer::clusterFeastures(Mat& dvisMat, float* features, int width, int height, int channel)
{
	srand(1);

	int sz = width * height;
	int cNum = m_config.m_clusterNum;
	float* centers = new float[cNum * channel];
	memset(centers, 0, sizeof(float)* cNum * channel);

	cvflann::Matrix<float> featureMat(features, sz, channel);
	cvflann::Matrix<float> centerMat(centers, cNum, channel);

#ifdef ENABLE_VIS
	printf("Before Cluster Num = %d\n", cNum);
#endif

	vector<vector<int>> indices;
	m_labelNum = cvflann::hierarchicalClustering<cvflann::L2<float>>(featureMat, centerMat,
		cvflann::KMeansIndexParams(cNum, 11, cvflann::FLANN_CENTERS_RANDOM), indices);

	m_labelWidth  = width;
	m_labelHeight = height;
	m_labels.resize(sz);
	
	for (int i = 0; i < indices.size(); ++i)
	{
		for (int j = 0; j < indices[i].size(); ++j)
		{
			int px = indices[i][j] % width;
			int py = indices[i][j] / width;
			m_labels[py * width + px] = i;
		}
	}

#ifdef ENABLE_VIS
	visualizeClusterRandom(dvisMat, indices, width, height);
	printf("After Cluster Num = %d\n", m_labelNum);
#endif
	
	delete[] centers;
}

void ColorTransfer::findKnns(Mat& visMat, const Mat& cntLab, int samples)
{
	srand(1);
	vector<vector<ClusterPixel>> subClusterPixels;
	getClusters(visMat, subClusterPixels, cntLab, samples);

	m_knn.clear();
	m_knnd.clear();
	m_knn.resize(cntLab.cols * cntLab.rows);
	m_knnd.resize(cntLab.cols * cntLab.rows);
	for (int i = 0; i < m_labelNum; ++i)
	{
		std::random_shuffle(subClusterPixels[i].begin(), subClusterPixels[i].end());
	}

	vector<vector<vector<int>>> knns(m_labelNum);
	vector<vector<vector<double>>> knnds(m_labelNum);

#pragma omp parallel for
	for (int i = 0; i < m_labelNum; ++i)
	{
		findSubKNNs(knns[i], knnds[i], subClusterPixels[i], cntLab.cols, cntLab.rows);
	}

	// merge all the knn results and compute weights
	sortMergeComputeWeight(knns, knnds);
}

void ColorTransfer::build_accumTable_downsample(const Mat& dimg, vector<Long3>& table, vector<Long3>& table2)
{
	assert(dimg.cols);

	int sz = dimg.cols * dimg.rows + 1;
	table.resize(sz);
	table2.resize(sz);

	table[0].data[0] = 0;
	table[0].data[1] = 0;
	table[0].data[2] = 0;
	table2[0].data[0] = 0;
	table2[0].data[1] = 0;
	table2[0].data[2] = 0;

	for (int id = 1; id < sz; ++id)
	{
		int x = (id - 1) % dimg.cols;
		int y = (id - 1) / dimg.cols;

		const Vec3b& col = dimg.at<Vec3b>(y, x);

		table[id].data[0] = table[id - 1].data[0] + col[0];
		table[id].data[1] = table[id - 1].data[1] + col[1];
		table[id].data[2] = table[id - 1].data[2] + col[2];

		table2[id].data[0] = table2[id - 1].data[0] + col[0] * col[0];
		table2[id].data[1] = table2[id - 1].data[1] + col[1] * col[1];
		table2[id].data[2] = table2[id - 1].data[2] + col[2] * col[2];
	}
}

void ColorTransfer::upsample_color_coefficients_bilinear(vector<double>& roughness, Mat& aMat, Mat& bMat, const Mat& daMat, const Mat& dbMat,
	const int dwidth, const int dheight, const int width, const int height)
{
	if (width > dwidth || height > dheight)
	{
		resize(daMat, aMat, Size(width, height), 0, 0, CV_INTER_LINEAR);
		resize(dbMat, bMat, Size(width, height), 0, 0, CV_INTER_LINEAR);
	}

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			const Vec3d& a = aMat.at<Vec3d>(y, x);
			const Vec3d& b = bMat.at<Vec3d>(y, x);
			const Vec3d& col = m_cntLabD.at<Vec3d>(y, x);

			int id = y * width + x;
			for (int c = 0; c < 3; ++c)
			{
				double nc = col[c] * a[c] + b[c];
				if (nc < 0 || nc > 1)
				{
					roughness[id] = 1e-6;
				}
				else
				{
					roughness[id] = 1.0;
				}
			}
		}
	}
}

void ColorTransfer::compute_gradientMat(double lamda, double alpha)
{
	double epsilon = 0.0001;

#pragma omp parallel for
	for (int y = 0; y < m_height; ++y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			m_gradX.at<double>(y, x) = 0;
			m_gradY.at<double>(y, x) = 0;

			double val = m_cntLabD.at<Vec3d>(y, x)[0];
			if (x + 1 < m_width)
			{
				double gx = m_cntLabD.at<Vec3d>(y, x + 1)[0] - val;
				m_gradX.at<double>(y, x) = sqrt(lamda / (pow(abs(gx), alpha) + epsilon));
			}
			if (y + 1 < m_height)
			{
				double gy = m_cntLabD.at<Vec3d>(y + 1, x)[0] - val;
				m_gradY.at<double>(y, x) = sqrt(lamda / (pow(abs(gy), alpha) + epsilon));
			}
		}
	}
}

void ColorTransfer::compute_gradientMat(Mat& gradX, Mat& gradY, const Mat& cntLabD, double lamda, double alpha)
{
	double epsilon = 0.0001;

	int width = gradX.cols;
	int height = gradX.rows;

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			double val = cntLabD.at<Vec3d>(y, x)[0];
			gradX.at<double>(y, x) = 0;
			gradY.at<double>(y, x) = 0;
			if (x + 1 < width)
			{
				double gx = cntLabD.at<Vec3d>(y, x + 1)[0] - val;
				gradX.at<double>(y, x) = sqrt(lamda / (pow(abs(gx), alpha) + epsilon));
			}
			if (y + 1 < height)
			{
				double gy = cntLabD.at<Vec3d>(y + 1, x)[0] - val;
				gradY.at<double>(y, x) = sqrt(lamda / (pow(abs(gy), alpha) + epsilon));
			}
		}
	}
}

void ColorTransfer::solve_nonlocal_downsample_gpu_gradient(Mat& aMat, Mat& bMat,
	const Mat& src, const Mat& ref,
	const Mat& cntRgb, int layer, float lambda, float alpha, float dWeight)
{
	Mat gradX(src.size(), CV_64FC1);
	Mat gradY(src.size(), CV_64FC1);
	compute_gradientMat(gradX, gradY, src, lambda, alpha);

	int height = cntRgb.rows;
	int width = cntRgb.cols;

	int patchSize = 3;
	int leftSize = -patchSize / 2;
	int rightSize = patchSize + leftSize;

	double nonlocalWeight = sqrt(m_config.m_nonlocalWeight / double(m_config.m_kNum));

	int nonlocalConstraints = 0;
	for (int i = 0; i < m_knnid.size(); ++i)
	{
		nonlocalConstraints += m_knnid[i].size();
	}

	//prepare
	int size = width * height * 2;
	int localConstraints = size * 4 * 2;
	int constraints = size + localConstraints + nonlocalConstraints * 2;
	int nzs = size + localConstraints * 2 + nonlocalConstraints * 4;

	//matrix	
	double *CA0 = new double[nzs];
	double *CA1 = new double[nzs];
	double *CA2 = new double[nzs];
	int    *Ccolumns = new int[nzs];
	int    *Crowindex = new int[constraints + 1];

	double *CBab0 = new double[constraints];
	double *CBab1 = new double[constraints];
	double *CBab2 = new double[constraints];

	memset(CA0, 0, nzs * sizeof(double));
	memset(CA1, 0, nzs * sizeof(double));
	memset(CA2, 0, nzs * sizeof(double));
	memset(Ccolumns, 0, nzs * sizeof(int));
	memset(Crowindex, 0, (constraints + 1) * sizeof(int));
	memset(CBab0, 0, constraints * sizeof(double));
	memset(CBab1, 0, constraints * sizeof(double));
	memset(CBab2, 0, constraints * sizeof(double));

	double *Rab0 = new double[size];
	double *Rab1 = new double[size];
	double *Rab2 = new double[size];

	memset(Rab0, 0, size * sizeof(double));
	memset(Rab1, 0, size * sizeof(double));
	memset(Rab2, 0, size * sizeof(double));

	// one-based indexing
	int nNonZeros = 0;
	int oneBased = 1;
	Crowindex[0] = nNonZeros + oneBased;
	int cid = 0;

	// data term
	if (1)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int aii = y * width + x;
				int bii = height * width + aii;
				
				double dataWeight = sqrt(m_weight.at<double>(y, x)) * sqrt(dWeight);

				const Vec3d& sVec = src.at<Vec3d>(y, x);
				const Vec3d& rVec = ref.at<Vec3d>(y, x);

				const Vec3d& aVec = aMat.at<Vec3d>(y, x);
				const Vec3d& bVec = bMat.at<Vec3d>(y, x);

				// initial guess
				Rab0[aii] = aVec[0];
				Rab1[aii] = aVec[1];
				Rab2[aii] = aVec[2];

				Rab0[bii] = bVec[0];
				Rab1[bii] = bVec[1];
				Rab2[bii] = bVec[2];

				CBab0[cid] = dataWeight * rVec[0];
				CBab1[cid] = dataWeight * rVec[1];
				CBab2[cid] = dataWeight * rVec[2];

				CA0[nNonZeros] = dataWeight * sVec[0];
				CA1[nNonZeros] = dataWeight * sVec[1];
				CA2[nNonZeros] = dataWeight * sVec[2];
				Ccolumns[nNonZeros] = aii + oneBased;
				nNonZeros++;

				CA0[nNonZeros] = dataWeight;
				CA1[nNonZeros] = dataWeight;
				CA2[nNonZeros] = dataWeight;
				Ccolumns[nNonZeros] = bii + oneBased;
				nNonZeros++;

				Crowindex[cid + 1] = nNonZeros + oneBased;
				cid++;
			}
		}
	}

	// local smooth term
	if (1)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int aii = y * width + x;
				int bii = height * width + aii;

				//locally smooth term
				if (x + 1 < width) // x gradient
				{
					const double gxw = gradX.at<double>(y, x);

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gxw;
					CA1[nNonZeros] = -gxw;
					CA2[nNonZeros] = -gxw;
					Ccolumns[nNonZeros] = aii + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gxw;
					CA1[nNonZeros] = gxw;
					CA2[nNonZeros] = gxw;
					Ccolumns[nNonZeros] = aii + 1 + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gxw;
					CA1[nNonZeros] = -gxw;
					CA2[nNonZeros] = -gxw;
					Ccolumns[nNonZeros] = bii + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gxw;
					CA1[nNonZeros] = gxw;
					CA2[nNonZeros] = gxw;
					Ccolumns[nNonZeros] = bii + 1 + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;
				}
				if (x - 1 >= 0)
				{
					const double gxw = gradX.at<double>(y, x - 1);
					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gxw;
					CA1[nNonZeros] = -gxw;
					CA2[nNonZeros] = -gxw;
					Ccolumns[nNonZeros] = aii - 1 + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gxw;
					CA1[nNonZeros] = gxw;
					CA2[nNonZeros] = gxw;
					Ccolumns[nNonZeros] = aii + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gxw;
					CA1[nNonZeros] = -gxw;
					CA2[nNonZeros] = -gxw;
					Ccolumns[nNonZeros] = bii - 1 + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gxw;
					CA1[nNonZeros] = gxw;
					CA2[nNonZeros] = gxw;
					Ccolumns[nNonZeros] = bii + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;
				}

				if (y + 1 < height) // y gradient
				{
					const double gyw = gradY.at<double>(y, x);

					if (gyw > 999999.f || gyw < -999999.f)
					{
						int hmm_test = 0;
						hmm_test++;
					}

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gyw;
					CA1[nNonZeros] = -gyw;
					CA2[nNonZeros] = -gyw;
					Ccolumns[nNonZeros] = aii + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gyw;
					CA1[nNonZeros] = gyw;
					CA2[nNonZeros] = gyw;
					Ccolumns[nNonZeros] = aii + width + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gyw;
					CA1[nNonZeros] = -gyw;
					CA2[nNonZeros] = -gyw;
					Ccolumns[nNonZeros] = bii + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gyw;
					CA1[nNonZeros] = gyw;
					CA2[nNonZeros] = gyw;
					Ccolumns[nNonZeros] = bii + width + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;
				}
				if (y - 1 >= 0)
				{
					const double gyw = gradY.at<double>(y - 1, x);
					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gyw;
					CA1[nNonZeros] = -gyw;
					CA2[nNonZeros] = -gyw;
					Ccolumns[nNonZeros] = aii - width + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gyw;
					CA1[nNonZeros] = gyw;
					CA2[nNonZeros] = gyw;
					Ccolumns[nNonZeros] = aii + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;

					CBab0[cid] = 0.0;
					CBab1[cid] = 0.0;
					CBab2[cid] = 0.0;

					CA0[nNonZeros] = -gyw;
					CA1[nNonZeros] = -gyw;
					CA2[nNonZeros] = -gyw;
					Ccolumns[nNonZeros] = bii - width + oneBased;
					nNonZeros++;

					CA0[nNonZeros] = gyw;
					CA1[nNonZeros] = gyw;
					CA2[nNonZeros] = gyw;
					Ccolumns[nNonZeros] = bii + oneBased;
					nNonZeros++;

					Crowindex[cid + 1] = nNonZeros + oneBased;
					cid++;
				}

			}
		}
	}

	// nonlocal smooth data
	if (1)
	{
		for (int c = 0; c < m_knnid.size(); ++c)
		{
			int num = m_knnid[c].size();
			int x = c % width;
			int y = c / width;
			const Vec3d& sVec0 = src.at<Vec3d>(y, x);
			int aii0 = c;
			int bii0 = height * width + aii0;

			for (int ki = 0; ki < num; ++ki)
			{
				int id1 = m_knnid[c][ki].id;
				int x1 = id1 % width;
				int y1 = id1 / width;
				const Vec3d& sVec1 = src.at<Vec3d>(y1, x1);
				int aii1 = y1 * width + x1;
				int bii1 = height * width + aii1;

				double iw = sqrt(m_knnid[c][ki].w) * nonlocalWeight;

				CBab0[cid] = 0;
				CBab1[cid] = 0;
				CBab2[cid] = 0;

				CA0[nNonZeros] = iw;
				CA1[nNonZeros] = iw;
				CA2[nNonZeros] = iw;
				Ccolumns[nNonZeros] = min(aii0, aii1) + oneBased;
				nNonZeros++;

				CA0[nNonZeros] = -iw;
				CA1[nNonZeros] = -iw;
				CA2[nNonZeros] = -iw;
				Ccolumns[nNonZeros] = max(aii1, aii0) + oneBased;
				nNonZeros++;

				Crowindex[cid + 1] = nNonZeros + oneBased;
				cid++;

				CBab0[cid] = 0.0;
				CBab1[cid] = 0.0;
				CBab2[cid] = 0.0;

				CA0[nNonZeros] = iw;
				CA1[nNonZeros] = iw;
				CA2[nNonZeros] = iw;
				Ccolumns[nNonZeros] = min(bii0, bii1) + oneBased;
				nNonZeros++;

				CA0[nNonZeros] = -iw;
				CA1[nNonZeros] = -iw;
				CA2[nNonZeros] = -iw;
				Ccolumns[nNonZeros] = max(bii1, bii0) + oneBased;
				nNonZeros++;

				Crowindex[cid + 1] = nNonZeros + oneBased;
				cid++;
			}
		}
	}

	// check matrix format
	sparse_check(cid, Crowindex, Ccolumns);

	double tolerance = 1e-6;
	int itrs = layer == 4 ? 50 : 100;

	solve_ls_cg_gpu(size, cid, CA0, Ccolumns, Crowindex, Rab0, CBab0, nNonZeros, tolerance, itrs);
	solve_ls_cg_gpu(size, cid, CA1, Ccolumns, Crowindex, Rab1, CBab1, nNonZeros, tolerance, itrs);
	solve_ls_cg_gpu(size, cid, CA2, Ccolumns, Crowindex, Rab2, CBab2, nNonZeros, tolerance, itrs);

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int aid = y * width + x;
			int bid = aid + width * height;
			aMat.at<Vec3d>(y, x) = Vec3d(Rab0[aid], Rab1[aid], Rab2[aid]);
			bMat.at<Vec3d>(y, x) = Vec3d(Rab0[bid], Rab1[bid], Rab2[bid]);
		}
	}

	delete[] CA0;
	delete[] CA1;
	delete[] CA2;

	delete[] Ccolumns;
	delete[] Crowindex;

	delete[] CBab0;
	delete[] CBab1;
	delete[] CBab2;

	delete[] Rab0;
	delete[] Rab1;
	delete[] Rab2;
}

void ColorTransfer::solve_WLS_roughness_cpu(const vector<double>& roughness, Mat& aMat, Mat& bMat, double alpha, double lamda)
{
	compute_gradientMat(lamda, alpha);

	//prepare
	int size = m_width * m_height;
	int nzs = size * 3;

	//matrix	
	double *A = new double[nzs];
	int  *columns = new int[nzs];
	int  *rowindex = new int[size + 1];

	double *Ba0 = new double[size];
	double *Ba1 = new double[size];
	double *Ba2 = new double[size];
	double *Bb0 = new double[size];
	double *Bb1 = new double[size];
	double *Bb2 = new double[size];

	memset(A, 0, nzs * sizeof(double));
	memset(columns, 0, nzs * sizeof(int));
	memset(rowindex, 0, (size + 1) * sizeof(int));
	memset(Ba0, 0, size * sizeof(double));
	memset(Ba1, 0, size * sizeof(double));
	memset(Ba2, 0, size * sizeof(double));
	memset(Bb0, 0, size * sizeof(double));
	memset(Bb1, 0, size * sizeof(double));
	memset(Bb2, 0, size * sizeof(double));

	double *Ra0 = new double[size];
	double *Ra1 = new double[size];
	double *Ra2 = new double[size];
	double *Rb0 = new double[size];
	double *Rb1 = new double[size];
	double *Rb2 = new double[size];

	memset(Ra0, 0, size * sizeof(double));
	memset(Ra1, 0, size * sizeof(double));
	memset(Ra2, 0, size * sizeof(double));
	memset(Rb0, 0, size * sizeof(double));
	memset(Rb1, 0, size * sizeof(double));
	memset(Rb2, 0, size * sizeof(double));

	// one-based indexing
	int nNonZeros = 0;
	int oneBased = 1;
	bool aNonzero[3] = { false, false, false };
	bool bNonzero[3] = { false, false, false };
	rowindex[0] = nNonZeros + oneBased;
		for (int y = 0; y < m_height; y++)
	{
		for (int x = 0; x < m_width; x++)
		{
			double a00 = 0.0, a01 = 0.0, a10 = 0.0;
			int ii = y * m_width + x;

			// data term
			double w = roughness[y * m_width + x];
			const Vec3d& aVec = aMat.at<Vec3d>(y, x);
			const Vec3d& bVec = bMat.at<Vec3d>(y, x);

			if (aVec[0])
			{
				aNonzero[0] = true;
				Ba0[ii] = w * aVec[0];
			}
			if (aVec[1])
			{
				aNonzero[1] = true;
				Ba1[ii] = w * aVec[1];
			}
			if (aVec[2])
			{
				aNonzero[2] = true;
				Ba2[ii] = w * aVec[2];
			}

			if (bVec[0])
			{
				bNonzero[0] = true;
				Bb0[ii] = w * bVec[0];
			}
			if (bVec[1])
			{
				bNonzero[1] = true;
				Bb1[ii] = w * bVec[1];
			}
			if (bVec[2])
			{
				bNonzero[2] = true;
				Bb2[ii] = w * bVec[2];
			}

			a00 += w;

			//locally smooth term
			if (x + 1 < m_width) // x gradient
			{
				const double gxw = pow(m_gradX.at<double>(y, x), 2);
				a00 += gxw;
				a01 -= gxw;
			}
			if (x - 1 >= 0)
			{
				const double gxw = pow(m_gradX.at<double>(y, x - 1), 2);
				a00 += gxw;
			}

			if (y + 1 < m_height) // y gradient
			{
				const double gyw = pow(m_gradY.at<double>(y, x), 2);
				a00 += gyw;
				a10 -= gyw;
			}
			if (y - 1 >= 0)
			{
				const double gyw = pow(m_gradY.at<double>(y - 1, x), 2);
				a00 += gyw;
			}

			A[nNonZeros] = a00;
			if (a00 <= 1)
			{
				printf("A matrix nonzeros = %d, a00 = %f, x = %d, y = %d\n", nNonZeros, a00, x, y);
			}
			columns[nNonZeros] = ii + oneBased;
			nNonZeros++;
			if (x + 1 < m_width)
			{
				A[nNonZeros] = a01;
				columns[nNonZeros] = ii + 1 + oneBased;
				nNonZeros++;
			}
			if (y + 1 < m_height)
			{
				A[nNonZeros] = a10;
				columns[nNonZeros] = ii + m_width + oneBased;
				nNonZeros++;
			}
			rowindex[ii + 1] = nNonZeros + oneBased;
		}
	}

	// check matrix format
	sparse_check(size, rowindex, columns);

	// check if there is nonzero element in Bas and Bbs, otherwise no need to solve MKL
	solve_direct_cpu(aMat, bMat,
		nNonZeros, size, oneBased,
		A, rowindex, columns,
		aNonzero[0] ? Ba0 : NULL, Ra0,
		aNonzero[1] ? Ba1 : NULL, Ra1,
		aNonzero[2] ? Ba2 : NULL, Ra2,
		bNonzero[0] ? Bb0 : NULL, Rb0,
		bNonzero[1] ? Bb1 : NULL, Rb1,
		bNonzero[2] ? Bb2 : NULL, Rb2);

	delete[] A;
	delete[] columns;
	delete[] rowindex;
	delete[] Ba0;
	delete[] Ba1;
	delete[] Ba2;
	delete[] Bb0;
	delete[] Bb1;
	delete[] Bb2;

	delete[] Ra0;
	delete[] Ra1;
	delete[] Ra2;
	delete[] Rb0;
	delete[] Rb1;
	delete[] Rb2;
}

#ifdef ENABLE_VIS
void ColorTransfer::getHeat(float v, float vmin, float vmax, uchar& r, uchar& g, uchar& b)
{
	r = 255;
	g = 255;
	b = 255;

	if (v < vmin)
		v = vmin;

	if (v > vmax)
		v = vmax;

	v = (v - vmin) / (vmax - vmin);

	double dr, dg, db;

	if (v < 0.1242)
	{
		db = 0.504 + ((1. - 0.504) / 0.1242) * v;
		dg = dr = 0.;
	}
	else if (v < 0.3747)
	{
		db = 1.;
		dr = 0.;
		dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
	}
	else if (v < 0.6253)
	{
		db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
		dg = 1.;
		dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
	}
	else if (v < 0.8758)
	{
		db = 0.;
		dr = 1.;
		dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
	}
	else
	{
		db = 0.;
		dg = 0.;
		dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
	}

	r = min(255, int(255 * dr));
	g = min(255, int(255 * dg));
	b = min(255, int(255 * db));
}
#endif

void ColorTransfer::transfer_color_downsample(float* errData, const Mat& cntRgb, const Mat& cntLab, const Mat& stlLab,
	int patchSize, int layer, int downWidth, int downHeight, int samples, const Mat& cntImg, const Mat& stlImg, const Mat& respMask)
{
	int width = downWidth;
	int height = downHeight;
	int leftSize = -patchSize / 2;
	int rightSize = patchSize + leftSize;

	double scaleFactor = 1.0 / 255.0;

#ifdef ENABLE_VIS
	m_patchVis = Mat::zeros(height * patchSize * 2, width * patchSize, CV_8UC3);
#endif

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			double cntMeanVal[3], cntVarVal[3], stlMeanVal[3], stlVarVal[3];

			int sxy[2] = { max(x + leftSize, 0), max(y + leftSize, 0) };
			int exy[2] = { min(x + rightSize, width), min(y + rightSize, height) };

#ifdef ENABLE_VIS
			for (int sy = sxy[1]; sy < exy[1]; ++sy)
			{
				for (int sx = sxy[0]; sx < exy[0]; ++sx)
				{
					const Vec3b& col = stlImg.at<Vec3b>(sy, sx);
					m_patchVis.at<Vec3b>(y * patchSize * 2 + sy - sxy[1], x * patchSize + sx - sxy[0]) = col;
				}
			}

			for (int cy = sxy[1]; cy < exy[1]; ++cy)
			{
				for (int cx = sxy[0]; cx < exy[0]; ++cx)
				{
					const Vec3b& col = cntImg.at<Vec3b>(cy, cx);
					m_patchVis.at<Vec3b>(y * patchSize * 2 + patchSize + cy - sxy[1], x * patchSize + cx - sxy[0]) = col;
				}
			}
#endif

			int cSum = (exy[0] - sxy[0]) * (exy[1] - sxy[1]);

			Long3 cntMean = getValue(m_cntTable, sxy[0], exy[0], sxy[1], exy[1], width);
			Long3 stlMean = getValue(m_stlTable, sxy[0], exy[0], sxy[1], exy[1], width);

			Long3 cntVar = getValue(m_cntTable2, sxy[0], exy[0], sxy[1], exy[1], width);
			Long3 stlVar = getValue(m_stlTable2, sxy[0], exy[0], sxy[1], exy[1], width);

			cntMeanVal[0] = cntMean.data[0] / (double)cSum;
			cntMeanVal[1] = cntMean.data[1] / (double)cSum;
			cntMeanVal[2] = cntMean.data[2] / (double)cSum;

			double cvs[3] = {
				max(cntVar.data[0] / (double)cSum - cntMeanVal[0] * cntMeanVal[0], 0.0),
				max(cntVar.data[1] / (double)cSum - cntMeanVal[1] * cntMeanVal[1], 0.0),
				max(cntVar.data[2] / (double)cSum - cntMeanVal[2] * cntMeanVal[2], 0.0)
			};


			cntVarVal[0] = max(sqrt(cvs[0]), 0.0);
			cntVarVal[1] = max(sqrt(cvs[1]), 0.0);
			cntVarVal[2] = max(sqrt(cvs[2]), 0.0);

			stlMeanVal[0] = stlMean.data[0] / (double)cSum;
			stlMeanVal[1] = stlMean.data[1] / (double)cSum;
			stlMeanVal[2] = stlMean.data[2] / (double)cSum;

			stlVarVal[0] = max(sqrt(max(stlVar.data[0] / (double)cSum - stlMeanVal[0] * stlMeanVal[0], 0.0)), 0.0);
			stlVarVal[1] = max(sqrt(max(stlVar.data[1] / (double)cSum - stlMeanVal[1] * stlMeanVal[1], 0.0)), 0.0);
			stlVarVal[2] = max(sqrt(max(stlVar.data[2] / (double)cSum - stlMeanVal[2] * stlMeanVal[2], 0.0)), 0.0);

			Vec3d& aVec = m_a.at<Vec3d>(y, x);
			aVec[0] = stlVarVal[0] / (cntVarVal[0] + m_config.m_varEpslon);
			aVec[1] = stlVarVal[1] / (cntVarVal[1] + m_config.m_varEpslon);
			aVec[2] = stlVarVal[2] / (cntVarVal[2] + m_config.m_varEpslon);

			Vec3d& bVec = m_b.at<Vec3d>(y, x);
			bVec[0] = (stlMeanVal[0] - cntMeanVal[0] * aVec[0]) * scaleFactor;
			bVec[1] = (stlMeanVal[1] - cntMeanVal[1] * aVec[1]) * scaleFactor;
			bVec[2] = (stlMeanVal[2] - cntMeanVal[2] * aVec[2]) * scaleFactor;
		}
	}

#ifdef ENABLE_VIS
#pragma omp parallel for
	for (int y = 0; y < m_height; ++y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			int dx = x / samples;
			int dy = y / samples;
			Vec3d& resCol = m_resInitD.at<Vec3d>(y, x);
			const Vec3d& col = m_cntLabD.at<Vec3d>(y, x);

			const Vec3d& a = m_a.at<Vec3d>(dy, dx);
			const Vec3d& b = m_b.at<Vec3d>(dy, dx);

			resCol = Vec3d(
				min(max(col[0] * a[0] + b[0], 0.0), 1.0),
				min(max(col[1] * a[1] + b[1], 0.0), 1.0),
				min(max(col[2] * a[2] + b[2], 0.0), 1.0));

			m_aVisInit.at<Vec3b>(y, x) = Vec3b(
				min(max(int(a[0] * 50), 0), 255),
				min(max(int(a[1] * 50), 0), 255),
				min(max(int(a[2] * 50), 0), 255)
				);

			m_bVisInit.at<Vec3b>(y, x) = Vec3b(
				min(max(int(b[0] * 255 + 127), 0), 255),
				min(max(int(b[1] * 255 + 127), 0), 255),
				min(max(int(b[2] * 255 + 127), 0), 255)
				);

		}
	}
#endif

	if (errData)
	{
		double minDist = MAX_VAL;
		double maxDist = MIN_VAL;
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int id = y * width + x;
				double err2 = errData[id];
				if (err2 < minDist)
				{
					minDist = err2;
				}
				if (err2 > maxDist)
				{
					maxDist = err2;
				}
			}
		}

#pragma omp parallel for
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				double err2 = errData[y * width + x];
				double err = (err2 - minDist) / (maxDist - minDist);

				m_weight.at<double>(y, x) = max(1.0 - err, 1e-6);

#ifdef ENABLE_VIS
				uchar rgb[3];
				getHeat(err, 0, 1.0, rgb[0], rgb[1], rgb[2]);
				m_errMap.at<Vec3b>(y, x) = Vec3b(rgb[2], rgb[1], rgb[0]);
#endif
			}
		}
	}
	else
	{
#pragma omp parallel for
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				m_weight.at<double>(y, x) = 1.0;

#ifdef ENABLE_VIS
				uchar rgb[3];
				getHeat(1.0, 0, 1.0, rgb[0], rgb[1], rgb[2]);
				m_errMap.at<Vec3b>(y, x) = Vec3b(rgb[2], rgb[1], rgb[0]);
#endif
			}
		}
	}

	double initVal = m_config.m_wlsLamdaInit;
	double normFactor = double(m_width * m_height) / double(width * height);
	double lamda = initVal * normFactor; 

	clock_t start, finish;
	double duration;

	// correct colors by nonlocal constraints
	start = clock();

	solve_nonlocal_downsample_gpu_gradient(m_a, m_b, cntLab, stlLab, cntRgb, layer, m_config.m_localWeight, m_config.m_wlsAlpha, normFactor);

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("Nonlocal Solve Time: %lf\n", duration);

	// upsample by NN
	Mat aCpy = m_a(Rect(0, 0, downWidth, downHeight)).clone();
	Mat bCpy = m_b(Rect(0, 0, downWidth, downHeight)).clone();
	
	vector<double> roughness(m_width * m_height, 1.0);
	upsample_color_coefficients_bilinear(roughness, m_a, m_b, aCpy, bCpy, downWidth, downHeight, m_width, m_height);
	
#ifdef ENABLE_VIS
#pragma omp parallel for
		for (int y = 0; y < m_height; ++y)
		{
			for (int x = 0; x < m_width; ++x)
			{
				int dx = x / samples;
				int dy = y / samples;
				Vec3d& resCol = m_resNonlocalD.at<Vec3d>(y, x);
				const Vec3d& col = m_cntLabD.at<Vec3d>(y, x);

				const Vec3d& a = m_a.at<Vec3d>(y, x);
				const Vec3d& b = m_b.at<Vec3d>(y, x);

				resCol = Vec3d(
					min(max(col[0] * a[0] + b[0], 0.0), 1.0),
					min(max(col[1] * a[1] + b[1], 0.0), 1.0),
					min(max(col[2] * a[2] + b[2], 0.0), 1.0));

				m_aNonlocal.at<Vec3b>(y, x) = Vec3b(
					min(max(int(a[0] * 50), 0), 255),
					min(max(int(a[1] * 50), 0), 255),
					min(max(int(a[2] * 50), 0), 255)
					);

				m_bNonlocal.at<Vec3b>(y, x) = Vec3b(
					min(max(int(b[0] * 255 + 127), 0), 255),
					min(max(int(b[1] * 255 + 127), 0), 255),
					min(max(int(b[2] * 255 + 127), 0), 255)
					);

			}
		}
#endif
		
		// (Optional) further smoother for the final level, can be skipped by setting if (0) 
		if (1)
		{
			if (downHeight == m_height && downWidth == m_width)
			{
				lamda = lamda * 4;
			}
		}

		// smooth colors by local constraints
		start = clock();

		solve_WLS_roughness_cpu(roughness, m_a, m_b, m_config.m_wlsAlpha, lamda);
		
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;

		printf("WLS Solve Time: %lf\n", duration);

#pragma omp parallel for
	for (int y = 0; y < m_height; ++y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			Vec3d& resCol = m_resD.at<Vec3d>(y, x);
			const Vec3d& col = m_cntLabD.at<Vec3d>(y, x);
			const Vec3d& a = m_a.at<Vec3d>(y, x);
			const Vec3d& b = m_b.at<Vec3d>(y, x);

			resCol = Vec3d(
				min(max(col[0] * a[0] + b[0], 0.0), 1.0),
				min(max(col[1] * a[1] + b[1], 0.0), 1.0),
				min(max(col[2] * a[2] + b[2], 0.0), 1.0));

#ifdef ENABLE_VIS
			m_aVis.at<Vec3b>(y, x) = Vec3b(
				min(max(int(a[0] * 50), 0), 255),
				min(max(int(a[1] * 50), 0), 255),
				min(max(int(a[2] * 50), 0), 255)
				);

			m_bVis.at<Vec3b>(y, x) = Vec3b(
				min(max(int(b[0] * 255 + 127), 0), 255),
				min(max(int(b[1] * 255 + 127), 0), 255),
				min(max(int(b[2] * 255 + 127), 0), 255)
				);
#endif
		}
	}

	Mat refineCSLab = Mat::zeros(m_resD.size(), CV_8UC3);
	m_resD.convertTo(refineCSLab, CV_8U, 255.0);
	cvtColor(refineCSLab, m_res, CV_Lab2BGR);

	Mat refineCSInitLab = Mat::zeros(m_resInitD.size(), CV_8UC3);
	m_resInitD.convertTo(refineCSInitLab, CV_8U, 255.0);
	cvtColor(refineCSInitLab, m_resInit, CV_Lab2BGR);

	Mat refineCSNonlocalLab = Mat::zeros(m_resNonlocalD.size(), CV_8UC3);
	m_resNonlocalD.convertTo(refineCSNonlocalLab, CV_8U, 255.0);
	cvtColor(refineCSNonlocalLab, m_resNonlocal, CV_Lab2BGR);
}
