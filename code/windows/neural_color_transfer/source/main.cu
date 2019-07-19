#include "GeneralizedPatchmatch.cuh"
#include "Classifier.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

#include "ColorTransfer/ColorTransfer.h"
#include "ColorTransfer/Config.h"
#include "CmdLine.h"

#include <Windows.h>
#include <direct.h>

using namespace Utility;

vector<int> jpgQuality(1, 99);

struct Parameters
{
	std::vector<std::string> layers; //which layers are used as content

	int EM;
	int iter;
	bool flag_constraint;
	float constraint;
	float energy;
};


void get_input(CmdLine& cmdLine, Config& config, int& gpuId)
{
	cmdLine.Param("m", config.m_modelDir,  "Directory of network models.");
	cmdLine.Param("i", config.m_inputDir,  "Input directory of content and style images and pairs.txt.");
	cmdLine.Param("o", config.m_outputDir, "Output directory of result images.");
	cmdLine.Param("g", gpuId,              "GPU ID (default: 0).");
	
	// parameters to be fine-tuned
	cmdLine.Param("bds", config.m_reverseWeight, "Weight of reverse color in BDS voting (default: 2.0).");
	
	// parameters usually fixed
	cmdLine.Param("eps", config.m_varEpslon,      "Eps is used to avoid dividing zero (default: 0.6 with range in [0-255]).");
	cmdLine.Param("nl",  config.m_nonlocalWeight, "Weight of nonlocal constraint (default: 0.4.");
	cmdLine.Param("l",   config.m_localWeight,    "Weight of local constraitn (default: 0.001).");
	cmdLine.Param("w",   config.m_wlsLamdaInit,   "Initial value of WLS weight (default: 0.0234375).");
}


__host__ void transfer_color_single_bds(Mat& refineCS, Classifier& classifier_C, Classifier& classifier_S,
	const Config& config, const Mat& cnt, const Mat& stl, const char* preName)
{
	const int param_size = 11;
	const int patch_size = config.m_patchSize;

	//set parameters
	Parameters params;
	params.layers.push_back("conv5_1");
	params.layers.push_back("conv4_1");
	params.layers.push_back("conv3_1");
	params.layers.push_back("conv2_1");
	params.layers.push_back("conv1_1");

	int numlayer = params.layers.size();
	char fname[256];

	params.EM = 3;
	params.iter = 10;
	params.flag_constraint = 0;
	params.constraint = 10.0;
	params.energy = 100.0;

	vector<Mat> cntImages(numlayer), stlImages(numlayer);
	cntImages[numlayer - 1] = cnt.clone();
	stlImages[numlayer - 1] = stl.clone();
	
	Mat img_content = cntImages[numlayer - 1];
	Mat img_style = stlImages[numlayer - 1];

	std::vector<int> range;
	int maxLen = max(max(max(img_content.cols, img_content.rows), img_style.cols), img_style.rows);
	range.push_back(maxLen / 16);
	range.push_back(maxLen / 32);
	range.push_back(maxLen / 64);
	range.push_back(32);
	range.push_back(32);

	ColorTransfer colorTransfer(config, img_content);

	//load caffe
	std::vector<float *> data_C1, data_C;
	data_C1.resize(params.layers.size());
	data_C.resize(params.layers.size());

	std::vector<Dim> data_C_size;
	data_C_size.resize(params.layers.size());
	classifier_C.Predict(img_content, params.layers, data_C1, data_C, data_C_size);

	std::vector<float *> data_S1;
	std::vector<Dim> data_S_size;

	data_S1.resize(params.layers.size());
	data_S_size.resize(params.layers.size());

	classifier_S.Predict(img_style, params.layers, vector<float*>(), data_S1, data_S_size);

	for (int l = numlayer - 2; l >= 0; --l)
	{
		resize(cntImages[l + 1], cntImages[l], Size(data_C_size[l].width, data_C_size[l].height), 0, 0, CV_INTER_LINEAR);
		resize(stlImages[l + 1], stlImages[l], Size(data_S_size[l].width, data_S_size[l].height), 0, 0, CV_INTER_LINEAR);
	}

	clock_t start, finish;
	double duration;
	start = clock();

	int *params_host, *params_device_ab, *params_device_ba;
	unsigned int *ann_device, *bnn_device;
	float *annd_device, *bnnd_device;

	int ful_ann_size = img_content.cols * img_content.rows;
	int ful_bnn_size = img_style.cols * img_style.rows;

	params_host = (int *)malloc(param_size * sizeof(int));

	cudaMalloc(&params_device_ab, param_size * sizeof(int));
	cudaMalloc(&params_device_ba, param_size * sizeof(int));

	cudaMalloc(&ann_device, ful_ann_size * sizeof(unsigned int));
	cudaMalloc(&annd_device, ful_ann_size * sizeof(float));

	cudaMalloc(&bnn_device, ful_bnn_size * sizeof(unsigned int));
	cudaMalloc(&bnnd_device, ful_bnn_size * sizeof(float));
	
	// data used in graphcut and wls pyramid
	float* ann_err_host = (float*)malloc(ful_ann_size * sizeof(float));
	float* bnn_err_host = (float*)malloc(ful_bnn_size * sizeof(float));
	float* bds_err_host = (float*)malloc(ful_ann_size * sizeof(float));
	unsigned int* ann_flow_host = (unsigned int*)malloc(ful_ann_size * sizeof(unsigned int));
	unsigned int* bnn_flow_host = (unsigned int*)malloc(ful_bnn_size * sizeof(unsigned int));

	int channels = data_C_size[0].channel;
	int slice = data_C_size[0].width * data_C_size[0].height;
	float *inputFeatureData = (float*)malloc(slice * channels * sizeof(float));
	float *features = (float*)malloc(slice * channels * sizeof(float));
	cudaMemcpy(inputFeatureData, data_C1[0], slice * channels * sizeof(float), cudaMemcpyDeviceToHost);

	#pragma omp parallel for
	for (int y = 0; y < data_C_size[0].height; ++y)
	{
		for (int x = 0; x < data_C_size[0].width; ++x)
		{
			float resp = 0.f;
			int id = y * data_C_size[0].width + x;
			for (int c = 0; c < channels; ++c)
			{
				resp += inputFeatureData[slice * c + id] * inputFeatureData[slice * c + id];
			}
			if (resp > MIN_VAL)
			{
				resp = sqrt(resp);
				for (int c = 0; c < channels; ++c)
				{
					features[id * channels + c] = inputFeatureData[slice * c + id] / resp;
				}
			}
		}
	}

	Mat dvisMat;
	colorTransfer.clusterFeastures(dvisMat, features, data_C_size[0].width, data_C_size[0].height, channels);

#ifdef ENABLE_VIS
	sprintf(fname, "%s\\%s_cluster_small.png", config.m_outputDir.c_str(), preName);
	imwrite(fname, dvisMat);
#endif

	free(features);
	free(inputFeatureData);

	//feature match
	for (int curr_layer = 0; curr_layer < numlayer; curr_layer++) //from 32 to 512
	{
		int pre_layer = curr_layer - 1;

		clock_t pstart, pfinish;
		double pduration;
		pstart = clock();

		int aw = data_C_size[curr_layer].width;
		int ah = data_C_size[curr_layer].height;
		int cur_ann_size = aw * ah;

		int bw = data_S_size[curr_layer].width;
		int bh = data_S_size[curr_layer].height;
		int cur_bnn_size = bw * bh;

		int  cur_num_a = data_C_size[curr_layer].channel * cur_ann_size;
		dim3 cur_blocksPerGrid_a(aw / GPU_GRID + 1, ah / GPU_GRID + 1, 1);
		dim3 cur_threadsPerBlock_a(GPU_GRID, GPU_GRID, 1);

		int  cur_num_b = data_S_size[curr_layer].channel * cur_bnn_size;
		dim3 cur_blocksPerGrid_b(bw / GPU_GRID + 1, bh / GPU_GRID + 1, 1);
		dim3 cur_threadsPerBlock_b(GPU_GRID, GPU_GRID, 1);

		//set parameteers-ab		
		params_host[0] = data_C_size[curr_layer].channel;
		params_host[1] = ah;
		params_host[2] = aw;
		params_host[3] = bh;
		params_host[4] = bw;
		params_host[5] = patch_size;
		params_host[6] = params.iter;
		params_host[7] = range[curr_layer];
		params_host[8] = params.flag_constraint;
		params_host[9] = params.constraint;
		params_host[10] = 1.0;

		//copy to device
		cudaMemcpy(params_device_ab, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//set parameters-ba	
		params_host[0] = data_S_size[curr_layer].channel;
		params_host[1] = bh;
		params_host[2] = bw;
		params_host[3] = ah;
		params_host[4] = aw;

		//copy to device
		cudaMemcpy(params_device_ba, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//initialize ann if needed
		if (curr_layer == 0)//initialize, rows and cols both less than 32, just use one block
		{
			init_Ann_kernel << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(ann_device, params_device_ab);
			init_Ann_kernel << <cur_blocksPerGrid_b, cur_threadsPerBlock_b >> >(bnn_device, params_device_ba);
		}
		else
		{
			//upsampling, notice this block's dimension is twice the ann at this point
			unsigned int * ann_tmp;
			cudaMalloc(&ann_tmp, cur_ann_size * sizeof(unsigned int));
			upSample_kernel << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(ann_device, ann_tmp, params_device_ab,
				data_C_size[pre_layer].width, data_C_size[pre_layer].height);//get new ann_device
			cudaMemcpy(ann_device, ann_tmp, cur_ann_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
			cudaFree(ann_tmp);

			unsigned int * bnn_tmp;
			cudaMalloc(&bnn_tmp, cur_bnn_size * sizeof(unsigned int));
			upSample_kernel << <cur_blocksPerGrid_b, cur_threadsPerBlock_b >> >(bnn_device, bnn_tmp, params_device_ba,
				data_S_size[pre_layer].width, data_S_size[pre_layer].height);//get new ann_device
			cudaMemcpy(bnn_device, bnn_tmp, cur_bnn_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
			cudaFree(bnn_tmp);
		}

		//normarlize two data
		float *Ndata_C1, *Ndata_S1, *response_C1;
		cudaMalloc(&Ndata_C1, cur_num_a * sizeof(float));
		cudaMalloc(&Ndata_S1, cur_num_b * sizeof(float));
		cudaMalloc(&response_C1, cur_ann_size * sizeof(float));

		if (data_C_size[curr_layer].channel <= 3)
		{
			norm1(Ndata_S1, data_S1[curr_layer], NULL, data_S_size[curr_layer]);
		}
		else
		{
			norm(Ndata_S1, data_S1[curr_layer], NULL, data_S_size[curr_layer]);
		}

		if (data_C_size[curr_layer].channel <= 3)
		{
			norm1(Ndata_C1, data_C1[curr_layer], response_C1, data_C_size[curr_layer]);
		}
		else
		{
			norm(Ndata_C1, data_C1[curr_layer], response_C1, data_C_size[curr_layer]);
		}

		Mat down_cnt = cntImages[curr_layer];
		Mat down_stl = stlImages[curr_layer];
		Mat smlRes;
		float bds_wgt = config.m_reverseWeight;
		
		//cs patchmatch	
		patchmatch_single << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(Ndata_C1, Ndata_S1, NULL, ann_device, annd_device, params_device_ab);
		patchmatch_single << <cur_blocksPerGrid_b, cur_threadsPerBlock_b >> >(Ndata_S1, Ndata_C1, NULL, bnn_device, bnnd_device, params_device_ba);

		cudaMemcpy(ann_err_host, annd_device, cur_ann_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(ann_flow_host, ann_device, cur_ann_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(bnn_flow_host, bnn_device, cur_bnn_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(bnn_err_host, bnnd_device, cur_bnn_size * sizeof(float), cudaMemcpyDeviceToHost);

		smlRes = reconstruct_bds(down_cnt, down_stl, ann_flow_host, bnn_flow_host, patch_size, 1.f, bds_wgt);

		float *vote_Ndata_C1;
		float *vote_weight;
		float *copy_Ndata_C1;

		cudaMalloc(&vote_Ndata_C1, cur_num_a * sizeof(float));
		cudaMalloc(&copy_Ndata_C1, cur_num_a * sizeof(float));
		cudaMalloc(&vote_weight, cur_ann_size * sizeof(float));

		cudaMemcpy(copy_Ndata_C1, Ndata_C1, cur_num_a * sizeof(float), cudaMemcpyDeviceToDevice);

		avg_vote_bds_a << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(ann_device, data_S1[curr_layer], vote_Ndata_C1, vote_weight, params_device_ab, 1.f);
		avg_vote_bds_b << <cur_blocksPerGrid_b, cur_threadsPerBlock_b >> >(bnn_device, data_S1[curr_layer], vote_Ndata_C1, vote_weight, params_device_ab, bds_wgt);
		avg_vote_bds << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(vote_Ndata_C1, vote_weight, params_device_ab);

		if (data_C_size[curr_layer].channel <= 3)
		{
			norm1(Ndata_C1, vote_Ndata_C1, response_C1, data_C_size[curr_layer]);
		}
		else
		{
			norm(Ndata_C1, vote_Ndata_C1, response_C1, data_C_size[curr_layer]);
		}

		feature_distance << <cur_blocksPerGrid_a, cur_threadsPerBlock_a >> >(copy_Ndata_C1, Ndata_C1, annd_device, params_device_ab);
			
		cudaMemcpy(bds_err_host, annd_device, cur_ann_size * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(vote_Ndata_C1);
		cudaFree(vote_weight);
		cudaFree(copy_Ndata_C1);

		cudaFree(Ndata_C1);
		cudaFree(Ndata_S1);
		cudaFree(response_C1);

		pfinish = clock();
		pduration = (double)(pfinish - pstart) / CLOCKS_PER_SEC;

		printf("Patch Match Time: %lf sec.\n", pduration);

#ifdef ENABLE_VIS
		Mat flow;
		flow = reconstruct_flow(down_cnt, down_stl, ann_flow_host, patch_size);
		sprintf(fname, "%s\\%s_aFlow_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, flow);

		flow = reconstruct_flow(down_stl, down_cnt, bnn_flow_host, patch_size);
		sprintf(fname, "%s\\%s_bFlow_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, flow);

		sprintf(fname, "%s\\%s_tCnt_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, down_cnt);

		sprintf(fname, "%s\\%s_tStl_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, down_stl);
		
#endif

		Mat cntLab = Mat::zeros(down_cnt.size(), CV_8UC3);
		cvtColor(down_cnt, cntLab, CV_BGR2Lab);
		Mat cntLabD = Mat::zeros(down_cnt.size(), CV_64FC3);
		Mat cntRgbD = Mat::zeros(down_cnt.size(), CV_64FC3);
		cntLab.convertTo(cntLabD, CV_64F, 1.0 / 255.0);
		down_cnt.convertTo(cntRgbD, CV_64F, 1.0 / 255.0);
	
		Mat visMat;
		colorTransfer.findKnns(visMat, cntLabD, pow(2, curr_layer));

#ifdef ENABLE_VIS
		sprintf(fname, "%s\\%s_knn_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, visMat);
#endif
		int samples = pow(2, numlayer - 1 - curr_layer);
		int estimatePatchSize = patch_size;
		
		colorTransfer.build_accumTable_downsample(cntLab, colorTransfer.getCntTable(), colorTransfer.getCntTable2());
		
		Mat stlLab = Mat::zeros(smlRes.size(), CV_8UC3);
		cvtColor(smlRes, stlLab, CV_BGR2Lab);
		colorTransfer.build_accumTable_downsample(stlLab, colorTransfer.getStlTable(), colorTransfer.getStlTable2());

		Mat stlLabD = Mat::zeros(smlRes.size(), CV_64FC3);
		stlLab.convertTo(stlLabD, CV_64F, 1.0 / 255.0);

		colorTransfer.transfer_color_downsample(bds_err_host, cntRgbD, cntLabD, stlLabD,
			estimatePatchSize, curr_layer, data_C_size[curr_layer].width, data_C_size[curr_layer].height, samples, down_cnt, smlRes);
		
		refineCS = colorTransfer.getRes();

#ifdef ENABLE_VIS
		const Mat& errMap = colorTransfer.getErrorMap();
		sprintf(fname, "%s\\%s_errMap_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, errMap);

		const Mat& refineCSInit = colorTransfer.getResInit();
		sprintf(fname, "%s\\%s_refine_init_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, refineCSInit);

		const Mat& refineCSNonlocal = colorTransfer.getResNonlocal();
		sprintf(fname, "%s\\%s_refine_nonlocal_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, refineCSNonlocal);

		const Mat& aVis = colorTransfer.getAVis();
		sprintf(fname, "%s\\%s_aVis_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, aVis);

		const Mat& aVisInit = colorTransfer.getAVisInit();
		sprintf(fname, "%s\\%s_aVis_init_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, aVisInit);

		const Mat& aNonlocal = colorTransfer.getANonlocal();
		sprintf(fname, "%s\\%s_aVis_nonlocal_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, aNonlocal);

		const Mat& bVis = colorTransfer.getBVis();
		sprintf(fname, "%s\\%s_bVis_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, bVis);

		const Mat& bVisInit = colorTransfer.getBVisInit();
		sprintf(fname, "%s\\%s_bVis_init_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, bVisInit);

		const Mat& bNonlocal = colorTransfer.getBNonlocal();
		sprintf(fname, "%s\\%s_bVis_nonlocal_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, bNonlocal);

		const Mat& patchVis = colorTransfer.getPatchVis();
		sprintf(fname, "%s\\%s_patchVis_%d.png", config.m_outputDir.c_str(), preName, curr_layer);
		imwrite(fname, patchVis);
#endif

		if (curr_layer < numlayer - 1)
		{
			classifier_C.Predict(refineCS, params.layers, data_C1);
		}
	}

	cudaFree(params_device_ab);
	cudaFree(params_device_ba);

	cudaFree(ann_device);
	cudaFree(annd_device);
	cudaFree(bnn_device);
	cudaFree(bnnd_device);

	free(params_host);
	free(ann_flow_host);
	free(bnn_flow_host);
	free(ann_err_host);
	free(bnn_err_host);
	free(bds_err_host);

	for (int i = 0; i < numlayer; i++)
	{
		cudaFree(data_S1[i]);
	}

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;

	printf("**Finished Time: %lf sec.\n", duration);
}

void transfer_single(Classifier& classifier_C, Classifier& classifier_S, Config& config, int gpu_id)
{
	mkdir(config.m_outputDir.c_str());

	// read file names from pairs.txt
	string pairsFile = config.m_inputDir + "/pairs.txt";
	FILE* fp = fopen(pairsFile.c_str(), "r");
	if (fp == NULL)
	{
		printf("Error: File %s does not exist in the input directory.\n", pairsFile.c_str());
	}

	char cntFile[260];
	char stlFile[260];
	float bdsWeight = 0.f;
	while (fscanf(fp, "%s %s %f\n", cntFile, stlFile, &bdsWeight) != EOF)
	{
		string cntStr(cntFile);
		string stlStr(stlFile);
		config.m_reverseWeight = bdsWeight;

		printf("-----------------***********************----------------------\n");
		printf("Content: %s, style: %s, BDS weight: %f.\n", cntStr.c_str(), stlStr.c_str(), config.m_reverseWeight);

		cntStr = config.m_inputDir + "/" + cntStr;
		stlStr = config.m_inputDir + "/" + stlStr;

		Mat cnt = imread(cntStr);
		if (!cnt.cols)
		{
			printf("Error: Fail reading content image: %s\n", cntStr.c_str());
			continue;
		}
		printf("\n**Read content file: %s, w = %d, h = %d\n", cntStr.c_str(), cnt.cols, cnt.rows);

		Mat stl = imread(stlStr.c_str());
		if (!stl.cols)
		{
			printf("Error: Fail reading style image: %s\n", stlStr.c_str());
			continue;
		}
		printf("Read style file: %s, w = %d, h = %d\n", stlStr.c_str(), stl.cols, stl.rows);

		// resize images to make longer side no larger than MAX_SIZE defined in Config.h
		if (cnt.cols > MAX_SIZE || cnt.rows > MAX_SIZE)
		{
			int cw = MAX_SIZE;
			int ch = cw / (float)cnt.cols * cnt.rows;
			if (cnt.cols < cnt.rows)
			{
				ch = MAX_SIZE;
				cw = ch / (float)cnt.rows * cnt.cols;
			}
			resize(cnt, cnt, Size(cw, ch));
		}

		if (stl.cols > MAX_SIZE || stl.rows > MAX_SIZE)
		{
			int sw = MAX_SIZE;
			int sh = sw / (float)stl.cols * stl.rows;
			if (stl.cols < stl.rows)
			{
				sh = MAX_SIZE;
				sw = sh / (float)stl.rows * stl.cols;
			}
			resize(stl, stl, Size(sw, sh));
		}

		char fileName[260];
		int pos = cntStr.find_last_of('\\/') + 1;
		int len = cntStr.find_last_of('.') - pos;
		string cntPre = cntStr.substr(pos, len);
		pos = stlStr.find_last_of('\\/') + 1;
		len = stlStr.find_last_of('.') - pos;
		string stlPre = stlStr.substr(pos, len);

		sprintf(fileName, "%s_%s", cntPre.c_str(), stlPre.c_str());

		Mat refineCS;
		transfer_color_single_bds(refineCS, classifier_C, classifier_S, config, cnt, stl, fileName);

		sprintf(fileName, "%s/%s_%s_%2.2f.png", config.m_outputDir.c_str(), cntPre.c_str(), stlPre.c_str(), config.m_reverseWeight);
		imwrite(fileName, refineCS);
		printf("Final output file: %s.\n\n", fileName);
	}

	fclose(fp);
}


int main(int argc, char** argv)
{
	CmdLine cmdLine;
	Config* config = new Config();

	int gpuID = 0;
	int type  = 2;
	int cid = 0;
	int sid = 0;

	get_input(cmdLine, *config, gpuID);
	if (!cmdLine.Parse(argc, argv)) 
	{
		return -1;
	}

	int count = 1;
	cudaGetDeviceCount(&count);
	cudaSetDevice(gpuID);
	cudaDeviceReset();
	printf("The number of device is: %d, set device %d.\n", count, gpuID);

	size_t freeMem = 0, totalMem = 0;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("Total Memory = %lu, Free Memory = %lu.\n", totalMem, freeMem);

	// initialize neural network log
	//::google::InitGoogleLogging("neural_color_transfer");

	string model_file = "\\vgg19\\VGG_ILSVRC_19_layers_deploy.prototxt";
	string trained_file = "\\vgg19\\VGG_ILSVRC_19_layers.caffemodel";
	string root = config->m_modelDir;
	
	string model_path = root + model_file;
	string train_path = root + trained_file;
	Classifier classifier_C(model_path, train_path);
	Classifier classifier_S(model_path, train_path);

	transfer_single(classifier_C, classifier_S, *config, gpuID);

	//::google::ShutdownGoogleLogging();

	delete config;
	return 0;
}