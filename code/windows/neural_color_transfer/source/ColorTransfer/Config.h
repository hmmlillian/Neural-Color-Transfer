#ifndef CONFIG_H
#define CONFIG_H

#define GPU_GRID           24
#define MAX_SIZE           1000
#define MIN_VAL            -(1e8)
#define MAX_VAL            1e8
#define ZERO_THRESH        1e-8
//#define ENABLE_VIS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>

using namespace cv;

static const int RandomColorList[260] =
{
	0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0x00FFFF, 0xFF00FF, 0x9F7262, 0xD31B4B,
	0x48AA9E, 0x42FB40, 0x3F21D8, 0x04B383, 0x188C50, 0xDBF8B0, 0x9C96EA, 0x39C3C3,
	0xBF2688, 0x46CBC8, 0xDD979E, 0xC4DC91, 0x9D161C, 0x87F9F8, 0x135CB6, 0x5DB6EE,
	0xE43484, 0xC8A9E3, 0x269B97, 0xEADA0A, 0x203BC7, 0xF949DC, 0x115C9E, 0x92723C,
	0xE06264, 0xACB122, 0xF9E5B2, 0x953E82, 0x5BF530, 0x398773, 0xDDEAB2, 0x3EC10A,
	0x21D7C8, 0xCB0373, 0x26E79D, 0xD33755, 0x66FAA7, 0x8DC6AC, 0x5630D8, 0x76BA99,
	0x3E2816, 0xEF8475, 0x9E8B07, 0x036A64, 0x578371, 0x6EE4D4, 0xC21A7E, 0x2D9CDF,
	0x5978EE, 0x09AA85, 0x7FFFA7, 0x5E0D31, 0xFA6354, 0xF7FF00, 0x1BF7D7, 0x5BC6CA,
	0xE78A33, 0x4850A0, 0x416CCC, 0xE31007, 0x32CCBC, 0x612976, 0xE43AEC, 0x4C2633,
	0xD34451, 0x0DC0A8, 0x059316, 0xF1399C, 0x6D801D, 0x5C894F, 0x7B7E2D, 0x35AEE6,
	0x059BD7, 0x8B1FBB, 0x280CF3, 0x421F44, 0x8CE16A, 0xE5DB80, 0x4BBAAD, 0x96FDF4,
	0xEA3D68, 0xBF51CB, 0xC36DC1, 0x964F0E, 0xE1F667, 0xADA7F7, 0x407208, 0xF0A082,
	0xF13660, 0x3CA266, 0xA37E6B, 0x3B6E2A, 0xDA4421, 0xF42FAA, 0x37246F, 0x2609CB,
	0xE7F700, 0x2AB1E9, 0x5ACB91, 0xE8CFDA, 0x38ECE7, 0x0AD7AA, 0x425BB0, 0x587102,
	0x0FB7A4, 0x15955E, 0xAA22DE, 0x712AB9, 0x4BB9AA, 0x267BCA, 0x9C3668, 0xFDB0FF,
	0xFFFE7D, 0xA80150, 0x5DA041, 0xF2DA52, 0xD612C8, 0x8E7CC0, 0x736A32, 0x7FBDF8,
	0x26F7A5, 0x831B61, 0x3617A1, 0x99307C, 0xC2105A, 0x1A51FF, 0x4D8E80, 0x8BCAB0,
	0x45B86B, 0xE0E5B2, 0xDD39EF, 0xFA426A, 0x421F71, 0xFFD06D, 0xCCA762, 0x35F571,
	0xAE68B6, 0x78778B, 0x6FD2DE, 0xBC4896, 0x7D9B70, 0x8269F9, 0x916DF4, 0x375055,
	0x17C396, 0x2BADFA, 0xB8D061, 0x134369, 0x15D33B, 0xB71E85, 0xA22AE4, 0x6191C5,
	0x68F62B, 0x1FD245, 0xE11681, 0xA60B47, 0x6A633E, 0x002C2B, 0xAD003A, 0x9E0B2B,
	0x9E6FBF, 0x0EC73D, 0xBE7C7D, 0xC75F23, 0xAFEB1B, 0x7BE547, 0x0398C3, 0x36D9C0,
	0x90DEAD, 0x1D003D, 0x4C43E1, 0x75BB66, 0x846221, 0x4F7B3F, 0x022648, 0xE8DB21,
	0x70957C, 0x815EE0, 0x427A8D, 0x87F3ED, 0xCD010F, 0x138EC9, 0x5332AB, 0x043099,
	0x59B575, 0x7BF4CD, 0x7F984B, 0x491447, 0xF506D0, 0x9EDE76, 0x12959A, 0x74AA63,
	0xAACBAD, 0x49B5AB, 0x7B02C3, 0xC48140, 0x0AEE0E, 0xDFA5ED, 0x21115A, 0xAB2CEB,
	0x4BB1A4, 0xC8D7A3, 0x5F38D0, 0x878083, 0xDCE393, 0x7DCB20, 0x8DFFD1, 0x2083D6,
	0xD1DAD8, 0x6A407A, 0xD9C460, 0x5BC4E9, 0xB12984, 0x72DC8E, 0xC3D1F7, 0xC91053,
	0xDCC447, 0x50D828, 0x06B059, 0xFBE8CA, 0x44EEEB, 0x9C7ACB, 0x1ED2E9, 0x296BB7,
	0x5935DA, 0xEE7E49, 0x29CC74, 0xFB2D0C, 0x465630, 0x81436A, 0x8EC51E, 0xECDEF0,
	0x73E015, 0x5B7905, 0xE0996A, 0x5D5143, 0x0942C9, 0xF4BF05, 0x5F2EE2, 0x429CD4,
	0x6A9687, 0x68FF80, 0xA5F844, 0xBFD14D, 0x3FDAF1, 0x165F10, 0x47E2C0, 0x3FFD43,
	0x59DA35, 0xEE497E, 0x2974CC, 0x2D0CFB
};


class Config
{
public:
	Config()
	{
		/// adjustable parameters
		m_reverseWeight  = 2.0;
		m_varEpslon      = 0.60;
		m_nonlocalWeight = 2.0;
		m_localWeight    = 0.125;
		m_wlsLamdaInit   = 0.024;
		/// end

		m_clusterNum = 10;
		m_kNum       = 8;
		m_patchSize  = 3;
		m_wlsAlpha   = 1.2;	
	}

	~Config()
	{
		
	}

public:

	// paths
	string m_inputDir;
	string m_outputDir;
	string m_modelDir;

	// variable
	double m_reverseWeight;
	double m_varEpslon;
	double m_nonlocalWeight;
	double m_localWeight;
	double m_wlsLamdaInit;
	
	// constant
	double m_kNum;
	double m_patchSize;
	double m_wlsAlpha;
	double m_clusterNum;
};

#endif