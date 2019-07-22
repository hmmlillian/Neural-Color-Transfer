# Neural Color Transfer

This is the implementation of single-reference color transfer proposed in the paper [**Progressive Color Transfer with Dense Semantic Correspondences**](https://arxiv.org/abs/1807.06587) by [Mingming He](http://mingminghe.com/), [Jing Liao](https://liaojing.github.io/html/index.html), [Dongdong Chen](http://www.dongdongchen.bid/), [Lu Yuan](http://www.lyuan.org/) and [Pedro V. Sander](http://www.cse.ust.hk/~psander/) in ACM Transactions on Graphics (2019).


## Introduction

**Neural Color Transfer** is a progressive color transfer framework, which jointly optimizes dense semantic correspondencesin the deep feature domain and the local color transfer inthe image domain.

![image](https://github.com/hmmlillian/Neural-Color-Transfer/blob/master/demo/intro.pdf)

Given two input images (one color source image *S* and one color reference image) which share semantically-related content, but may vary dramatically in appearance or structure, the proposed framework first estimates dense correspondence between them using deep features (extracted from VGG19 at level *L*) and then applies local color transfer to the source image *S* based on the correspondence. The process repeats from high level (*L=5*) to low level (*L=1*).

For more results, please refer to our [Supplementary](http://mingminghe.com/neural_color_transfer/comparison.html).


## Disclaimer

This is a C++ combined with CUDA implementation. It is worth noticing that:
- The codes are built based on [Caffe](https://github.com/Microsoft/caffe) and [Deep Image Analogy](https://github.com/msracver/Deep-Image-Analogy).
- The codes only have been tested on Windows 10 and Windows Server 2012 R2 with CUDA 8 or 7.5.
- The codes only have been tested on several Nvidia GPU: Titan X, Titan Z, K40, GTX1070, GTX770.
- The size of input image is limited, mostly the longer side of input images are around 700 and should not be large than 1000.


## Getting Started

### Prerequisites
- Windows (64bit)
- CUDA 8 or CUDA 75 (with CuDNN)
- [Intel® Parallel Studio XE for Windows](https://software.intel.com/en-us/parallel-studio-xe/choose-download/free-trial-cluster-windows-c-fortran)
- Visual Studio 2013

### Build
The codes requires compiling in Visual Studio as follows:
- Build [Caffe](http://caffe.berkeleyvision.org/) at first. Just follow the tutorial [here](https://github.com/Microsoft/caffe).
- Open solution ```Caffe``` under ```code\windows\``` and add ```neural_color_transfer.vcxproj``` under ```code\Windows\neural_color_transfer\```.
- Set <MKL_DIR> to the installation directory of Intel® Parallel Studio XE for Windows, e.g., ```<MKL_DIR>..\..\..\NugetPackages\mkl\compilers_and_libraries_2018.1.156\windows</MKL_DIR>```.
- Set <OPENCV_DIR> to the directory of OpenCV library, e.g., ```<OPENCV_DIR>..\..\..\NugetPackages\OpenCV.2.4.10\build\native</OPENCV_DIR>```.
- Build project ```neural_color_transfer.vcxproj```.

### Download Models
You need to download models before running a demo.
- Go to ```demo\model\vgg19\``` folder and download [model]( 
  http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel).

### Demo
We prepare an example under the folder ```demo\``` with:

(1) Input data folder ```example\``` including two parts:
- A folder ```input\``` with the input images (color source images and color reference images) inside.
- A file ```pairs.txt``` to specify a source, a reference and a BDS weight (2.0 as default) as an example in each line, e.g., 
  ```
  in/in0.png in/tar0.png 2.0
  in/in1.png in/tar1.png 2.0
  ...
  ```

(2) Executable script ```run.bat``` including one command:
  ```
  neural_color_transfer.exe -m [MODEL_DIR] -i [INPUT_ROOT_DIR] -o [OUTPUT_DIR] -g [GPU_ID]
  e.g., exe\neural_color_transfer.exe -m model\ -i example\ -o example\res\ -g 0
  ```  

### Run
We provide pre-built executable files [exe](https://drive.google.com/file/d/1r7zfDIU_S99hmWKpNXLxMDH-3iuCLSoQ/view?usp=sharing). Please download and unzip in folder ```demo\```, and try them.

### Tips
- The proposed algorithm is more applicable to transfer color between images with semantically-related content.
- Ajusting the parameter of BDS weight (defined in pairs.txt, regularly in [0, 4]) according to similarity of input images leads to better results if their content are not highly related, smaller value for less similar pair.


## Citation
If you find **Neural Color Transfer** helpful for your research, please consider citing:
```
@article{he2019progressive,
  title={Progressive color transfer with dense semantic correspondences},
  author={He, Mingming and Liao, Jing and Chen, Dongdong and Yuan, Lu and Sander, Pedro V},
  journal={ACM Transactions on Graphics (TOG)},
  volume={38},
  number={2},
  pages={13},
  year={2019},
  publisher={ACM}
}
```
