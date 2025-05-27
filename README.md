C++实现ONNXRuntime平台的完整推理流程，可使用模型有YOLOv5、YOLOv8、RT-DETR等。

本程序可运行在任何电脑上，显卡是非必要的，内含详细代码注释。

#### 1.开发环境及依赖

VisualStudio2022（C++开发环境），OpenCV 4.8.0，onnxruntime-gpu-1.17.3，CUDA（不限制版本和显卡型号）

关于显卡的说明：即使没有Nvidia显卡，也可以使用onnxruntime的GPU版本的包。ONNX模型的推理不完全依赖Nvidia显卡，没有显卡也可以正常运行程序，只不过推理速度稍慢。

#### 2.使用步骤

1.准备工作，下载OpenCV 4.8.0、onnxruntime-gpu-1.17.3。

> 下载地址：
>
> https://github.com/microsoft/onnxruntime/releases/tag/v1.17.3
>
> https://opencv.org/releases/

2.拉取项目到本地。

3.为项目配置：附加包含目录、附加库目录、附加依赖项。

包含目录的路径

> \opencv480\build\include
>
> \onnxruntime-win-x64-gpu-1.17.3\include

附加库目录的路径

> \opencv480\build\x64\vc16\lib
>
> \onnxruntime-win-x64-gpu-1.17.3\lib

附加依赖项的路径

> onnxruntime.lib
>
> onnxruntime_providers_cuda.lib
>
> onnxruntime_providers_shared.lib
>
> opencv_world480d.lib

4.把dll文件放到执行文件exe的同级目录。

> \opencv480\build\x64\vc16\bin 中的 opencv_world480d.dll。
>
> \onnxruntime-win-x64-gpu-1.17.3\lib 中的 onnxruntime.dll、onnxruntime_providers_cuda.dll、onnxruntime_providers_shared.dll。

5.在OnnxYolov5.cpp的同级目录下新建assets文件夹，内部放入自己的 .onnx 模型文件以及图片。

6.编译执行。
