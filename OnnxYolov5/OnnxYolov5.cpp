#include <iostream>
#include <string>
#include <fstream> // For file existence check

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"

int main()
{
	std::string modelPath0 = "./assets/best0113.onnx";
	std::wstring modelPath = std::wstring(modelPath0.begin(), modelPath0.end());	// 这步必须有
	std::string imagePath = "./assets/3.png";

	std::cout << "Model Path: " << modelPath0 << std::endl;
	std::cout << "Image Path: " << imagePath << std::endl;
	std::cout << std::endl;

	// 检查模型文件是否能正常读取
	std::ifstream f(modelPath.c_str());
	if (!f.good()) {
		std::cerr << "Error: Model file does not exist or is not accessible at: " << modelPath0 << std::endl;
		return 1;
	}
	f.close();
	std::cout << "Model file found at: " << modelPath0 << std::endl;

	// 初始化ort环境
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "object");

	// 设置会话选项
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);	// 设置图优化级别
	session_options.SetIntraOpNumThreads(2);	// 设置线程数
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);	// 选择设备，其否使用cuda
	std::cout << "GPU" << std::endl;
	// 创建会话
	Ort::Session session_(env, modelPath.c_str(), session_options);

	//输入、输出的节点个数
	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;
	Ort::AllocatorWithDefaultOptions allocator;

	// 获取输入的节点的名称，形状
	int input_c = 0;
	int input_h = 0;
	int input_w = 0;
	// 节点通常是1个，有时是多个，input_nodes_num一般等于1
	for (int i = 0; i < input_nodes_num; i++) {
		// 获取输入节点的名称并存储
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		//input_node_names.push_back(input_name.get());

		// 获取当前节点的输入形状
		auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		input_c = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format: " << input_c << "x" << input_h << "x" << input_w << std::endl;
	}

	// 获取输入的节点的名称，形状
	int output0 = 0;
	int output1 = 0;
	int output2 = 0;
	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		//output_node_names.push_back(output_name.get());

		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		output0 = outShapeInfo[0];
		output1 = outShapeInfo[1];
		output2 = outShapeInfo[2];
		std::cout << "output format: " << output0 << "x" << output1 << "x" << output2 << std::endl;
	}

	return 0;
}

