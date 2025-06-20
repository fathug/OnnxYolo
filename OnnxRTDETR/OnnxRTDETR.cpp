﻿#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

int main()
{
	float confidence = 0.45;	// 后处理中的置信度阈值

	std::string modelPath0 = "./assets/rtdetr-l.onnx";
	std::wstring modelPath = std::wstring(modelPath0.begin(), modelPath0.end());
	std::string imagePath = "./assets/5.jpg";

	std::cout << "Model Path: " << modelPath0 << std::endl;
	std::cout << "Image Path: " << imagePath << std::endl;

	// 检查模型文件是否能正常读取
	std::ifstream f(modelPath.c_str());
	if (!f.good()) {
		std::cerr << "Error: Model file does not exist or is not accessible at: " << modelPath0 << std::endl;
		return 1;
	}
	f.close(); // 关闭文件流
	std::cout << "Model file found at: " << modelPath0 << std::endl;

	// 初始化ONNXRuntime环境。Ort::Env 用于管理ONNXRuntime环境。在创建 Session 之前必须先创建 Env。
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "object");

	// 设置会话选项
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);	// 设置图优化级别
	session_options.SetIntraOpNumThreads(1);	// 设置线程数。控制单个操作符（如卷积、矩阵乘法）在执行时可以使用的线程数量。

	// 获取设备
	auto providers = Ort::GetAvailableProviders();
	for (auto p : providers)
		std::cout << "可支持设备: " << p << std::endl;
	auto isCudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");

	// CUDA手动开关
	bool isCuda = false;
	if (isCuda && (isCudaAvailable != providers.end())) {
		OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);	// 选择设备，其否使用cuda。注释掉则默认使用cpu
		std::cout << "当前使用CUDA" << std::endl;
	}
	else
	{
		std::cout << "当前使用CPU" << std::endl;
	}

	// 创建会话
	Ort::Session session_(env, modelPath.c_str(), session_options);	//模型文件的路径 (宽字符串)

	//输入、输出的节点个数
	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	// ORT的默认内存分配器
	Ort::AllocatorWithDefaultOptions allocator;

	// 获取输入节点的信息
	int input_b = 0;
	int input_c = 0;
	int input_h = 0;
	int input_w = 0;
	// 节点通常是1个，有时是多个，input_nodes_num一般等于1
	for (int i = 0; i < input_nodes_num; i++) {
		// 获取输入节点的名称并存储
		// GetInputNameAllocated获取指定索引的输入节点的名称,返回一个 Ort::AllocatedStringPtr 对象，它是一个智能指针，管理着由分配器分配的 C 字符串。
		// .get() 方法返回底层的 char* 指针。
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());

		// 获取当前节点的输入形状
		// GetTensorTypeAndShapeInfo(): 从类型信息中获取张量类型和形状信息。GetShape(): 获取张量的形状，返回一个 std::vector<int64_t>。
		auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		input_b = inputShapeInfo[0];
		input_c = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format: " << input_b << "x" << input_c << "x" << input_h << "x" << input_w << std::endl;
	}

	// 获取输出节点的信息
	int numBatch = 0;
	int numQuery = 0;
	int numAttributes = 0;

	for (int i = 0; i < output_nodes_num; i++) {
		//for (int i = 0; i < 1; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());

		// YOLOv5: [batch_size, num_predictions, num_attributes]: [1, 25200, 85]: (85 = 4 (bbox) + 1 (confidence) + 80 (class scores))
		// YOLOv8: [batch_size, num_attributes, num_predictions]: [1, 84, 8400]: (84 = 4 (bbox) + 80 (class scores))
		// RTDETR: [batch_size, num_Query, num_attributes]: [1, 300, 84]
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		numBatch = outShapeInfo[0];
		numQuery = outShapeInfo[1];
		numAttributes = outShapeInfo[2];
		std::cout << "output format: " << numBatch << "x" << numQuery << "x" << numAttributes << std::endl;
	}

	cv::Mat srcImage = cv::imread(imagePath);
	int srcWidth = srcImage.cols;
	int srcHeight = srcImage.rows;

	// 将原始图像填充为正方形，多余部分补零
	int maxLength = std::max(srcHeight, srcWidth);
	cv::Mat paddingImage = cv::Mat::zeros(cv::Size(maxLength, maxLength), CV_8UC3);
	cv::Rect roi(0, 0, srcWidth, srcHeight);
	srcImage.copyTo(paddingImage(roi));	// 贴到左上角

	// 缩放因子，用于将输出坐标映射回padding图像
	float x_factor = paddingImage.cols / static_cast<float>(input_w);
	float y_factor = paddingImage.rows / static_cast<float>(input_h);

	// blobFromImage 会将图像转换为 NCHW 格式的 4D 张量 (N=1)
	cv::Mat blob = cv::dnn::blobFromImage(paddingImage, 1.0 / 255.0,
		cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	std::cout << "blob size: " << blob.size[0] << "*" << blob.size[1] << "*" << blob.size[2] << "*" << blob.size[3] << std::endl;

	// 输入张量中元素的总数
	size_t tpixels = static_cast<rsize_t>(input_c * input_h * input_w);
	// 定义输入张量的形状 [batch_size, channels, height, width]
	std::array<int64_t, 4> input_shape_info{ 1, input_c, input_h, input_w };

	// 准备数据输入
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);	// 内存信息，指定张量数据存储在cpu中
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total() * tpixels, input_shape_info.data(), input_shape_info.size());

	// 模型输入输出所需数据(名称及其数量)，Run 方法需要输入和输出节点名称的 C 风格字符串数组
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

	// 假的输入张量，用于模型预热。与模型输入尺寸、类型一致即可
	// 假张量适用于没有图用于预热的情况，有图的情况下可使用 input_tensor_ 进行预热
	size_t dummy_input_size = static_cast<size_t>(input_b) * input_c * input_h * input_w;
	std::vector<float> dummy_input_data(dummy_input_size, 0.0f);
	std::array<int64_t, 4> dummy_input_shape{ input_b, input_c, input_h, input_w };
	auto dummy_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value dummy_input_tensor = Ort::Value::CreateTensor<float>(
		dummy_allocator_info,
		dummy_input_data.data(),
		dummy_input_data.size(),
		dummy_input_shape.data(),
		dummy_input_shape.size());

	// 模型预热
	std::cout << "\n模型正在预热 (Warm-up)..." << std::endl;
	for (int i = 0; i < 3; ++i) {	// 执行3次以上，后续一次推理速度会比较稳定
		try {
			session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &dummy_input_tensor, 1, outNames.data(), outNames.size());
		}
		catch (const std::exception& e) {
			std::cerr << "预热出错: " << e.what() << std::endl;
			return 1;
		}
	}
	std::cout << "预热完成.\n" << std::endl;

	auto inferenceStart = std::chrono::high_resolution_clock::now();

	// 执行推理
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	auto inferenceEnd = std::chrono::high_resolution_clock::now();

	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	// RTDETR: [batch_size, num_Query, num_attributes]
	// num_attributes: 每个框的属性数量 (例如 84 = cx, cy, w, h, , class_score1, class_score2, ...)
	cv::Mat det_output(numQuery, numAttributes, CV_32F, (float*)pdata);

	//// 转置，方便后续处理。YOLOv8的输出需要此操作。
	//cv::Mat det_output;
	//cv::transpose(det_output0, det_output);

	std::vector<cv::Rect> boxes;		// 目标框的坐标位置
	std::vector<float> confidences;		// 目标框的置信度
	std::vector<int> classIds;			// 目标框的类别得分

	// 遍历所有检测到的候选框 (det_output的每一行代表一个候选框)
	for (int i = 0; i < det_output.rows; i++) {
		//YOLOv5需要先进行阈值筛选
		//float confidence = det_output.at<float>(i, 4);
		//if (confidence < 0.45) {
		//	continue;
		//}

		// 获得当前目标框的所有类别得分
		cv::Mat classes_scores = det_output.row(i).colRange(4, numAttributes);

		cv::Point classIdPoint;		// 用于存储分类中的得分最大值索引(坐标)
		double score;				// 用于存储分类中的得分最大值
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 处理分类得分较高的目标框
		if (score > confidence)
		{
			// 计算在原始图像上,锚框的中心点坐标和宽高
			// 在输入图像上目标框的中心点坐标和宽高
			// 注意，RTDETR输出的坐标是比例形式，需用模型输入尺寸进行抓换；YOLO输出的坐标是像素形式
			float cx = det_output.at<float>(i, 0) * input_w;
			float cy = det_output.at<float>(i, 1) * input_h;
			float ow = det_output.at<float>(i, 2) * input_w;
			float oh = det_output.at<float>(i, 3) * input_h;
			//原始图像上目标框的左上角坐标
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			//原始图像上目标框的宽高
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);

			// 记录目标框信息
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// 数据集标签名字
	std::vector<std::string> labels =
	{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
	"20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
	"30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
	"40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
	"50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
	"60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
	"70", "71", "72", "73", "74", "75", "76", "77", "78", "79" };

	// 遍历输出的目标框
	for (int i = 0; i < boxes.size(); i++) {
		// 获取判定的类别
		int cId = classIds[i];
		std::string label = labels[cId];
		// 获取评分
		auto conf = confidences[i];
		std::ostringstream oss;
		oss << std::fixed << std::setprecision(2) << conf;
		std::string confStr = oss.str();
		std::string text = label + ":" + confStr;

		// 获取预测矩形框
		auto box = boxes[i];

		cv::rectangle(srcImage, box, cv::Scalar(0, 0, 255), 1, 8, 0);
		cv::putText(srcImage, text.c_str(), box.br(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);
	}
	std::cout << "Get " << boxes.size() << " boxes" << std::endl;

	cv::imshow("DetectResult", srcImage);
	cv::imwrite("./assets/output.jpg", srcImage);

	auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEnd - inferenceStart).count();
	std::cout << modelPath0 << "模型推理耗时: " << inference_duration << " ms" << std::endl;

	cv::waitKey(0);

	// 释放资源
	session_options.release();
	session_.release();

	return 0;
}
