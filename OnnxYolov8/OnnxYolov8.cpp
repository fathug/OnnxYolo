#include <iostream>
#include <string>
#include <fstream>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

int main()
{
	std::string modelPath0 = "./assets/yolov8n.onnx";
	std::wstring modelPath = std::wstring(modelPath0.begin(), modelPath0.end());
	std::string imagePath = "./assets/1.jpg";

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
	int numPredictions = 0;
	int numAttributes = 0;

	//for (int i = 0; i < output_nodes_num; i++) {
	for (int i = 0; i < 1; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());

		// YOLOv5[1, 25200, 85]: (85 = 4 (bbox) + 1 (confidence) + 80 (class scores))
		// YOLOv8[1, 84, 8400]: (84 = 4 (bbox) + 80 (class scores))
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		numBatch = outShapeInfo[0];
		numAttributes = outShapeInfo[1];
		numPredictions = outShapeInfo[2];
		std::cout << "output format: " << numBatch << "x" << numAttributes << "x" << numPredictions << std::endl;
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

	// 执行推理
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
		return 1;
	}

	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	// Yolov8: [batch_size, num_attributes, num_predictions]
	// num_attributes: 每个框的属性数量 (例如 84 = cx, cy, w, h, , class_score1, class_score2, ...)
	cv::Mat det_output0(numAttributes, numPredictions, CV_32F, (float*)pdata);

	// 转置，方便后续处理
	cv::Mat det_output;
	cv::transpose(det_output0, det_output);

	std::vector<cv::Rect> boxes;		// 目标框的坐标位置
	std::vector<float> confidences;		// 目标框的置信度
	std::vector<int> classIds;			// 目标框的类别得分

	// 遍历所有检测到的候选框 (det_output的每一行代表一个候选框)
	for (int i = 0; i < det_output.rows; i++) {
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
		if (score > 0.25)
		{
			// 计算在原始图像上,目标框的左上角坐标和宽高
			// 在输入图像上目标框的中心点坐标和宽高
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
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

	// NMS:非极大值抑制，去除同一目标的多余结果。
	std::vector<int> indexes;	// 存储经过 NMS 后保留的目标框在 boxes 向量中的索引
	if (!boxes.empty()) {
		cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
	}
	std::cout << "get " << indexes.size() << " boxes after NMS." << std::endl;

	std::vector<std::string> labels =
	{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
	"20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
	"30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
	"40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
	"50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
	"60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
	"70", "71", "72", "73", "74", "75", "76", "77", "78", "79" };

	// 遍历筛选出的目标框
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];		// 获取当前目标框序号
		int cid = classIds[idx];	// 获取目标框分类得分
		cv::rectangle(srcImage, boxes[idx], cv::Scalar(0, 0, 255), 1, 8, 0);
		cv::putText(srcImage, labels[cid].c_str(), boxes[idx].br(), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
	}

	cv::imshow("DetectResult", srcImage);
	cv::waitKey(0);

	// 释放资源
	session_options.release();
	session_.release();

	return 0;
}
