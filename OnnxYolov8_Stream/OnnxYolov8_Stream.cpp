#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"


// 函数：在图像上绘制FPS
void put_fps_on_frame(cv::Mat& frame, double fps) {
	std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 4);
	cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}

int main()
{
	std::string modelPath0 = "./assets/yolov8n.onnx";
	std::wstring modelPath = std::wstring(modelPath0.begin(), modelPath0.end());

	std::cout << "Model Path: " << modelPath0 << std::endl;

	// 检查模型文件是否能正常读取
	std::ifstream f(modelPath.c_str());
	if (!f.good()) {
		std::cerr << "Error: Model file does not exist or is not accessible at: " << modelPath0 << std::endl;
		return 1;
	}
	f.close();
	std::cout << "Model file found at: " << modelPath0 << std::endl;

	// ONNXRuntime初始化
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "object");
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	session_options.SetIntraOpNumThreads(1);

	auto providers = Ort::GetAvailableProviders();
	auto isCudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider");
	bool isCuda = false;
	if (isCuda && (isCudaAvailable != providers.end())) {
		OrtCUDAProviderOptions cudaOption;
		session_options.AppendExecutionProvider_CUDA(cudaOption);
		std::cout << "Current device: CUDA" << std::endl;
	}
	else {
		std::cout << "Current device: CPU" << std::endl;
	}

	Ort::Session session_(env, modelPath.c_str(), session_options);

	// 获取模型输入输出信息
	Ort::AllocatorWithDefaultOptions allocator;
	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();
	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	int input_c = 0, input_h = 0, input_w = 0;
	auto inputShapeInfo = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	input_c = inputShapeInfo[1];
	input_h = inputShapeInfo[2];
	input_w = inputShapeInfo[3];
	std::cout << "Input format: " << inputShapeInfo[0] << "x" << input_c << "x" << input_h << "x" << input_w << std::endl;

	auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
	input_node_names.push_back(input_name_ptr.get());

	int numAttributes = 0, numPredictions = 0;
	auto outShapeInfo = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	numAttributes = outShapeInfo[1];
	numPredictions = outShapeInfo[2];
	std::cout << "Output format: " << outShapeInfo[0] << "x" << numAttributes << "x" << numPredictions << std::endl;

	auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
	output_node_names.push_back(output_name_ptr.get());

	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

	// 模型预热
	std::cout << "\nModel is warming-up..." << std::endl;
	try {
		std::vector<float> dummy_input_data(static_cast<size_t>(1) * input_c * input_h * input_w, 0.0f);
		std::array<int64_t, 4> dummy_input_shape{ 1, input_c, input_h, input_w };
		auto dummy_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Ort::Value dummy_input_tensor = Ort::Value::CreateTensor<float>(dummy_allocator_info, dummy_input_data.data(), dummy_input_data.size(), dummy_input_shape.data(), dummy_input_shape.size());
		for (int i = 0; i < 3; ++i) {
			session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &dummy_input_tensor, 1, outNames.data(), outNames.size());
		}
	}
	catch (const Ort::Exception& e) {
		std::cerr << "Warm-up error: " << e.what() << std::endl;
		return 1;
	}
	std::cout << "Warm-up completed.\n" << std::endl;


	// 新增：初始化摄像头
	cv::VideoCapture cap(0); // 0 代表默认摄像头，也可以是视频文件路径
	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open camera or video stream." << std::endl;
		return 1;
	}
	std::cout << "Camera opened successfully. Press ESC to exit." << std::endl;

	cv::Mat frame;
	auto last_fps_time = std::chrono::high_resolution_clock::now();
	double fps = 0.0;

	// 新增：主处理循环
	while (true)
	{
		// 1. 读取摄像头帧
		cap.read(frame);
		if (frame.empty()) {
			std::cout << "End of video stream." << std::endl;
			break;
		}

		// 图像预处理 (与原逻辑相同, 但应用于每一帧)
		int srcWidth = frame.cols;
		int srcHeight = frame.rows;

		int maxLength = std::max(srcHeight, srcWidth);
		cv::Mat paddingImage = cv::Mat::zeros(cv::Size(maxLength, maxLength), CV_8UC3);
		frame.copyTo(paddingImage(cv::Rect(0, 0, srcWidth, srcHeight)));

		float x_factor = paddingImage.cols / static_cast<float>(input_w);
		float y_factor = paddingImage.rows / static_cast<float>(input_h);

		cv::Mat blob = cv::dnn::blobFromImage(paddingImage, 1.0 / 255.0,
			cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);

		// 创建输入张量
		std::array<int64_t, 4> input_shape_info{ 1, input_c, input_h, input_w };
		auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), blob.total(), input_shape_info.data(), input_shape_info.size());

		// 执行推理
		auto inferenceStart = std::chrono::high_resolution_clock::now();
		std::vector<Ort::Value> ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
		auto inferenceEnd = std::chrono::high_resolution_clock::now();
		auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEnd - inferenceStart).count();
		//std::cout << "Inference time: " << inference_duration << " ms" << std::endl; // 打印单帧推理时间

		// 后处理
		const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
		cv::Mat det_output0(numAttributes, numPredictions, CV_32F, (float*)pdata);
		cv::Mat det_output;
		cv::transpose(det_output0, det_output);

		std::vector<cv::Rect> boxes;
		std::vector<float> confidences;
		std::vector<int> classIds;

		for (int i = 0; i < det_output.rows; i++) {
			cv::Mat classes_scores = det_output.row(i).colRange(4, numAttributes);
			cv::Point classIdPoint;
			double score;
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
			if (score > 0.45)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
				int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
				int width = static_cast<int>(ow * x_factor);
				int height = static_cast<int>(oh * y_factor);

				boxes.push_back({ x, y, width, height });
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}

		std::vector<int> indexes;
		if (!boxes.empty()) {
			cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
		}

		std::vector<std::string> labels(80);
		for (int i = 0; i < 80; ++i) labels[i] = std::to_string(i);

		for (size_t i = 0; i < indexes.size(); i++) {
			int idx = indexes[i];
			int cid = classIds[idx];
			double conf = confidences[idx];

			std::ostringstream oss;
			oss << std::fixed << std::setprecision(2) << conf;
			std::string confStr = oss.str();

			std::string text = labels[cid] + ":" + confStr;


			cv::rectangle(frame, boxes[idx], cv::Scalar(0, 0, 255), 2);
			cv::putText(frame, text.c_str(), boxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
		}

		// 新增：计算并显示FPS
		auto current_time = std::chrono::high_resolution_clock::now();
		fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_time).count();
		last_fps_time = current_time;
		put_fps_on_frame(frame, fps);

		// 新增：实时显示结果
		cv::imshow("YOLO Real-Time Detection", frame);

		// 新增：退出循环
		if (cv::waitKey(1) == 27) { // 27是ESC键的ASCII码
			break;
		}
	}

	// 释放资源
	cap.release();
	cv::destroyAllWindows();
	session_options.release();
	session_.release();

	return 0;
}