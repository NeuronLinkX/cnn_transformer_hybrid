// LibTorch의 TorchScript 모듈 로딩 API를 포함한다.
#include <torch/script.h>
// OpenCV 이미지 로딩 및 전처리 API를 포함한다.
#include <opencv2/opencv.hpp>
// 콘솔 출력을 위해 iostream을 포함한다.
#include <iostream>
// 출력 경로 처리를 위해 filesystem을 포함한다.
#include <filesystem>
// manifest와 배치 출력 파일 처리를 위해 fstream을 포함한다.
#include <fstream>
// 고정된 CIFAR-10 클래스 이름 테이블을 위해 array를 포함한다.
#include <array>
// 형식화된 주석 문자열 생성을 위해 sstream을 포함한다.
#include <sstream>
// 일반 컨테이너 사용을 위해 vector를 포함한다.
#include <vector>
// 행 파싱과 경로 문자열 처리를 위해 string을 포함한다.
#include <string>
// 런타임 오류 보고를 위해 stdexcept를 포함한다.
#include <stdexcept>
namespace fs = std::filesystem;

// 정적 CIFAR-10 클래스 라벨 테이블을 정의한다.
static const std::array<const char*, 10> CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

struct PredictionResult {
    int pred_idx;
    float confidence;
};

struct ManifestEntry {
    int order;
    int dataset_index;
    int target_idx;
    std::string target_name;
    fs::path image_path;
    std::string file_name;
};

static bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::vector<std::string> split_tsv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    std::istringstream stream(line);
    while (std::getline(stream, current, '\t')) {
        fields.push_back(current);
    }
    if (ends_with(line, "\t")) {
        fields.emplace_back();
    }
    return fields;
}

// 프로젝트 로컬 outputs 디렉터리 안에 기본 출력 이미지 경로를 만든다.
static fs::path make_output_path(const fs::path& input_path) {
    // 주석이 그려진 결과는 고정된 outputs 디렉터리에 저장한다.
    const fs::path output_dir = "outputs";
    // 출력 디렉터리가 없으면 생성한다.
    fs::create_directories(output_dir);
    // 입력 파일 stem을 유지하고 저장 파일명에 접미사를 붙인다.
    return output_dir / (input_path.stem().string() + "_prediction.jpg");
}

// 원본 이미지에 예측 클래스와 신뢰도를 그려 저장한다.
static fs::path annotate_and_save(
    const cv::Mat& image_bgr,
    const std::string& class_name,
    float confidence,
    const fs::path& output_path) {
    // 작은 CIFAR 이미지를 확대해 저장 결과를 눈으로 확인하기 쉽게 만든다.
    cv::Mat annotated;
    if (image_bgr.cols < 256 || image_bgr.rows < 256) {
        const double scale_x = 320.0 / static_cast<double>(image_bgr.cols);
        const double scale_y = 320.0 / static_cast<double>(image_bgr.rows);
        const double scale = std::max(scale_x, scale_y);
        cv::resize(
            image_bgr,
            annotated,
            cv::Size(),
            scale,
            scale,
            cv::INTER_NEAREST);
    } else {
        // 더 큰 입력은 원래 해상도를 유지한다.
        annotated = image_bgr.clone();
    }

    // 예측 클래스를 사람이 읽기 쉬운 텍스트로 만든다.
    std::ostringstream label_stream;
    label_stream << "prediction: " << class_name;
    const std::string label_text = label_stream.str();

    // 신뢰도 점수를 사람이 읽기 쉬운 텍스트로 만든다.
    std::ostringstream conf_stream;
    conf_stream.precision(4);
    conf_stream << std::fixed << "confidence: " << confidence;
    const std::string conf_text = conf_stream.str();

    // 추가 의존성 없이 OpenCV가 바로 렌더링할 수 있는 폰트를 선택한다.
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    // 작은 CIFAR 이미지에서도 읽히도록 원본 너비 기준으로 글자 크기를 조절한다.
    const double font_scale = std::max(0.45, annotated.cols / 800.0);
    // 가독성을 위해 두께를 약간 키운다.
    const int thickness = std::max(1, annotated.cols / 300);
    // 예측 오버레이 텍스트는 녹색으로 표시한다.
    const cv::Scalar text_color(0, 255, 0);
    // 대비를 위해 어두운 채움 배경을 사용한다.
    const cv::Scalar bg_color(0, 0, 0);
    // 이미지 경계에서 약간의 여백을 둔다.
    const int margin = 10;
    // 두 줄 텍스트 사이에 세로 간격을 둔다.
    const int line_gap = 8;
    int baseline = 0;

    // 예측 라벨의 크기를 측정한다.
    const cv::Size label_size = cv::getTextSize(label_text, font_face, font_scale, thickness, &baseline);
    // 신뢰도 라벨의 크기를 측정한다.
    const cv::Size conf_size = cv::getTextSize(conf_text, font_face, font_scale, thickness, &baseline);

    // 공통 오버레이 박스 너비를 계산한다.
    const int box_width = std::max(label_size.width, conf_size.width) + 2 * margin;
    // 두 줄을 포함한 오버레이 박스 높이를 계산한다.
    const int box_height = label_size.height + conf_size.height + 3 * margin + line_gap;

    // 텍스트 뒤에 채워진 사각형을 그려 가독성을 유지한다.
    cv::rectangle(annotated, cv::Rect(margin, margin, box_width, box_height), bg_color, cv::FILLED);

    // 예측 클래스 텍스트를 그린다.
    cv::putText(
        annotated,
        label_text,
        cv::Point(2 * margin, 2 * margin + label_size.height),
        font_face,
        font_scale,
        text_color,
        thickness,
        cv::LINE_AA);

    // 예측 라인 아래에 신뢰도 텍스트를 그린다.
    cv::putText(
        annotated,
        conf_text,
        cv::Point(2 * margin, 3 * margin + label_size.height + line_gap + conf_size.height),
        font_face,
        font_scale,
        text_color,
        thickness,
        cv::LINE_AA);

    // 필요하면 출력 경로의 상위 디렉터리를 생성한다.
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    // 주석이 그려진 결과 이미지를 저장한다.
    if (!cv::imwrite(output_path.string(), annotated)) {
        throw std::runtime_error("failed to save annotated output image");
    }

    // 로그 출력을 위해 최종 저장 경로를 반환한다.
    return output_path;
}

// Python의 Normalize와 resize 동작을 C++에서 동일하게 맞춘다.
static torch::Tensor preprocess(const cv::Mat& bgr) {
    // 빈 입력 이미지는 즉시 거부한다.
    if (bgr.empty()) throw std::runtime_error("empty image");

    // RGB 변환, 리사이즈, float 변환용 중간 Mat을 준비한다.
    cv::Mat rgb, resized, f32;
    // OpenCV 기본 BGR 이미지를 RGB 순서로 바꾼다.
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    // 학습 때 사용한 것과 같은 224x224 크기로 리사이즈한다.
    cv::resize(rgb, resized, cv::Size(224, 224));
    // float32로 변환하고 [0, 1] 범위로 스케일링한다.
    resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    // HWC float 버퍼를 텐서로 감싼 뒤 clone하여 메모리 소유권을 텐서가 갖게 한다.
    auto tensor = torch::from_blob(f32.data, {1, 224, 224, 3}, torch::kFloat32).clone();
    // PyTorch 입력 형식에 맞게 NHWC를 NCHW로 바꾼다.
    tensor = tensor.permute({0, 3, 1, 2});

    // 정규화 평균 텐서를 만든다.
    auto mean = torch::tensor({0.4914f, 0.4822f, 0.4465f}).view({1, 3, 1, 1});
    // 정규화 표준편차 텐서를 만든다.
    auto std = torch::tensor({0.2470f, 0.2435f, 0.2616f}).view({1, 3, 1, 1});
    // Python 파이프라인과 동일한 채널별 정규화를 적용한다.
    return (tensor - mean) / std;
}

static PredictionResult predict_image(torch::jit::script::Module& model, const fs::path& image_path) {
    cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    auto input = preprocess(img);
    auto output = model.forward({input}).toTensor();
    auto prob = torch::softmax(output, 1);
    const int pred = prob.argmax(1).item<int>();
    const float conf = prob[0][pred].item<float>();
    return {pred, conf};
}

static std::vector<ManifestEntry> load_manifest(const fs::path& manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + manifest_path.string());
    }

    std::vector<ManifestEntry> entries;
    std::string line;
    bool first_line = true;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        auto fields = split_tsv_line(line);
        if (first_line) {
            first_line = false;
            if (!fields.empty() && fields[0] == "order") {
                continue;
            }
        }

        if (fields.size() < 4) {
            throw std::runtime_error("manifest row has fewer than 4 tab-separated fields");
        }

        ManifestEntry entry{};
        if (fields.size() >= 5) {
            entry.order = std::stoi(fields[0]);
            entry.dataset_index = std::stoi(fields[1]);
            entry.target_idx = std::stoi(fields[2]);
            entry.target_name = fields[3];
            entry.file_name = fields[4];
        } else {
            entry.order = std::stoi(fields[0]);
            entry.dataset_index = std::stoi(fields[1]);
            entry.target_name = fields[2];
            entry.target_idx = -1;
            entry.file_name = fields[3];
        }
        entry.image_path = manifest_path.parent_path() / entry.file_name;
        entries.push_back(entry);
    }

    if (entries.empty()) {
        throw std::runtime_error("manifest does not contain any samples");
    }
    return entries;
}

static fs::path run_manifest_inference(
    torch::jit::script::Module& model,
    const fs::path& manifest_path,
    const fs::path& output_path) {
    const auto entries = load_manifest(manifest_path);
    if (output_path.has_parent_path()) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("failed to open batch output path: " + output_path.string());
    }

    std::array<std::array<int, 10>, 10> confusion{};
    int total = 0;
    int correct = 0;

    output << "order\tdataset_index\ttarget_idx\ttarget_name\tpred_idx\tpred_name\tconfidence\tcorrect\timage_path\n";
    for (const auto& entry : entries) {
        const auto prediction = predict_image(model, entry.image_path);
        bool is_correct = false;
        if (entry.target_idx >= 0 && entry.target_idx < static_cast<int>(CIFAR10_CLASSES.size())) {
            is_correct = prediction.pred_idx == entry.target_idx;
            confusion[entry.target_idx][prediction.pred_idx] += 1;
        } else if (prediction.pred_idx >= 0 && prediction.pred_idx < static_cast<int>(CIFAR10_CLASSES.size())) {
            is_correct = std::string(CIFAR10_CLASSES[prediction.pred_idx]) == entry.target_name;
        }

        total += 1;
        if (is_correct) {
            correct += 1;
        }

        output << entry.order << '\t'
               << entry.dataset_index << '\t'
               << entry.target_idx << '\t'
               << entry.target_name << '\t'
               << prediction.pred_idx << '\t'
               << CIFAR10_CLASSES[prediction.pred_idx] << '\t'
               << prediction.confidence << '\t'
               << (is_correct ? 1 : 0) << '\t'
               << entry.image_path.string() << '\n';
    }

    const double accuracy = total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
    std::cout << "batch_predictions: " << output_path.string() << std::endl;
    std::cout << "samples=" << total << " correct=" << correct << " accuracy=" << accuracy << std::endl;
    std::cout << "[CONFUSION_MATRIX]" << std::endl;
    for (const auto& row : confusion) {
        for (std::size_t col = 0; col < row.size(); ++col) {
            if (col > 0) {
                std::cout << ' ';
            }
            std::cout << row[col];
        }
        std::cout << std::endl;
    }
    return output_path;
}

int main(int argc, const char* argv[]) {
    // 먼저 명령행 인자 개수가 올바른지 검사한다.
    if (argc < 3 || argc > 4) {
        // 기대하는 사용법 문자열을 출력한다.
        std::cerr << "usage: cifar10_infer <model.pt> <image_path|manifest.tsv> [output_path]\n";
        // 잘못된 입력일 때는 0이 아닌 종료 코드를 반환한다.
        return 1;
    }

    try {
        // 추론에는 gradient가 필요 없으므로 autograd를 비활성화한다.
        torch::NoGradGuard no_grad;
        // 디스크에서 TorchScript 모듈을 불러온다.
        torch::jit::script::Module model = torch::jit::load(argv[1]);
        // 모듈을 평가 모드로 전환한다.
        model.eval();

        const fs::path input_path(argv[2]);
        if (input_path.extension() == ".tsv") {
            const fs::path output_path = (argc == 4)
                ? fs::path(argv[3])
                : (fs::path("outputs") / "batch_predictions.tsv");
            run_manifest_inference(model, input_path, output_path);
        } else {
            // 디스크에서 BGR 순서의 입력 이미지를 불러온다.
            cv::Mat img = cv::imread(input_path.string(), cv::IMREAD_COLOR);
            const auto prediction = predict_image(model, input_path);
            // 세 번째 인자 또는 기본 outputs 디렉터리로부터 출력 이미지 경로를 결정한다.
            const fs::path output_path = (argc == 4) ? fs::path(argv[3]) : make_output_path(input_path);
            // GUI 없이도 확인할 수 있도록 주석이 그려진 결과 이미지를 저장한다.
            const fs::path saved_path = annotate_and_save(
                img,
                CIFAR10_CLASSES[prediction.pred_idx],
                prediction.confidence,
                output_path);

            // 예측 클래스 라벨과 신뢰도 점수를 출력한다.
            std::cout << "prediction: " << CIFAR10_CLASSES[prediction.pred_idx]
                      << " confidence=" << prediction.confidence << std::endl;
            // 저장된 출력 경로를 출력해 결과 파일을 열 수 있게 한다.
            std::cout << "saved_result: " << saved_path.string() << std::endl;
        }
    } catch (const c10::Error& e) {
        // LibTorch 전용 예외는 별도 메시지로 보고한다.
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        // LibTorch 실패 전용 종료 코드를 반환한다.
        return 2;
    } catch (const std::exception& e) {
        // 일반 런타임 예외는 공통 메시지로 보고한다.
        std::cerr << "error: " << e.what() << std::endl;
        // 일반 실패 종료 코드를 반환한다.
        return 3;
    }
    // 추론이 정상 종료되면 성공 코드를 반환한다.
    return 0;
}
