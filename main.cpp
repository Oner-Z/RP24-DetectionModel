#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "OpenvinoInfer.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

struct AppOptions {
    string modelXmlPath;
    string modelBinPath;
    string device;
    string videoPath;
    int detectColor; // 0/1 （与库中约定保持一致）
};

static string deriveBinPathFromXml(const string &xmlPath) {
    if (xmlPath.size() >= 4 && xmlPath.substr(xmlPath.size() - 4) == ".xml") {
        return xmlPath.substr(0, xmlPath.size() - 4) + ".bin";
    }
    return xmlPath + ".bin";
}

static AppOptions parseArgs(int argc, char **argv) {
    AppOptions opts;
    opts.modelXmlPath = "../Model/0526.xml";  // 相对于build目录的路径
    opts.modelBinPath = deriveBinPathFromXml(opts.modelXmlPath);
    opts.device = "CPU";
    opts.videoPath = "../video_test/red/v2.avi";
    opts.detectColor = 1;
    return opts;
}

static cv::Scalar colorForId(int colorId) {
    // 可视化颜色与实际观感对齐：1->Red, 0->Blue
    if (colorId == 1) return cv::Scalar(0, 0, 255);   // Red (BGR)
    if (colorId == 0) return cv::Scalar(255, 0, 0);   // Blue
    return cv::Scalar(0, 255, 255); // 其它：Yellow 作为 fallback
}

static string colorName(int colorId) {
    if (colorId == 1) return "Red";
    if (colorId == 0) return "Blue";
    return "Other";
}

static const vector<string> &classNames() {
    static vector<string> names = {
        "G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"
    };
    return names;
}

static void drawDetections(cv::Mat &frameToDrawOn, const vector<Object> &objects, double scaleX, double scaleY) {
    const auto &names = classNames();

    for (const auto &obj : objects) {
        cv::Scalar drawColor = colorForId(obj.color);

        // 四个关键点（左上起逆时针：0-1, 2-3, 4-5, 6-7）
        std::vector<cv::Point2f> pts640 = {
            cv::Point2f(obj.landmarks[0], obj.landmarks[1]),
            cv::Point2f(obj.landmarks[2], obj.landmarks[3]),
            cv::Point2f(obj.landmarks[4], obj.landmarks[5]),
            cv::Point2f(obj.landmarks[6], obj.landmarks[7])
        };
        // 映射回原尺寸
        std::vector<cv::Point2f> pts;
        pts.reserve(4);
        for (const auto &p : pts640) {
            pts.emplace_back(p.x * scaleX, p.y * scaleY);
        }

        // 连接关键点形成四边形
        for (int i = 0; i < 4; ++i) {
            cv::line(frameToDrawOn, pts[i], pts[(i + 1) % 4], drawColor, 2);
        }

        // 关键点圆点
        for (const auto &p : pts) {
            cv::circle(frameToDrawOn, p, 3, drawColor, -1);
        }

        // 文本信息：类别、颜色、置信度
        string labelText;
        if (obj.label >= 0 && obj.label < static_cast<int>(names.size())) {
            labelText = names[obj.label];
        } else {
            labelText = std::to_string(obj.label);
        }
        char probBuf[32];
        std::snprintf(probBuf, sizeof(probBuf), "%.2f", obj.prob);
        string info = labelText + " | " + colorName(obj.color) + " | conf=" + probBuf;

        // 定位文本到关键点上方
        float min_x = pts[0].x, min_y = pts[0].y;
        for (int i = 1; i < 4; ++i) {
            min_x = std::min(min_x, pts[i].x);
            min_y = std::min(min_y, pts[i].y);
        }
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::Point textOrg(std::max(0, (int)min_x), std::max(textSize.height + 2, (int)min_y - 6)); // 坐标已在原尺寸空间
        cv::rectangle(frameToDrawOn,
                      cv::Rect(textOrg.x, textOrg.y - textSize.height, textSize.width + 4, textSize.height + 4),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(frameToDrawOn, info, cv::Point(textOrg.x + 2, textOrg.y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char **argv) {
    AppOptions opts = parseArgs(argc, argv);

    cout << "模型XML: " << opts.modelXmlPath << endl;
    cout << "模型BIN: " << opts.modelBinPath << endl;
    cout << "设备: " << opts.device << endl;
    cout << "视频: " << opts.videoPath << endl;
    cout << "detect_color: " << opts.detectColor << endl;

    // 使用原有的双参数构造函数
    OpenvinoInfer inferer(opts.modelXmlPath, opts.modelBinPath, opts.device);

    // 打开视频
    cv::VideoCapture cap(opts.videoPath);
    if (!cap.isOpened()) {
        cout << "无法打开视频: " << opts.videoPath << endl;
        return 1;
    }

    const int modelW = inferer.IMAGE_WIDTH;
    const int modelH = inferer.IMAGE_HEIGHT;

    cv::Mat frame;
    cv::Mat frameForInfer;
    cv::Mat frameForShow;
    
    double fps = 0.0;
    auto lastTick = std::chrono::steady_clock::now();
    auto frameStartTime = std::chrono::steady_clock::now();

    while (true) {
        frameStartTime = std::chrono::steady_clock::now();
        
        if (!cap.read(frame)) break;

        // 生成送入模型的输入（按模型分辨率缩放到 640x640）
        cv::resize(frame, frameForInfer, cv::Size(modelW, modelH));

        // 推理
        inferer.infer(frameForInfer, opts.detectColor);

        // 可视化：将检测结果映射回原图尺寸进行绘制
        frameForShow = frame.clone();
        double scaleX = static_cast<double>(frame.cols) / modelW;
        double scaleY = static_cast<double>(frame.rows) / modelH;
        drawDetections(frameForShow, inferer.tmp_objects, scaleX, scaleY);

        // FPS 估算
        auto now = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTick).count();
        lastTick = now;
        if (ms > 0) fps = 1000.0 / ms; else fps = 0.0;

        char fpsBuf[64];
        std::snprintf(fpsBuf, sizeof(fpsBuf), "FPS: %.1f", fps);
        cv::putText(frameForShow, fpsBuf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow("RP24 Detection", frameForShow);
        
        // 帧率控制：计算已用时间，如果小于目标帧时间则等待
        auto frameEndTime = std::chrono::steady_clock::now();
        double frameElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndTime - frameStartTime).count();
        
        int waitKeyTime = 1;
        
        int key = cv::waitKey(waitKeyTime);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }else if(key == ' '){
            cv::waitKey(0);
        }else if(key == 's'){
            cv::imwrite("result.jpg", frameForShow);
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}