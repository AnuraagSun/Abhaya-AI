// main.cpp
#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QGroupBox>
#include <QGridLayout>
#include <QProgressBar>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QWaitCondition>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QPen>
#include <QFont>
#include <QPalette>
#include <QColor>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QStandardPaths>
#include <QDebug>
#include <QElapsedTimer>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Placeholder for MediaPipe C++ API
// MediaPipe's C++ API is complex and requires Bazel build system or pre-built libraries.
// For this example, we'll use a dummy function to simulate landmark detection.
// In a real project, you would link against MediaPipe's C++ libraries.
#include "mediapipe_api.h" // This is a placeholder header

#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <ctime>
#include <iomanip>

// --- Configuration and Constants ---

// Default configuration structure
struct Config {
    int camera_index = 0;
    int frame_skip = 2;
    double ear_threshold = 0.25;
    double mar_threshold = 0.50;
    int consecutive_frames = 3;
    int prediction_window_size = 10;
    int alert_duration_ms = 3000;
    int calibration_duration_s = 5;
    std::string model_path = "drowsiness_model.pkl"; // Note: ML model loading is complex in C++
    std::string scaler_path = "scaler.pkl";
    std::string log_file_path = "drowsiness_log_cpp.txt";
    std::string alert_sound_path = "alert_sound.wav"; // Note: Sound playing is complex in C++
    int gui_refresh_rate_ms = 30;
    double head_pitch_threshold = 15.0;
    double head_yaw_threshold = 20.0;
    double blinking_threshold = 0.18;
    double yawning_threshold = 0.65;
    double looking_away_threshold = 0.25;
    bool enable_ml_prediction = true;
    bool enable_visual_feedback = true;
    bool enable_audio_alerts = true;
    bool enable_logging = true;
    bool enable_calibration = true;
    double ml_confidence_threshold = 0.70;
    double perclos_threshold = 0.30;
    int blink_duration_ms = 150;
    int yawn_duration_ms = 1500;
    double detection_timeout_s = 10.0;
    bool face_mesh_refinement = true;
    double min_detection_confidence = 0.5;
    double min_tracking_confidence = 0.5;
    int gui_width = 1400;
    int gui_height = 900;
    std::string font_family = "Arial";
    int font_size = 12;
    std::string theme = "dark";
    // Enhanced Config
    bool enable_environmental_simulation = true;
    double environmental_co2_baseline = 400.0;
    double environmental_temp_baseline = 22.0;
    double environmental_humidity_baseline = 50.0;
    double environmental_co2_rate_increase = 5.0;
    double environmental_temp_rate_increase = 0.1;
    double environmental_humidity_rate_increase = 1.0;
    bool enable_behavioral_modeling = true;
    bool enable_trip_metadata = true;
    int trip_duration_threshold = 120;
    std::map<std::string, double> time_of_day_factor = {{"early_morning", 1.2}, {"afternoon", 1.0}, {"evening", 1.1}, {"night", 1.5}};
    bool enable_driver_profile = true;
    std::string profile_file_path = "driver_profile_cpp.json";
    bool enable_prediction_history = true;
    int history_buffer_size = 50;
    bool enable_data_export = true;
    std::string export_data_path = "detection_data_cpp.json";
    bool enable_performance_monitoring = true;
    int performance_log_interval = 10;
    bool enable_advanced_visualization = true;
    bool enable_microsleep_detection = true;
    double microsleep_ear_threshold = 0.15;
    int microsleep_duration_ms = 500;
    bool enable_adaptive_thresholds = true;
    double adaptive_learning_rate = 0.01;
    int adaptive_window_size = 100;
    bool enable_fatigue_score = true;
    double fatigue_score_base = 50.0;
    double fatigue_score_ear_weight = -10.0;
    double fatigue_score_mar_weight = 5.0;
    double fatigue_score_pitch_weight = 3.0;
    double fatigue_score_blink_weight = 2.0;
    double fatigue_score_yawn_weight = 8.0;
    double fatigue_score_max = 100.0;
    double fatigue_score_min = 0.0;
    int fatigue_prediction_horizon = 15;
    bool enable_emergency_protocol = true;
    std::vector<std::string> emergency_sms_recipients = {};
    std::vector<std::string> emergency_email_recipients = {};
};

// --- Logging ---

void log_message(const std::string& level, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    std::ofstream log_file(Config().log_file_path, std::ios::app);
    if (log_file.is_open()) {
        log_file << ss.str() << " - " << level << " - " << message << std::endl;
        log_file.close();
    }
    std::cout << ss.str() << " - " << level << " - " << message << std::endl;
}

// --- Core Detection Logic Classes ---

class FeatureExtractor {
private:
    Config config;
    std::vector<int> LEFT_EYE_INDICES = {33, 160, 158, 133, 153, 144};
    std::vector<int> RIGHT_EYE_INDICES = {263, 388, 386, 362, 384, 373};
    std::vector<int> MOUTH_INDICES = {61, 39, 0, 269, 291, 178, 81, 40, 82, 13, 312, 214, 177, 178, 14, 87, 178, 88, 95, 181, 84, 102, 215, 10, 199, 9, 176, 89, 151, 57, 43, 107, 95, 181, 84, 102, 215, 10, 199, 9, 176, 89, 151, 57, 43, 107};
    int NOSE_TIP_INDEX = 1;
    int CHIN_INDEX = 152;
    int LEFT_EAR_INDEX = 127;
    int RIGHT_EAR_INDEX = 356;
    double EAR_THRESH;
    double MAR_THRESH;
    double HEAD_PITCH_THRESH;
    double HEAD_YAW_THRESH;
    int BLINK_DURATION_MS;
    int YAWN_DURATION_MS;
    int PERCLOS_WINDOW_SIZE;
    int MICROSLEEP_DURATION_MS;
    double MICROSLEEP_EAR_THRESH;

    double eye_closed_start_time = 0;
    double mouth_open_start_time = 0;
    double microsleep_start_time = 0;
    std::vector<double> eye_aspect_ratio_history;
    std::vector<double> blink_rate_history;
    std::vector<double> yawn_rate_history;
    std::vector<double> perclos_history;
    std::vector<bool> looking_away_history;
    std::vector<bool> nodding_history;
    std::vector<std::pair<double, double>> head_pose_history;
    std::vector<double> ear_buffer;
    std::vector<double> mar_buffer;
    std::vector<double> microsleep_events;
    std::vector<double> blinking_events;
    std::vector<double> yawning_events;

public:
    FeatureExtractor(const Config& c) : config(c) {
        EAR_THRESH = config.blinking_threshold;
        MAR_THRESH = config.yawning_threshold;
        HEAD_PITCH_THRESH = config.head_pitch_threshold;
        HEAD_YAW_THRESH = config.head_yaw_threshold;
        BLINK_DURATION_MS = config.blink_duration_ms;
        YAWN_DURATION_MS = config.yawn_duration_ms;
        PERCLOS_WINDOW_SIZE = config.prediction_window_size;
        MICROSLEEP_DURATION_MS = config.microsleep_duration_ms;
        MICROSLEEP_EAR_THRESH = config.microsleep_ear_threshold;
    }

    double calculate_ear(const std::vector<cv::Point2f>& eye_landmarks) {
        double A = cv::norm(eye_landmarks[1] - eye_landmarks[5]);
        double B = cv::norm(eye_landmarks[2] - eye_landmarks[4]);
        double C = cv::norm(eye_landmarks[0] - eye_landmarks[3]);
        return (A + B) / (2.0 * C);
    }

    double calculate_mar(const std::vector<cv::Point2f>& mouth_landmarks) {
        double A = cv::norm(mouth_landmarks[3] - mouth_landmarks[7]); // Vertical
        double B = cv::norm(mouth_landmarks[0] - mouth_landmarks[4]); // Horizontal
        return A / B;
    }

    std::tuple<double, double, double> estimate_head_pose(const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_shape) {
        // 3D model points of face landmarks (simplified)
        std::vector<cv::Point3f> model_points = {
            cv::Point3f(0.0f, 0.0f, 0.0f),             // Nose tip
            cv::Point3f(0.0f, -300.0f, -67.5f),        // Chin
            cv::Point3f(68.0f, -200.0f, -110.0f),      // Left eye left corner
            cv::Point3f(-68.0f, -200.0f, -110.0f),     // Right eye right corne
            cv::Point3f(50.0f, 150.0f, -100.0f),       // Left Mouth corner
            cv::Point3f(-50.0f, 150.0f, -100.0f)       // Right mouth corner
        };

        std::vector<cv::Point2f> image_points = {
            landmarks[NOSE_TIP_INDEX],
            landmarks[CHIN_INDEX],
            landmarks[LEFT_EAR_INDEX],
            landmarks[RIGHT_EAR_INDEX],
            landmarks[61], // Left mouth corner
            landmarks[291] // Right mouth corner
        };

        double focal_length = static_cast<double>(frame_shape.width);
        cv::Point2f center = cv::Point2f(frame_shape.width / 2.0, frame_shape.height / 2.0);
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);

        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

        cv::Mat rotation_vector, translation_vector;
        bool success = cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

        if (success) {
            cv::Mat rotation_matrix;
            cv::Rodrigues(rotation_vector, rotation_matrix);

            double pitch = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
            double yaw = atan2(-rotation_matrix.at<double>(2, 0), sqrt(rotation_matrix.at<double>(2, 1)*rotation_matrix.at<double>(2, 1) + rotation_matrix.at<double>(2, 2)*rotation_matrix.at<double>(2, 2)));
            double roll = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));

            return std::make_tuple(cv::degrees(pitch), cv::degrees(yaw), cv::degrees(roll));
        }
        return std::make_tuple(0.0, 0.0, 0.0);
    }

    void detect_microsleep(double ear, double current_time) {
        bool is_eye_very_closed = ear < MICROSLEEP_EAR_THRESH;
        if (is_eye_very_closed && microsleep_start_time == 0) {
            microsleep_start_time = current_time;
        } else if (!is_eye_very_closed && microsleep_start_time != 0) {
            double duration_ms = (current_time - microsleep_start_time) * 1000;
            if (duration_ms <= MICROSLEEP_DURATION_MS) {
                microsleep_events.push_back(current_time);
                log_message("WARNING", "MICROSLEEP DETECTED, duration: " + std::to_string(duration_ms) + "ms");
            }
            microsleep_start_time = 0;
        }
        double cutoff_time = current_time - 1.0;
        microsleep_events.erase(std::remove_if(microsleep_events.begin(), microsleep_events.end(),
            [cutoff_time](double t) { return t <= cutoff_time; }), microsleep_events.end());
    }

    void detect_blinks(double ear, double current_time) {
        bool is_eye_closed = ear < EAR_THRESH;
        if (is_eye_closed && eye_closed_start_time == 0) {
            eye_closed_start_time = current_time;
        } else if (!is_eye_closed && eye_closed_start_time != 0) {
            double duration_ms = (current_time - eye_closed_start_time) * 1000;
            if (duration_ms >= BLINK_DURATION_MS) {
                blinking_events.push_back(current_time);
                log_message("DEBUG", "Blink detected, duration: " + std::to_string(duration_ms) + "ms");
            }
            eye_closed_start_time = 0;
        }
        double cutoff_time = current_time - 1.0;
        blinking_events.erase(std::remove_if(blinking_events.begin(), blinking_events.end(),
            [cutoff_time](double t) { return t <= cutoff_time; }), blinking_events.end());
    }

    void detect_yawns(double mar, double current_time) {
        bool is_mouth_open = mar > MAR_THRESH;
        if (is_mouth_open && mouth_open_start_time == 0) {
            mouth_open_start_time = current_time;
        } else if (!is_mouth_open && mouth_open_start_time != 0) {
            double duration_ms = (current_time - mouth_open_start_time) * 1000;
            if (duration_ms >= YAWN_DURATION_MS) {
                yawning_events.push_back(current_time);
                log_message("DEBUG", "Yawn detected, duration: " + std::to_string(duration_ms) + "ms");
            }
            mouth_open_start_time = 0;
        }
        double cutoff_time = current_time - 2.0;
        yawning_events.erase(std::remove_if(yawning_events.begin(), yawning_events.end(),
            [cutoff_time](double t) { return t <= cutoff_time; }), yawning_events.end());
    }

    double calculate_perclos() {
        if (eye_aspect_ratio_history.size() < 10) return 0.0;
        int threshold_count = std::count_if(eye_aspect_ratio_history.begin(), eye_aspect_ratio_history.end(),
            [this](double ear) { return ear < EAR_THRESH; });
        return static_cast<double>(threshold_count) / eye_aspect_ratio_history.size();
    }

    bool is_looking_away(const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_shape) {
        double h = frame_shape.height;
        double w = frame_shape.width;
        double left_eye_x = 0, right_eye_x = 0;
        for (int idx : LEFT_EYE_INDICES) left_eye_x += landmarks[idx].x;
        for (int idx : RIGHT_EYE_INDICES) right_eye_x += landmarks[idx].x;
        left_eye_x /= LEFT_EYE_INDICES.size();
        right_eye_x /= RIGHT_EYE_INDICES.size();
        double nose_x = landmarks[NOSE_TIP_INDEX].x;

        double deviation = std::max(std::abs(left_eye_x - nose_x), std::abs(right_eye_x - nose_x));
        return deviation > w * config.looking_away_threshold;
    }

    bool detect_nodding() {
        if (head_pose_history.size() < 3) return false;
        std::vector<double> recent_pitches;
        for (int i = std::max(0, static_cast<int>(head_pose_history.size()) - 3); i < head_pose_history.size(); ++i) {
            recent_pitches.push_back(head_pose_history[i].first);
        }
        if (recent_pitches.size() >= 3) {
            if ((recent_pitches[0] < recent_pitches[1] && recent_pitches[1] > recent_pitches[2]) ||
                (recent_pitches[0] > recent_pitches[1] && recent_pitches[1] < recent_pitches[2])) {
                if (std::abs(recent_pitches[1]) > HEAD_PITCH_THRESH) {
                    return true;
                }
            }
        }
        return false;
    }

    std::map<std::string, double> extract_features(const std::vector<cv::Point2f>& landmarks, const cv::Size& frame_shape, double current_time) {
        std::map<std::string, double> features;

        // EAR
        std::vector<cv::Point2f> left_eye_points, right_eye_points;
        for (int idx : LEFT_EYE_INDICES) left_eye_points.push_back(landmarks[idx]);
        for (int idx : RIGHT_EYE_INDICES) right_eye_points.push_back(landmarks[idx]);
        double left_ear = calculate_ear(left_eye_points);
        double right_ear = calculate_ear(right_eye_points);
        double ear = (left_ear + right_ear) / 2.0;
        eye_aspect_ratio_history.push_back(ear);
        if (eye_aspect_ratio_history.size() > PERCLOS_WINDOW_SIZE) eye_aspect_ratio_history.erase(eye_aspect_ratio_history.begin());
        features["ear"] = ear;
        ear_buffer.push_back(ear);
        if (ear_buffer.size() > config.adaptive_window_size) ear_buffer.erase(ear_buffer.begin());

        // MAR
        std::vector<cv::Point2f> mouth_points;
        for (int idx : MOUTH_INDICES) mouth_points.push_back(landmarks[idx]);
        double mar = calculate_mar(mouth_points);
        features["mar"] = mar;
        mar_buffer.push_back(mar);
        if (mar_buffer.size() > config.adaptive_window_size) mar_buffer.erase(mar_buffer.begin());

        // Head Pose
        auto [pitch, yaw, roll] = estimate_head_pose(landmarks, frame_shape);
        features["head_pitch"] = pitch;
        features["head_yaw"] = yaw;
        features["head_roll"] = roll;
        head_pose_history.push_back(std::make_pair(pitch, yaw));
        if (head_pose_history.size() > 5) head_pose_history.erase(head_pose_history.begin());

        // Detect blinks, yawns, microsleeps
        detect_microsleep(ear, current_time);
        detect_blinks(ear, current_time);
        detect_yawns(mar, current_time);

        // Calculate derived features
        features["blink_rate"] = static_cast<double>(blinking_events.size()) / 1.0;
        features["yawn_rate"] = static_cast<double>(yawning_events.size()) / 2.0;
        features["microsleep_rate"] = static_cast<double>(microsleep_events.size()) / 1.0;
        features["perclos"] = calculate_perclos();
        features["is_looking_away"] = is_looking_away(landmarks, frame_shape) ? 1.0 : 0.0;
        features["is_nodding"] = detect_nodding() ? 1.0 : 0.0;

        // Store history for dashboard
        perclos_history.push_back(features["perclos"]);
        if (perclos_history.size() > config.history_buffer_size) perclos_history.erase(perclos_history.begin());
        looking_away_history.push_back(features["is_looking_away"] > 0.5);
        if (looking_away_history.size() > config.history_buffer_size) looking_away_history.erase(looking_away_history.begin());
        nodding_history.push_back(features["is_nodding"] > 0.5);
        if (nodding_history.size() > config.history_buffer_size) nodding_history.erase(nodding_history.begin());
        blink_rate_history.push_back(features["blink_rate"]);
        if (blink_rate_history.size() > config.history_buffer_size) blink_rate_history.erase(blink_rate_history.begin());
        yawn_rate_history.push_back(features["yawn_rate"]);
        if (yawn_rate_history.size() > config.history_buffer_size) yawn_rate_history.erase(yawn_rate_history.begin());

        return features;
    }

    std::pair<double, double> get_adaptive_thresholds() {
        if (!config.enable_adaptive_thresholds) {
            return std::make_pair(config.ear_threshold, config.mar_threshold);
        }
        double new_ear_thresh = config.ear_threshold;
        double new_mar_thresh = config.mar_threshold;
        if (!ear_buffer.empty()) {
            double avg_ear = std::accumulate(ear_buffer.begin(), ear_buffer.end(), 0.0) / ear_buffer.size();
            double var_ear = 0.0;
            for (double val : ear_buffer) var_ear += (val - avg_ear) * (val - avg_ear);
            var_ear /= ear_buffer.size();
            double std_ear = std::sqrt(var_ear);
            new_ear_thresh = std::max(0.15, std::min(0.35, avg_ear - (std_ear * 0.5)));
        }
        if (!mar_buffer.empty()) {
            double avg_mar = std::accumulate(mar_buffer.begin(), mar_buffer.end(), 0.0) / mar_buffer.size();
            double var_mar = 0.0;
            for (double val : mar_buffer) var_mar += (val - avg_mar) * (val - avg_mar);
            var_mar /= mar_buffer.size();
            double std_mar = std::sqrt(var_mar);
            new_mar_thresh = std::max(0.40, std::min(0.80, avg_mar + (std_mar * 0.5)));
        }
        return std::make_pair(new_ear_thresh, new_mar_thresh);
    }
};

// PredictionEngine and other classes would follow a similar pattern,
// but implementing ML models (like RandomForest) in C++ without external libraries
// like Dlib or Shogun is non-trivial. For this example, we'll focus on the rule-based core.

class PredictionEngine {
private:
    Config config;
    std::vector<std::map<std::string, double>> feature_buffer;
    std::vector<std::map<std::string, double>> prediction_history;
    double fatigue_score;

public:
    PredictionEngine(const Config& c) : config(c), fatigue_score(config.fatigue_score_base) {}

    std::pair<int, double> predict(const std::map<std::string, double>& features) {
        // Rule-based prediction
        int drowsy_signs = 0;
        double ear = features.at("ear");
        double mar = features.at("mar");
        double head_pitch = std::abs(features.at("head_pitch"));
        bool is_looking_away = features.at("is_looking_away") > 0.5;
        bool is_nodding = features.at("is_nodding") > 0.5;
        double perclos = features.at("perclos");
        double microsleep_rate = features.at("microsleep_rate");

        if (ear < config.ear_threshold) drowsy_signs++;
        if (mar > config.mar_threshold) drowsy_signs++;
        if (head_pitch > config.head_pitch_threshold) drowsy_signs++;
        if (is_looking_away) drowsy_signs++;
        if (is_nodding) drowsy_signs++;
        if (perclos > config.perclos_threshold) drowsy_signs++;
        if (microsleep_rate > 0) drowsy_signs += 2;

        int prediction = (drowsy_signs >= 3) ? 1 : 0; // 1: Drowsy, 0: Alert
        double confidence = 0.5; // Placeholder for rule-based confidence
        return std::make_pair(prediction, confidence);
    }

    void update_buffer(const std::map<std::string, double>& features) {
        feature_buffer.push_back(features);
        if (feature_buffer.size() > config.prediction_window_size) {
            feature_buffer.erase(feature_buffer.begin());
        }
    }

    double calculate_fatigue_score(const std::map<std::string, double>& features) {
        double score = config.fatigue_score_base;
        score += features.at("ear") * config.fatigue_score_ear_weight;
        score += features.at("mar") * config.fatigue_score_mar_weight;
        score += std::abs(features.at("head_pitch")) * config.fatigue_score_pitch_weight;
        score += features.at("blink_rate") * config.fatigue_score_blink_weight;
        score += features.at("yawn_rate") * config.fatigue_score_yawn_weight;
        score += features.at("microsleep_rate") * config.fatigue_score_yawn_weight * 2;
        fatigue_score = std::max(config.fatigue_score_min, std::min(config.fatigue_score_max, score));
        return fatigue_score;
    }

    double predict_fatigue_horizon(const std::map<std::string, double>& features) {
        double current_score = calculate_fatigue_score(features);
        // Simplified prediction: assume score increases linearly based on current rate
        // A real implementation would use a time-series model
        double predicted_score = current_score + (config.fatigue_prediction_horizon * 0.1); // Placeholder rate
        return std::max(config.fatigue_score_min, std::min(config.fatigue_score_max, predicted_score));
    }

    void update_prediction_history(int prediction, double confidence, const std::map<std::string, double>& features) {
        if (config.enable_prediction_history) {
            std::map<std::string, double> entry = features;
            entry["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            entry["prediction"] = static_cast<double>(prediction);
            entry["confidence"] = confidence;
            entry["fatigue_score"] = calculate_fatigue_score(features);
            prediction_history.push_back(entry);
            if (prediction_history.size() > config.history_buffer_size) {
                prediction_history.erase(prediction_history.begin());
            }
        }
    }

    double get_fatigue_score() const { return fatigue_score; }
    const std::vector<std::map<std::string, double>>& get_prediction_history() const { return prediction_history; }
};

class AlertSystem {
private:
    Config config;
    bool alert_active = false;
    double alert_start_time = 0;
    bool emergency_triggered = false;
    double emergency_start_time = 0;

public:
    AlertSystem(const Config& c) : config(c) {}

    void trigger_alert(const std::string& alert_level = "medium") {
        if (alert_active) return;

        alert_active = true;
        alert_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;

        if (alert_level == "critical") {
            log_message("CRITICAL", "CRITICAL DROWSINESS ALERT TRIGGERED!");
            if (config.enable_emergency_protocol) {
                trigger_emergency_protocol();
            }
        } else {
            log_message("WARNING", "DROWSINESS ALERT (" + alert_level + ") TRIGGERED!");
        }

        if (config.enable_audio_alerts) {
            // Placeholder for audio alert (requires PortAudio, OpenAL, or system-specific calls)
            log_message("INFO", "Would play audio alert.");
        }
    }

    bool is_alert_active() {
        if (alert_active) {
            double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            if (current_time - alert_start_time > config.alert_duration_ms / 1000.0) {
                alert_active = false;
            }
        }
        return alert_active;
    }

    void reset_alert() {
        alert_active = false;
        emergency_triggered = false;
    }

    void trigger_emergency_protocol() {
        if (!emergency_triggered) {
            emergency_triggered = true;
            emergency_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            log_message("CRITICAL", "EMERGENCY PROTOCOL INITIATED.");
            for (const auto& recipient : config.emergency_sms_recipients) {
                log_message("INFO", "Would send SMS alert to " + recipient);
            }
            for (const auto& recipient : config.emergency_email_recipients) {
                log_message("INFO", "Would send email alert to " + recipient);
            }
        }
    }

    bool is_emergency_active() {
        if (emergency_triggered) {
            double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            if (current_time - emergency_start_time > 60) { // Keep active for 1 minute
                emergency_triggered = false;
            }
        }
        return emergency_triggered;
    }
};

// --- Threading and Data Communication ---

class DetectionWorker : public QThread {
    Q_OBJECT

signals:
    void detection_update(const std::map<std::string, double>& data);
    void frame_update(const cv::Mat& frame);
    void status_update(const QString& message);
    void dashboard_update(const std::map<std::string, std::vector<double>>& data);
    void log_update(const QString& message);

public:
    DetectionWorker(const Config& c) : config(c), is_running(false), is_paused(false) {}
    ~DetectionWorker() { stop(); }

public slots:
    void start_detection() {
        is_running = true;
        start();
    }
    void stop_detection() {
        is_running = false;
        wait();
    }
    void pause_detection() { is_paused = true; }
    void resume_detection() { is_paused = false; }

protected:
    void run() override {
        feature_extractor = std::make_unique<FeatureExtractor>(config);
        prediction_engine = std::make_unique<PredictionEngine>(config);
        alert_system = std::make_unique<AlertSystem>(config);

        cv::VideoCapture cap(config.camera_index);
        if (!cap.isOpened()) {
            emit status_update("ERROR: Could not open camera.");
            log_message("ERROR", "Could not open camera device.");
            return;
        }

        log_message("INFO", "Detection thread started.");
        int frame_count = 0;
        double last_detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        double performance_log_timer = last_detection_time;

        while (is_running) {
            if (is_paused) {
                QThread::msleep(100);
                continue;
            }

            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                emit status_update("ERROR: Failed to read from camera.");
                log_message("WARNING", "Failed to read frame from camera.");
                QThread::msleep(500);
                continue;
            }

            double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
            frame_count++;

            if (frame_count % (config.frame_skip + 1) == 0) {
                process_frame(frame, current_time);
            }

            emit frame_update(frame);

            if (config.enable_performance_monitoring &&
                (current_time - performance_log_timer) > config.performance_log_interval) {
                double fps = 1.0 / (current_time - last_detection_time);
                emit log_update(QString("Performance: FPS ~%1").arg(fps, 0, 'f', 2));
                performance_log_timer = current_time;
            }

            last_detection_time = current_time;
            QThread::msleep(10); // Small delay
        }

        cap.release();
        log_message("INFO", "Detection thread cleaned up.");
    }

private:
    Config config;
    bool is_running;
    bool is_paused;
    std::unique_ptr<FeatureExtractor> feature_extractor;
    std::unique_ptr<PredictionEngine> prediction_engine;
    std::unique_ptr<AlertSystem> alert_system;
    cv::VideoCapture cap;

    void process_frame(cv::Mat& frame, double current_time) {
        std::map<std::string, double> detection_data = {
            {"timestamp", current_time},
            {"drowsiness_level", 0.0},
            {"confidence", 0.0},
            {"status", 0.0}, // Using 0 for "No Face", 1 for "Alert", 2 for "Drowsy"
            {"ear", 0.0},
            {"mar", 0.0},
            {"head_pitch", 0.0},
            {"head_yaw", 0.0},
            {"fatigue_score", 50.0},
            {"predicted_fatigue_score", 50.0},
            {"alert_level", 0.0} // Using 0 for none, 1 for medium, 2 for high, 3 for critical
        };

        // Placeholder: Simulate landmark detection
        // In a real implementation, you would call MediaPipe here
        // std::vector<cv::Point2f> landmarks = mediapipe_api::detect_landmarks(frame);
        // For demo, create dummy landmarks
        std::vector<cv::Point2f> landmarks(478); // Assuming 478 points
        for (int i = 0; i < 478; ++i) {
            landmarks[i] = cv::Point2f(frame.cols * 0.5 + (i % 100) - 50, frame.rows * 0.5 + (i % 100) - 50);
        }

        if (!landmarks.empty()) {
            auto features = feature_extractor->extract_features(landmarks, frame.size(), current_time);
            detection_data.insert(features.begin(), features.end());

            prediction_engine->update_buffer(features);
            auto [pred_label, confidence] = prediction_engine->predict(features);
            detection_data["drowsiness_level"] = static_cast<double>(pred_label);
            detection_data["confidence"] = confidence;

            double fatigue_score = prediction_engine->calculate_fatigue_score(features);
            double predicted_fatigue = prediction_engine->predict_fatigue_horizon(features);
            detection_data["fatigue_score"] = fatigue_score;
            detection_data["predicted_fatigue_score"] = predicted_fatigue;

            std::string alert_level = "none";
            if (pred_label == 1 || fatigue_score > 75) alert_level = "critical";
            else if (pred_label == 1 || fatigue_score > 65) alert_level = "high";
            else if (pred_label == 1 || fatigue_score > 55) alert_level = "medium";
            else if (features.at("microsleep_rate") > 0) alert_level = "high";
            detection_data["alert_level"] = (alert_level == "none" ? 0.0 : (alert_level == "medium" ? 1.0 : (alert_level == "high" ? 2.0 : 3.0)));

            if (pred_label == 1 || detection_data["alert_level"] != 0) {
                detection_data["status"] = 2.0; // Drowsy
                std::string level_str = alert_level == "none" ? "" : " (" + alert_level + ")";
                log_message("WARNING", "DROWSY" + level_str + " - EAR: " + std::to_string(features["ear"]) +
                                          ", MAR: " + std::to_string(features["mar"]) +
                                          ", Pitch: " + std::to_string(features["head_pitch"]) +
                                          ", Confidence: " + std::to_string(confidence) +
                                          ", Fatigue: " + std::to_string(fatigue_score));

                if (alert_system->is_alert_active()) {
                    detection_data["status"] = 3.0; // Drowsy - Alert Active
                } else {
                    alert_system->trigger_alert(alert_level);
                }
            } else {
                detection_data["status"] = 1.0; // Alert
                alert_system->reset_alert();
            }

            if (config.enable_visual_feedback) {
                // Draw landmarks (simplified)
                for (const auto& pt : landmarks) {
                    cv::circle(frame, pt, 1, cv::Scalar(0, 255, 0), -1);
                }
            }

            if (config.enable_advanced_visualization) {
                cv::putText(frame, "EAR: " + std::to_string(features["ear"]).substr(0, 4),
                            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                cv::putText(frame, "Fatigue: " + std::to_string(fatigue_score).substr(0, 4),
                            cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                if (detection_data["alert_level"] != 0) {
                    cv::Scalar color = (detection_data["alert_level"] == 3.0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 165, 255);
                    cv::putText(frame, "ALERT!", cv::Point(frame.cols/2 - 50, frame.rows - 20),
                                cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
                }
            }
        } else {
            detection_data["status"] = 0.0; // No Face
            log_message("WARNING", "No face detected.");
            alert_system->reset_alert();
        }

        emit detection_update(detection_data);

        if (config.enable_prediction_history) {
            std::map<std::string, std::vector<double>> dashboard_data;
            dashboard_data["perclos_history"] = feature_extractor->perclos_history;
            dashboard_data["fatigue_score"] = {fatigue_score}; // Simplified for demo
            dashboard_data["is_face_detected"] = {landmarks.empty() ? 0.0 : 1.0};
            emit dashboard_update(dashboard_data);
        }

        prediction_engine->update_prediction_history(
            static_cast<int>(detection_data["drowsiness_level"]),
            detection_data["confidence"],
            features
        );
    }
};

// --- Custom Plot Widget for C++/Qt ---
class CustomPlotWidget : public QWidget {
    Q_OBJECT

public:
    CustomPlotWidget(const QString& title, QWidget* parent = nullptr) : QWidget(parent), m_title(title) {}

public slots:
    void set_data(const std::vector<double>& data) {
        m_data = data;
        update();
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        int w = width();
        int h = height();
        painter.fillRect(rect(), QColor(59, 59, 59));

        if (m_data.empty()) return;

        double max_val = *std::max_element(m_data.begin(), m_data.end());
        double min_val = *std::min_element(m_data.begin(), m_data.end());
        double range_val = (max_val != min_val) ? max_val - min_val : 1.0;

        QPen pen(QColor(0, 255, 0), 2);
        painter.setPen(pen);

        std::vector<QPointF> points;
        double step_x = static_cast<double>(w) / std::max(static_cast<int>(m_data.size()) - 1, 1);
        for (size_t i = 0; i < m_data.size(); ++i) {
            double x = i * step_x;
            double y = h - ((m_data[i] - min_val) / range_val) * h;
            points.push_back(QPointF(x, y));
        }

        if (points.size() > 1) {
            for (size_t i = 0; i < points.size() - 1; ++i) {
                painter.drawLine(points[i], points[i+1]);
            }
        }

        painter.setPen(QColor(255, 255, 255));
        painter.drawText(5, 15, m_title);
    }

private:
    QString m_title;
    std::vector<double> m_data;
};

// --- Main GUI Application ---
class DrowsinessDetectionGUI : public QMainWindow {
    Q_OBJECT

public:
    DrowsinessDetectionGUI(QWidget* parent = nullptr) : QMainWindow(parent) {
        setup_ui();
        detection_worker = new DetectionWorker(config);
        setup_connections();
    }

    ~DrowsinessDetectionGUI() {
        if (detection_worker->isRunning()) {
            detection_worker->stop_detection();
        }
        delete detection_worker;
    }

private slots:
    void start_detection() {
        update_config_from_ui();
        detection_worker->start_detection();
        start_button->setEnabled(false);
        stop_button->setEnabled(true);
        status_label->setText("Status: Running");
        status_label->setStyleSheet("color: #00ff00;");
    }

    void stop_detection() {
        detection_worker->stop_detection();
        start_button->setEnabled(true);
        stop_button->setEnabled(false);
        status_label->setText("Status: Stopped");
        status_label->setStyleSheet("color: #ff5555;");
        drowsiness_status_label->setText("Status: N/A");
        fatigue_score_label->setText("Current Score: N/A");
        predicted_fatigue_label->setText("Predicted Score (15min): N/A");
        fatigue_progress_bar->setValue(50);
        ear_label->setText("EAR: N/A");
        mar_label->setText("MAR: N/A");
        pitch_label->setText("Head Pitch: N/A");
        yaw_label->setText("Head Yaw: N/A");
        confidence_label->setText("Confidence: N/A");
        video_label->clear();
    }

    void update_detection_info(const std::map<std::string, double>& data) {
        double status_val = data.at("status");
        if (status_val == 2.0) { // Drowsy
            drowsiness_status_label->setText("Status: DROWSY");
            drowsiness_status_label->setStyleSheet("color: #ff5555;");
        } else if (status_val == 3.0) { // Drowsy - Alert Active
            drowsiness_status_label->setText("Status: DROWSY - ALERT ACTIVE");
            drowsiness_status_label->setStyleSheet("color: #ff0000; font-weight: bold;");
        } else if (status_val == 1.0) { // Alert
            drowsiness_status_label->setText("Status: ALERT");
            drowsiness_status_label->setStyleSheet("color: #00aaff;");
        } else { // No Face
            drowsiness_status_label->setText("Status: NO FACE");
            drowsiness_status_label->setStyleSheet("color: #ffaa00;");
        }

        ear_label->setText(QString("EAR: %1").arg(data.at("ear"), 0, 'f', 3));
        mar_label->setText(QString("MAR: %1").arg(data.at("mar"), 0, 'f', 3));
        pitch_label->setText(QString("Head Pitch: %1°").arg(data.at("head_pitch"), 0, 'f', 2));
        yaw_label->setText(QString("Head Yaw: %1°").arg(data.at("head_yaw"), 0, 'f', 2));
        confidence_label->setText(QString("Confidence: %1").arg(data.at("confidence"), 0, 'f', 2));

        fatigue_score_label->setText(QString("Current Score: %1").arg(data.at("fatigue_score"), 0, 'f', 1));
        predicted_fatigue_label->setText(QString("Predicted Score (15min): %1").arg(data.at("predicted_fatigue_score"), 0, 'f', 1));
        fatigue_progress_bar->setValue(static_cast<int>(data.at("fatigue_score")));
    }

    void update_video_feed(const cv::Mat& frame) {
        if (!frame.empty()) {
            cv::Mat rgb_frame;
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
            QImage qimg(rgb_frame.data, rgb_frame.cols, rgb_frame.rows, rgb_frame.step, QImage::Format_RGB888);
            QPixmap pixmap = QPixmap::fromImage(qimg);
            video_label->setPixmap(pixmap.scaled(video_label->width(), video_label->height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

    void update_status_bar(const QString& message) {
        status_label->setText(message);
        if (message.contains("ERROR")) {
            status_label->setStyleSheet("color: #ff0000;");
        } else {
            status_label->setStyleSheet("color: #ffff00;");
        }
    }

    void update_dashboard(const std::map<std::string, std::vector<double>>& data) {
        auto perclos_it = data.find("perclos_history");
        if (perclos_it != data.end()) {
            perclos_plot->set_data(perclos_it->second);
        }
        auto fatigue_it = data.find("fatigue_score");
        if (fatigue_it != data.end() && !fatigue_it->second.empty()) {
            // For simplicity, just update with the last value
            fatigue_score_plot->set_data({fatigue_it->second.back()});
        }
    }

    void append_to_log(const QString& message) {
        log_text_edit->append(message);
        log_text_edit->moveCursor(QTextCursor::End);
    }

private:
    Config config;
    DetectionWorker* detection_worker;
    QPushButton* start_button;
    QPushButton* stop_button;
    QLabel* status_label;
    QLabel* drowsiness_status_label;
    QLabel* fatigue_score_label;
    QLabel* predicted_fatigue_label;
    QProgressBar* fatigue_progress_bar;
    QLabel* ear_label;
    QLabel* mar_label;
    QLabel* pitch_label;
    QLabel* yaw_label;
    QLabel* confidence_label;
    QLabel* video_label;
    QTextEdit* log_text_edit;
    CustomPlotWidget* perclos_plot;
    CustomPlotWidget* fatigue_score_plot;

    void setup_ui() {
        setWindowTitle("Industrial Driver Drowsiness Detection System - C++");
        resize(config.gui_width, config.gui_height);

        if (config.theme == "dark") {
            setStyleSheet(
                "QMainWindow, QWidget, QGroupBox, QLabel, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QTabWidget::pane {"
                "    background-color: #2b2b2b; color: #ffffff;"
                "}"
                "QPushButton {"
                "    background-color: #3a3a3a; color: #ffffff; border: 1px solid #555555; padding: 5px;"
                "}"
                "QPushButton:hover {"
                "    background-color: #4a4a4a;"
                "}"
                "QTextEdit, QTableWidget {"
                "    background-color: #1e1e1e; color: #ffffff; border: 1px solid #555555;"
                "}"
                "QProgressBar::chunk {"
                "    background-color: #00aaff;"
                "}"
            );
        }

        QWidget* central_widget = new QWidget();
        setCentralWidget(central_widget);
        QHBoxLayout* main_layout = new QHBoxLayout(central_widget);

        QTabWidget* tab_widget = new QTabWidget();
        main_layout->addWidget(tab_widget);

        // Main Tab
        QWidget* main_tab = new QWidget();
        QHBoxLayout* main_layout_tab = new QHBoxLayout(main_tab);

        QWidget* left_panel = new QWidget();
        QVBoxLayout* left_layout = new QVBoxLayout(left_panel);

        QGroupBox* status_group = new QGroupBox("System Status");
        QVBoxLayout* status_layout = new QVBoxLayout(status_group);
        status_label = new QLabel("System Idle");
        status_label->setFont(QFont(QString::fromStdString(config.font_family.c_str()), config.font_size + 2));
        status_label->setStyleSheet("color: #00ff00;");
        status_layout->addWidget(status_label);

        drowsiness_status_label = new QLabel("Status: N/A");
        drowsiness_status_label->setFont(QFont(QString::fromStdString(config.font_family.c_str()), config.font_size + 2));
        drowsiness_status_label->setStyleSheet("color: #00aaff;");
        status_layout->addWidget(drowsiness_status_label);

        QGroupBox* fatigue_group = new QGroupBox("Fatigue Metrics");
        QGridLayout* fatigue_layout = new QGridLayout(fatigue_group);
        fatigue_score_label = new QLabel("Current Score: N/A");
        predicted_fatigue_label = new QLabel("Predicted Score (15min): N/A");
        fatigue_progress_bar = new QProgressBar();
        fatigue_progress_bar->setRange(0, 100);
        fatigue_progress_bar->setValue(50);
        fatigue_layout->addWidget(fatigue_score_label, 0, 0);
        fatigue_layout->addWidget(predicted_fatigue_label, 0, 1);
        fatigue_layout->addWidget(fatigue_progress_bar, 1, 0, 1, 2);

        QGroupBox* features_group = new QGroupBox("Real-Time Features");
        QGridLayout* features_layout = new QGridLayout(features_group);
        ear_label = new QLabel("EAR: N/A");
        mar_label = new QLabel("MAR: N/A");
        pitch_label = new QLabel("Head Pitch: N/A");
        yaw_label = new QLabel("Head Yaw: N/A");
        confidence_label = new QLabel("Confidence: N/A");
        features_layout->addWidget(ear_label, 0, 0);
        features_layout->addWidget(mar_label, 0, 1);
        features_layout->addWidget(pitch_label, 1, 0);
        features_layout->addWidget(yaw_label, 1, 1);
        features_layout->addWidget(confidence_label, 2, 0, 1, 2);

        QGroupBox* controls_group = new QGroupBox("Controls");
        QVBoxLayout* controls_layout = new QVBoxLayout(controls_group);
        start_button = new QPushButton("Start Detection");
        stop_button = new QPushButton("Stop Detection");
        stop_button->setEnabled(false);
        QPushButton* save_config_button = new QPushButton("Save Configuration");
        log_text_edit = new QTextEdit();
        log_text_edit->setMaximumHeight(150);
        log_text_edit->setReadOnly(true);

        controls_layout->addWidget(start_button);
        controls_layout->addWidget(stop_button);
        controls_layout->addWidget(save_config_button);
        controls_layout->addWidget(new QLabel("Log:"));
        controls_layout->addWidget(log_text_edit);

        left_layout->addWidget(status_group);
        left_layout->addWidget(fatigue_group);
        left_layout->addWidget(features_group);
        left_layout->addWidget(controls_group);
        left_layout->addStretch();

        QWidget* right_panel = new QWidget();
        QVBoxLayout* right_layout = new QVBoxLayout(right_panel);
        video_label = new QLabel();
        video_label->setAlignment(Qt::AlignCenter);
        video_label->setMinimumSize(640, 480);
        video_label->setStyleSheet("border: 1px solid #555555;");
        right_layout->addWidget(video_label);

        main_layout_tab->addWidget(left_panel);
        main_layout_tab->addWidget(right_panel);

        // Dashboard Tab
        QWidget* dashboard_tab = new QWidget();
        QVBoxLayout* dashboard_layout = new QVBoxLayout(dashboard_tab);

        perclos_plot = new CustomPlotWidget("PERCLOS History");
        fatigue_score_plot = new CustomPlotWidget("Fatigue Score History");

        QGridLayout* plots_layout = new QGridLayout();
        plots_layout->addWidget(perclos_plot, 0, 0);
        plots_layout->addWidget(fatigue_score_plot, 0, 1);

        dashboard_layout->addLayout(plots_layout);

        tab_widget->addTab(main_tab, "Main");
        tab_widget->addTab(dashboard_tab, "Dashboard");
    }

    void setup_connections() {
        connect(start_button, &QPushButton::clicked, this, &DrowsinessDetectionGUI::start_detection);
        connect(stop_button, &QPushButton::clicked, this, &DrowsinessDetectionGUI::stop_detection);
        connect(detection_worker, &DetectionWorker::detection_update, this, &DrowsinessDetectionGUI::update_detection_info);
        connect(detection_worker, &DetectionWorker::frame_update, this, &DrowsinessDetectionGUI::update_video_feed);
        connect(detection_worker, &DetectionWorker::status_update, this, &DrowsinessDetectionGUI::update_status_bar);
        connect(detection_worker, &DetectionWorker::dashboard_update, this, &DrowsinessDetectionGUI::update_dashboard);
        connect(detection_worker, &DetectionWorker::log_update, this, &DrowsinessDetectionGUI::append_to_log);
    }

    void update_config_from_ui() {
        // This is a simplified version. In a full UI, you would have spinboxes/checkboxes
        // and connect their signals to update the config struct.
        // For this example, we assume the struct is updated elsewhere or is constant.
    }
};

#include "main.moc" // Required for Qt signals/slots in C++

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    DrowsinessDetectionGUI window;
    window.show();

    return app.exec();
}