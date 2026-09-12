#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <iostream>

// This version uses OpenCV's DNN module to run a real YOLOv8 ONNX model.

using namespace cv;
using namespace std;
using namespace std::chrono;

struct ObjectInfo {
    Mat crop;
    steady_clock::time_point last_seen;
    steady_clock::time_point first_seen;
    string name;
    bool visible;
};

const int CAMERA_SOURCE = 0;
const string MODEL_PATH = "yolov8n.onnx"; // Make sure this ONNX model is in the execution directory
const int CROP_SIZE = 128;
const double APPEAR_THRESHOLD = 1.0; // seconds
const double REMOVE_DELAY = 3.0;      // seconds
const double REFRESH_INTERVAL = 0.2;  // seconds
const int OBJECTS_PER_ROW = 8;
const int MAX_OBJECTS = OBJECTS_PER_ROW * 8;
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.5;

vector<int> object_slots(MAX_OBJECTS, -1); // Each slot holds an obj_id or -1
map<int, ObjectInfo> object_dict;
// Class names for COCO dataset (used by yolov8n)
const vector<string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

int assign_slot(int obj_id) {
    // If already assigned, return its slot
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == obj_id)
            return idx;
    }
    // Find first empty slot
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == -1) {
            object_slots[idx] = obj_id;
            return idx;
        }
    }
    return -1;
}

void remove_slot(int obj_id) {
    for (int idx = 0; idx < object_slots.size(); ++idx) {
        if (object_slots[idx] == obj_id) {
            object_slots[idx] = -1;
        }
    }
}

// This struct will hold the output of the YOLO model
struct Detection {
    int class_id;
    float confidence;
    Rect bbox;
};

vector<Detection> run_inference(dnn::Net& net, const Mat& frame) {
    Mat blob;
    dnn::blobFromImage(frame, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    net.setInput(blob);
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    
    // The output of YOLOv8 is [1, 84, 8400]. We need to transpose it to [1, 8400, 84]
    // to process it row by row.
    Mat mat(outputs[0].size[1], outputs[0].size[2], CV_32F, (float*)outputs[0].data); // Create a 2D Mat header for the output
    Mat transposed_mat = mat.t(); // Transpose to get shape [8400, 84]
    // The transposed matrix is not continuous in memory. We must clone it to create a continuous block.
    Mat output_buffer = transposed_mat.clone();


    float x_factor = frame.cols / INPUT_WIDTH;
    float y_factor = frame.rows / INPUT_HEIGHT;
    float *data = (float *)output_buffer.data;
    const int dimensions = 84; // 4 (bbox) + 80 (classes)
    const int rows = 8400;
    const int num_classes = dimensions - 4;

    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float *classes_scores_ptr = data + 4;

        // Manually find the max score and class ID to bypass potential minMaxLoc issues in older OpenCV.
        float max_class_score = 0.0f;
        int class_id = 0;
        for (int j = 0; j < num_classes; j++) {
            if (classes_scores_ptr[j] > max_class_score) {
                max_class_score = classes_scores_ptr[j];
                class_id = j;
            }
        }

        if (max_class_score > CONFIDENCE_THRESHOLD) {
            confidences.push_back(max_class_score);
            class_ids.push_back(class_id);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(Rect(left, top, width, height));
        }
        data += dimensions;
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_result);

    vector<Detection> detections;
    for (int idx : nms_result) {
        detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
    }

    return detections;
}

int main() {
    dnn::Net net = dnn::readNet(MODEL_PATH);
    net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(dnn::DNN_TARGET_CPU);

    VideoCapture cap(CAMERA_SOURCE);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open camera source " << CAMERA_SOURCE << endl;
        return -1;
    }
    cout << "Camera connected successfully. Using model: " << MODEL_PATH << ". Press 'q' to exit." << endl;

    auto last_panel_update = steady_clock::now();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Unable to read frame, exiting." << endl;
            break;
        }
        auto now = steady_clock::now();

        // Run YOLO inference
        vector<Detection> detections = run_inference(net, frame);
        
        // This part is a placeholder for a real tracker.
        // For simplicity, we use the detection's index in the current frame as a temporary ID.
        // A real implementation would use a tracking algorithm (like ByteTrack) to maintain consistent IDs across frames.
        set<int> current_ids;
        for (int i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            int obj_id = i; // Simplified ID for this example
            Rect bbox = det.bbox & Rect(0, 0, frame.cols, frame.rows);
            if (bbox.width > 0 && bbox.height > 0) {
                Mat crop = frame(bbox).clone();
                resize(crop, crop, Size(CROP_SIZE, CROP_SIZE));
                if (object_dict.find(obj_id) == object_dict.end()) {
                    // Add boundary check for class_id to prevent out-of-bounds access
                    string obj_name = "Unknown";
                    if (det.class_id >= 0 && det.class_id < CLASS_NAMES.size()) {
                        obj_name = CLASS_NAMES[det.class_id];
                    }
                    object_dict[obj_id] = {crop, now, now, obj_name, true};
                } else {
                    object_dict[obj_id].crop = crop;
                    object_dict[obj_id].last_seen = now;
                    object_dict[obj_id].visible = true;
                }
                current_ids.insert(obj_id);
            }
        }
        // Mark objects not currently visible
        for (auto& [obj_id, info] : object_dict) {
            if (current_ids.find(obj_id) == current_ids.end()) {
                info.visible = false;
            }
        }
        // Remove objects that have disappeared for more than REMOVE_DELAY seconds
        vector<int> remove_ids;
        for (const auto& [obj_id, info] : object_dict) {
            if (!info.visible && duration_cast<duration<double>>(now - info.last_seen).count() > REMOVE_DELAY) {
                remove_ids.push_back(obj_id);
            }
        }
        for (int obj_id : remove_ids) {
            object_dict.erase(obj_id);
            remove_slot(obj_id);
        }

        // Draw bounding boxes on preview
        for (int i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            int obj_id = i; // Simplified ID
            string obj_name = "Unknown";
            if (det.class_id >= 0 && det.class_id < CLASS_NAMES.size()) {
                obj_name = CLASS_NAMES[det.class_id];
            }
            rectangle(frame, det.bbox, Scalar(0, 255, 0), 2);
            string label = obj_name + " (ID:" + to_string(obj_id) + ")";
            putText(frame, label, Point(det.bbox.x, det.bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
        imshow("YOLO Real-Time Tracking [C++] (Press 'q' to exit)", frame);

        // Refresh object display window every REFRESH_INTERVAL seconds
        if (duration_cast<duration<double>>(now - last_panel_update).count() > REFRESH_INTERVAL) {
            // Create a fixed-size vector for all slots to avoid out-of-bounds access.
            vector<Mat> slot_crops(MAX_OBJECTS, Mat::zeros(CROP_SIZE + 24, CROP_SIZE, CV_8UC3));

            int last_active_slot = -1;
            for (int idx = 0; idx < object_slots.size(); ++idx) {
                int obj_id = object_slots[idx];
                if (obj_id != -1 && object_dict.find(obj_id) != object_dict.end()) {
                    auto& info = object_dict[obj_id];
                    double appear_time = duration_cast<duration<double>>(now - info.first_seen).count();
                    if (appear_time >= APPEAR_THRESHOLD) {
                        last_active_slot = idx; // Keep track of the last slot with content
                        Mat label_img = Mat::zeros(24, CROP_SIZE, CV_8UC3);
                        string text = info.name + " (ID:" + to_string(obj_id) + ")";
                        putText(label_img, text, Point(5, 18), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
                        Mat crop_with_label;
                        vconcat(label_img, info.crop, crop_with_label);
                        rectangle(crop_with_label, Point(0,24), Point(CROP_SIZE-1,CROP_SIZE+23),
                                info.visible ? Scalar(0,255,0) : Scalar(0,165,255), 2);
                        slot_crops[idx] = crop_with_label;
                    }
                }
            }

            // Determine how many rows to display based on the last active slot.
            int rows_to_show = (last_active_slot == -1) ? 1 : (last_active_slot / OBJECTS_PER_ROW) + 1;

            vector<Mat> rows;
            for (int i = 0; i < rows_to_show; ++i) {
                Mat row;
                // Directly create a horizontal concatenation from the sub-vector of slot_crops.
                hconcat(vector<Mat>(slot_crops.begin() + i * OBJECTS_PER_ROW, slot_crops.begin() + (i + 1) * OBJECTS_PER_ROW), row);
                rows.push_back(row);
            }

            if (!rows.empty()) {
                Mat objects_panel;
                vconcat(rows, objects_panel);
                imshow("Objects", objects_panel);
            }
            last_panel_update = now;
        }

        char key = (char)waitKey(1);
        if (key == 'q' || key == 'Q') break;
    }

    cap.release();
    destroyAllWindows();
    cout << "Experiment finished." << endl;
    return 0;
}