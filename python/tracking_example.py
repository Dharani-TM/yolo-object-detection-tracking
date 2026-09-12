import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import sys
import argparse

# --- 設定 ---
parser = argparse.ArgumentParser(description="OpenCV Tracking Example")
parser.add_argument(
    '--tracker', 
    type=str, 
    default='CSRT', 
    help="選擇要使用的追蹤演算法模型。支援的選項包含：DASIAMRPN, NANOTRACK, CSRT, KCF, MIL, MOSSE 等。預設為 CSRT。"
)
args = parser.parse_args()
tracker_type = args.tracker.upper()

# Tracker Model Paths
NANOTRACK_BACKBONE = "nanotrack_backbone.onnx"
NANOTRACK_HEAD = "nanotrack_head.onnx"

# DaSiamRPN requires THREE files
DASIAMRPN_MODEL = "dasiamrpn_model.onnx"
DASIAMRPN_KERNEL_CLS1 = "dasiamrpn_kernel_cls1.onnx"
DASIAMRPN_KERNEL_R1 = "dasiamrpn_kernel_r1.onnx"

def create_tracker(tracker_type):
    """Creates a tracker based on the specified type, with error handling."""
    if tracker_type == 'DASIAMRPN':
        try:
            import os
            required_files = [DASIAMRPN_MODEL, DASIAMRPN_KERNEL_CLS1, DASIAMRPN_KERNEL_R1]
            if any(not os.path.exists(f) for f in required_files):
                print(f"\nError: DaSiamRPN models missing! Please ensure all 3 files are in the python/ directory.")
                return None
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = DASIAMRPN_MODEL
            params.kernel_cls1 = DASIAMRPN_KERNEL_CLS1
            params.kernel_r1 = DASIAMRPN_KERNEL_R1
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            return cv2.TrackerDaSiamRPN_create(params)
        except Exception as e:
            print(f"Error creating DaSiamRPN: {e}")
            return None

    if tracker_type == 'NANOTRACK':
        try:
            import os
            if not os.path.exists(NANOTRACK_BACKBONE) or not os.path.exists(NANOTRACK_HEAD):
                print(f"\nError: NanoTrack models not found!")
                return None
            params = cv2.TrackerNano_Params()
            params.backbone = NANOTRACK_BACKBONE
            params.neckhead = NANOTRACK_HEAD
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            return cv2.TrackerNano_create(params)
        except Exception as e:
            print(f"Error creating NanoTrack: {e}")
            return None

    tracker_builders = {
        'CSRT': cv2.legacy.TrackerCSRT.create,
        'KCF': cv2.legacy.TrackerKCF.create,
        'MIL': cv2.legacy.TrackerMIL.create,
        'MOSSE': cv2.legacy.TrackerMOSSE.create,
        'MEDIANFLOW': cv2.legacy.TrackerMedianFlow.create,
        'TLD': cv2.legacy.TrackerTLD.create,
    }
    try:
        builder = tracker_builders.get(tracker_type)
        if builder:
            return builder()
        else:
            print(f"錯誤：無效的追蹤器類型 '{tracker_type}'")
            return None
    except AttributeError:
        print("\n錯誤：您的 OpenCV 版本缺少追蹤器模組 (AttributeError)。")
        print("這通常是因為未安裝 'contrib' 版本的 OpenCV。")
        print("請嘗試執行以下指令來修復：")
        print("  pip uninstall opencv-python opencv-contrib-python")
        print("  pip install opencv-contrib-python\n")
        return None

# --- 主程式 ---
if __name__ == '__main__':
    # 建立追蹤器
    tracker = create_tracker(tracker_type)
    if not tracker:
        sys.exit()

    # 開啟攝影機
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("無法開啟攝影機")
        sys.exit()

    # 讀取第一幀
    ok, frame = video.read()
    if not ok:
        print("無法讀取影像")
        sys.exit()

    # 讓使用者手動選擇要追蹤的物件 (ROI: Region of Interest)
    # 按下 Enter 或 Space 確認選取，按下 C 取消
    display_frame = frame.copy()
    cv2.putText(display_frame, "Select object and press ENTER to start tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    bbox = cv2.selectROI("Select Object to Track", display_frame, False)
    
    if not bbox or bbox[2] == 0 or bbox[3] == 0:
        print("未選取物件，程式結束。")
        sys.exit()

    # 初始化追蹤器
    res = tracker.init(frame, bbox)
    ok = res is None or res
    cv2.destroyWindow("Select Object to Track")

    while True:
        # 讀取新的一幀
        ok, frame = video.read()
        if not ok:
            break

        # 開始計時
        timer = cv2.getTickCount()

        # 更新追蹤器
        ok, bbox = tracker.update(frame)

        # 計算 FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 繪製邊界框
        if ok:
            # 追蹤成功
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            status_text = "Tracking"
            status_color = (0, 255, 0)
        else:
            # 追蹤失敗
            status_text = "Tracking Failure"
            status_color = (0, 0, 255)

        # 顯示狀態和 FPS
        cv2.putText(frame, tracker_type + " Tracker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 170, 50), 2)
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 顯示結果
        cv2.imshow("Tracking", frame)

        # 按下 'q' 鍵結束
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
