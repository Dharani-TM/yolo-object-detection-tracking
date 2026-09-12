import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
import argparse
import numpy as np
import re
from ultralytics import YOLO

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="YOLO Object Detection and Tracking with OpenCV Single-Object Trackers")
parser.add_argument(
    '--source',
    type=str,
    default='0',
    help="Camera source (e.g., '0' for default camera, '/dev/video1', or 'video.mp4'). Default is '0'."
)
parser.add_argument(
    '--source_cfg',
    type=str,
    default=None,
    help="Directly specify camera resolution and FPS (e.g., '1920x1080@30'). If not specified, an interactive menu will appear for cameras."
)
parser.add_argument(
    '--model',
    type=str,
    default='yolov8n.pt',
    help="Path to the YOLO model to load (e.g., 'yolov8n.pt' or 'yolo11n.pt'). Default is 'yolov8n.pt'."
)
parser.add_argument(
    '--classes', 
    nargs='*', 
    type=int, 
    help="List of class IDs to detect. If not specified, all classes are detected. e.g., --classes 0 2"
)
parser.add_argument(
    '--conf', 
    type=float, 
    default=0.4, 
    help="Confidence threshold for detection (e.g., 0.25)."
)
parser.add_argument(
    '--imgsz', 
    type=int, 
    default=640, 
    help="Image size for inference (e.g., 640, 1280)."
)
parser.add_argument(
    '--det-freq',
    type=float,
    default=0.0,
    help="Detection frequency in Hz (FPS). 0 means use the source's native frequency (no limit)."
)
parser.add_argument(
    '--track-freq',
    type=float,
    default=0.0,
    help="Tracking frequency in Hz (FPS) for the OpenCV single-object tracker. 0 means use the source's native frequency."
)
args = parser.parse_args()

MODEL_NAME = args.model
TARGET_CLASSES = args.classes if args.classes is not None else []

# Set tracker config (optional, for tracking object IDs)
TRACKER_CONFIG = 'bytetrack.yaml' # Choose a tracker

CROP_SIZE = 48
APPEAR_THRESHOLD = 1    # Seconds an object must be continuously detected before display
REMOVE_DELAY = 3        # Seconds after disappearance before removing from display
REFRESH_INTERVAL = 0.2  # Refresh object window every few seconds
OBJECTS_PER_ROW = 8   # Number of objects per row

# Track each obj_id: {'crop': ..., 'last_seen': ..., 'first_seen': ..., 'name': ..., 'visible': ..., 'tracker': ..., 'bbox': ...}
object_dict = {}
last_panel_update = time.time()

# --- Global variables for selection ---
selected_obj_id = None
last_known_boxes = [] # To store the latest bounding boxes for click detection
cv_tracker = None
is_cv_tracking = False
CV_TRACKER_TYPE = 'NANOTRACK' # Options: 'DASIAMRPN', 'NANOTRACK', 'CSRT', 'KCF', 'MIL'
use_bg_tracking = True # Toggle for integrated tracking on background objects

SMOOTH_ALPHA = 0.05  # 平滑係數 (0.0~1.0)，越小越平滑

def smooth_bbox(old_bbox, new_bbox):
    if old_bbox is None: return new_bbox
    return tuple(SMOOTH_ALPHA * np.array(new_bbox) + (1 - SMOOTH_ALPHA) * np.array(old_bbox))

def get_color(idx):
    """Returns a consistent color for a given ID."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), 
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255)
    ]
    return colors[int(idx) % len(colors)]

# Tracker Model Paths
NANOTRACK_BACKBONE = "nanotrack_backbone.onnx"
NANOTRACK_HEAD = "nanotrack_head.onnx"

# DaSiamRPN requires THREE files
DASIAMRPN_MODEL = "dasiamrpn_model.onnx"
DASIAMRPN_KERNEL_CLS1 = "dasiamrpn_kernel_cls1.onnx"
DASIAMRPN_KERNEL_R1 = "dasiamrpn_kernel_r1.onnx"

tracking_request = None # To handle requests from the mouse callback

# --- Manual Drawing & Mode UI State ---
app_mode = 'YOLO'  # 'YOLO' or 'MANUAL'
is_drawing = False
drawing_start = None
drawing_current = None

UI_HEIGHT = 100
MODE_BTNS = {
    'YOLO': (130, 5, 230, 25),
    'MANUAL': (240, 5, 340, 25)
}
TRACKER_BTNS = {
    'NANOTRACK': (130, 30, 210, 50),
    'DASIAMRPN': (220, 30, 310, 50),
    'CSRT': (320, 30, 380, 50),
    'KCF': (390, 30, 440, 50),
    'MIL': (450, 30, 500, 50)
}
BG_TRACK_TOGGLE_BTN = (10, 55, 230, 75)

# Performance tracking variables
det_calc_time = 0.0
det_actual_fps = 0.0
track_calc_time = 0.0
track_actual_fps = 0.0
last_det_exec_time = time.time()
last_track_exec_time = time.time()


def scan_camera_modes(camera_source):
    print(f"\nScanning supported modes for Camera {camera_source}...")
    temp_cap = cv2.VideoCapture(camera_source)
    if not temp_cap.isOpened():
        return []
    
    supported_modes = set()
    common_resolutions = [(1920, 1080), (1280, 720), (800, 600), (640, 480), (320, 240)]
    common_fps = [60, 30, 24, 15]

    for w, h in common_resolutions:
        for fps in common_fps:
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            temp_cap.set(cv2.CAP_PROP_FPS, fps)
            actual_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
            if actual_w == w and actual_h == h and actual_fps > 0:
                 supported_modes.add((actual_w, actual_h, actual_fps))
    temp_cap.release()
    return sorted(list(supported_modes), key=lambda x: (x[0], x[2]), reverse=True)

# Load YOLO model
try:
    model = YOLO(MODEL_NAME)
    print(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# --- Camera Setup ---
source_val = int(args.source) if args.source.isdigit() else args.source
is_camera = isinstance(source_val, int) or str(source_val).startswith('/dev/video')

req_w, req_h, req_fps = None, None, None

if is_camera:
    if args.source_cfg:
        match = re.match(r"(\d+)x(\d+)@(\d+)", args.source_cfg)
        if match:
            req_w, req_h, req_fps = map(int, match.groups())
            print(f"Using provided source config: {req_w}x{req_h} @ {req_fps} FPS")
        else:
            print("Invalid format for --source_cfg. Expected WIDTHxHEIGHT@FPS (e.g., 1280x720@30).")
    else:
        modes = scan_camera_modes(source_val)
        if modes:
            print("\nPlease select a resolution and FPS:")
            for i, (w, h, fps) in enumerate(modes):
                print(f"  [{i}] {w}x{h} @ {fps} FPS")
            mode_idx = -1
            while not (0 <= mode_idx < len(modes)):
                try:
                    mode_idx = int(input(f"Enter mode index [0-{len(modes)-1}]: "))
                except ValueError:
                    print("Invalid input. Please enter a number.")
            req_w, req_h, req_fps = modes[mode_idx]
        else:
            print("Warning: Could not determine supported modes automatically.")

cap = cv2.VideoCapture(source_val)
if req_w and req_h and req_fps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FPS, req_fps)

if not cap.isOpened():
    print(f"Error: Unable to open source: {args.source}")
    exit()

source_fps = cap.get(cv2.CAP_PROP_FPS)
if source_fps <= 0 or np.isnan(source_fps):
    source_fps = req_fps if req_fps else 30.0 # fallback

det_freq = args.det_freq if 0 < args.det_freq <= source_fps else source_fps
track_freq = args.track_freq if 0 < args.track_freq <= source_fps else source_fps

det_interval = 1.0 / det_freq if args.det_freq > 0 else 0.0
track_interval = 1.0 / track_freq if args.track_freq > 0 else 0.0

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_type = "Camera" if is_camera else "Video file"
print(f"{source_type} opened successfully at {actual_w}x{actual_h}.")
print(f"Source FPS: {source_fps:.2f}, Detection Freq: {'No limit' if args.det_freq == 0 else f'{det_freq:.2f} Hz'}, Tracking Freq: {'No limit' if args.track_freq == 0 else f'{track_freq:.2f} Hz'}")

# --- OpenCV Tracker Management ---
def create_cv_tracker():
    """Creates an OpenCV tracker based on the specified type, with error handling."""
    if CV_TRACKER_TYPE == 'DASIAMRPN':
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

    if CV_TRACKER_TYPE == 'NANOTRACK':
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
        builder = tracker_builders.get(CV_TRACKER_TYPE)
        if builder:
            return builder()
        else:
            print(f"Error: Invalid tracker type '{CV_TRACKER_TYPE}'.")
            return None
    except AttributeError:
        print("\nError: Your OpenCV version is missing tracker modules.")
        return None

# --- Mouse Callback for Object Selection ---
def select_object_callback(event, x, y, flags, param):
    global selected_obj_id, tracking_request, app_mode, is_drawing, drawing_start, drawing_current, CV_TRACKER_TYPE, use_bg_tracking
    window_name = param['name']

    # Handle Mode UI Clicks
    if window_name == main_window_name:
        if y < UI_HEIGHT:
            if event == cv2.EVENT_LBUTTONDOWN:
                if MODE_BTNS['YOLO'][0] <= x <= MODE_BTNS['YOLO'][2] and MODE_BTNS['YOLO'][1] <= y <= MODE_BTNS['YOLO'][3]:
                    app_mode = 'YOLO'
                    tracking_request = {'action': 'stop'}
                    selected_obj_id = None
                    is_drawing = False
                elif MODE_BTNS['MANUAL'][0] <= x <= MODE_BTNS['MANUAL'][2] and MODE_BTNS['MANUAL'][1] <= y <= MODE_BTNS['MANUAL'][3]:
                    app_mode = 'MANUAL'
                    tracking_request = {'action': 'stop'}
                    selected_obj_id = None
                elif BG_TRACK_TOGGLE_BTN[0] <= x <= BG_TRACK_TOGGLE_BTN[2] and BG_TRACK_TOGGLE_BTN[1] <= y <= BG_TRACK_TOGGLE_BTN[3]:
                    use_bg_tracking = not use_bg_tracking
                    # If turning OFF, clear existing background trackers
                    if not use_bg_tracking:
                        for info in object_dict.values(): info['tracker'] = None
                else:
                    for tracker_name, coords in TRACKER_BTNS.items():
                        if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                            CV_TRACKER_TYPE = tracker_name
                            tracking_request = {'action': 'stop'}
                            selected_obj_id = None
                            # 聯動邏輯：切換模型時清除所有現有追蹤器，強制下一個週期重新建立
                            for info in object_dict.values():
                                info['tracker'] = None
                            break
            return
        
        # Offset y for frame logic
        y -= UI_HEIGHT

    if app_mode == 'YOLO':
        if event == cv2.EVENT_LBUTTONDOWN:
            found_id = None
            if window_name == objects_window_name:
                col = x // CROP_SIZE
                row = y // (CROP_SIZE + 24)
                slot_idx = row * OBJECTS_PER_ROW + col
                if 0 <= slot_idx < len(object_slots):
                    found_id = object_slots[slot_idx]
            else:
                for box in last_known_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        found_id = int(box.id.item())
                        break
            
            if found_id is not None:
                if selected_obj_id == found_id:
                    selected_obj_id = None
                    tracking_request = {'action': 'stop'}
                else:
                    selected_obj_id = found_id
                    tracking_request = {'action': 'start', 'id': found_id}
            elif is_cv_tracking:
                tracking_request = {'action': 'stop'}
    elif app_mode == 'MANUAL':
        if window_name == main_window_name:
            if event == cv2.EVENT_LBUTTONDOWN:
                is_drawing = True
                drawing_start = (x, y)
                drawing_current = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if is_drawing:
                    drawing_current = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if is_drawing:
                    is_drawing = False
                    drawing_current = (x, y)
                    x1, y1 = drawing_start
                    x2, y2 = drawing_current
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_max - x_min > 5 and y_max - y_min > 5:
                        tracking_request = {'action': 'start_manual', 'bbox': [x_min, y_min, x_max - x_min, y_max - y_min]}
                    else:
                        tracking_request = {'action': 'stop'}

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    for i in range(x1, x2, dash_length * 2):
        cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
        cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
    for i in range(y1, y2, dash_length * 2):
        cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
        cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

main_window_name = f"YOLO Real-Time Tracking [{MODEL_NAME}] (Press 'q' to exit)"
objects_window_name = "Objects"
cv2.namedWindow(main_window_name)
cv2.namedWindow(objects_window_name)
cv2.setMouseCallback(main_window_name, select_object_callback, param={'name': main_window_name})
cv2.setMouseCallback(objects_window_name, select_object_callback, param={'name': objects_window_name})

print(f"{source_type} {args.source} opened successfully. Using model: {MODEL_NAME}. Press 'q' to exit.")

MAX_OBJECTS = OBJECTS_PER_ROW * 8
object_slots = [None] * MAX_OBJECTS

def assign_slot(obj_id):
    for idx, oid in enumerate(object_slots):
        if oid == obj_id: return idx
    for idx, oid in enumerate(object_slots):
        if oid is None:
            object_slots[idx] = obj_id
            return idx
    return None

def remove_slot(obj_id):
    for idx, oid in enumerate(object_slots):
        if oid == obj_id: object_slots[idx] = None

def start_cv_tracking(frame, obj_id):
    global cv_tracker, is_cv_tracking, selected_obj_id
    box_to_track = None
    for box in last_known_boxes:
        if box.id is not None and int(box.id.item()) == obj_id:
            box_to_track = box
            break
    
    if box_to_track:
        coords = box_to_track.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        h_f, w_f, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        width, height = x2 - x1, y2 - y1

        if width <= 0 or height <= 0: return

        if CV_TRACKER_TYPE in ['NANOTRACK', 'DASIAMRPN']:
            shrink = 0.05
            dw, dh = width * shrink, height * shrink
            x1, y1 = int(x1 + dw), int(y1 + dh)
            width, height = int(width - 2 * dw), int(height - 2 * dh)
            x1, y1 = max(5, min(w_f - 10, x1)), max(5, min(h_f - 10, y1))
            width, height = max(10, min(w_f - x1 - 5, width)), max(10, min(h_f - y1 - 5, height))

        bbox_cv = (int(x1), int(y1), int(width), int(height))
        cv_tracker = create_cv_tracker()
        if not cv_tracker: return

        try:
            res = cv_tracker.init(frame, bbox_cv)
            if res is None or res:
                is_cv_tracking = True
                selected_obj_id = obj_id
                print(f"Started {CV_TRACKER_TYPE} tracking for ID: {selected_obj_id}")
            else:
                is_cv_tracking = False
        except Exception as e:
            print(f"Tracker init error: {e}")
            is_cv_tracking = False

def stop_cv_tracking():
    global cv_tracker, is_cv_tracking, selected_obj_id
    is_cv_tracking, cv_tracker, selected_obj_id = False, None, None
    print("Stopped CV tracking.")

def start_manual_cv_tracking(frame, bbox):
    global cv_tracker, is_cv_tracking, selected_obj_id
    
    cv_tracker = create_cv_tracker()
    if not cv_tracker: return

    try:
        res = cv_tracker.init(frame, tuple(bbox))
        if res is None or res:
            is_cv_tracking = True
            selected_obj_id = 'Manual'
            print(f"Started {CV_TRACKER_TYPE} tracking for Manual selection")
        else:
            is_cv_tracking = False
    except Exception as e:
        print(f"Tracker init error: {e}")
        is_cv_tracking = False

last_det_time = 0.0
last_track_time = 0.0
last_results = None
last_tracker_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        if not is_camera:
            print("Video reached the end. Looping from the beginning...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
            # Reset tracking dictionaries/states so they don't carry over from the end of the video
            object_dict.clear()
            for idx in range(len(object_slots)):
                object_slots[idx] = None
            cv_tracker = None
            is_cv_tracking = False
            selected_obj_id = None
            last_tracker_bbox = None
            last_known_boxes = []
        else:
            break

    now = time.time()

    # --- 1. Global Tracking Request Handling (Mouse) ---
    if tracking_request:
        if tracking_request['action'] == 'start':
            # YOLO ID Selection: Only ensure the internal tracker is ready
            tid = tracking_request['id']
            if tid in object_dict and object_dict[tid]['tracker'] is None:
                object_dict[tid]['tracker'] = create_cv_tracker()
                if object_dict[tid]['tracker']:
                    object_dict[tid]['tracker'].init(frame, object_dict[tid]['bbox'])
            # Ensure global tracker is OFF for YOLO objects to avoid double boxes
            is_cv_tracking = False
            cv_tracker = None
            last_tracker_bbox = None
        elif tracking_request['action'] == 'start_manual':
            start_manual_cv_tracking(frame, tracking_request['bbox'])
            last_tracker_bbox = None
        elif tracking_request['action'] == 'stop':
            stop_cv_tracking()
            last_tracker_bbox = None
        tracking_request = None

    # --- 2. Feature Tracking Loop ---
    t_track_start = time.time()
    tracking_performed = False

    # A. 優先更新選中物件 (每一幀都更新，不限頻率以保證流暢)
    if isinstance(selected_obj_id, int) and selected_obj_id in object_dict:
        info = object_dict[selected_obj_id]
        if info.get('tracker') is not None and info.get('visible', False):
            ok, bbox = info['tracker'].update(frame)
            if ok:
                info['bbox'] = smooth_bbox(info['bbox'], bbox) # 應用平滑化
                tracking_performed = True

    # B. 背景物件追蹤 (依據設定頻率執行)
    if track_interval == 0.0 or (now - last_track_time) >= track_interval:
        for obj_id, info in object_dict.items():
            if obj_id == selected_obj_id: continue # 已在上方處理
            if info.get('tracker') is not None and info.get('visible', False):
                ok, bbox = info['tracker'].update(frame)
                if ok:
                    info['bbox'] = smooth_bbox(info['bbox'], bbox) # 應用平滑化
                    tracking_performed = True
        last_track_time = now

    # C. 手動模式追蹤 (每一幀更新)
    if selected_obj_id == 'Manual' and is_cv_tracking and cv_tracker:
        ok, bbox = cv_tracker.update(frame)
        if ok:
            last_tracker_bbox = smooth_bbox(last_tracker_bbox, bbox) # 應用平滑化
            tracking_performed = True
        else:
            stop_cv_tracking()
            last_tracker_bbox = None

    if tracking_performed:
        track_calc_time = (time.time() - t_track_start) * 1000
        track_actual_fps = 1.0 / (now - last_track_exec_time) if (now - last_track_exec_time) > 0 else 0
        last_track_exec_time = now

    tracker_bbox = last_tracker_bbox

    # --- 3. YOLO Detection (Low Frequency Correction) ---
    if det_interval == 0.0 or (now - last_det_time) >= det_interval:
        t_det_start = time.time()
        results = model.track(frame, conf=args.conf, imgsz=args.imgsz, persist=True, tracker=TRACKER_CONFIG, classes=TARGET_CLASSES if TARGET_CLASSES else None, verbose=False)
        det_calc_time = (time.time() - t_det_start) * 1000
        det_actual_fps = 1.0 / (now - last_det_exec_time) if (now - last_det_exec_time) > 0 else 0
        last_det_exec_time = now
        last_results = results
        last_det_time = now

        boxes = results[0].boxes
        names = results[0].names if hasattr(results[0], 'names') else {}
        current_ids = set()
        last_known_boxes = boxes if boxes is not None and hasattr(boxes, 'id') else []

        if boxes is not None and hasattr(boxes, 'id'):
            for box in boxes:
                obj_id = int(box.id.item()) if box.id is not None else None
                if obj_id is not None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox_cv = (x1, y1, x2 - x1, y2 - y1)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                        cls_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                        obj_name = names[cls_id] if cls_id in names else f"ID{obj_id}"
                        
                        # Initialize or Re-sync Feature Tracker if BG tracking is enabled OR if it is the selected object
                        tracker_to_save = None
                        if use_bg_tracking or obj_id == selected_obj_id:
                            if obj_id not in object_dict or object_dict[obj_id].get('tracker') is None:
                                new_tracker = create_cv_tracker()
                                if new_tracker:
                                    new_tracker.init(frame, bbox_cv)
                                    tracker_to_save = new_tracker
                            else:
                                object_dict[obj_id]['tracker'].init(frame, bbox_cv)
                                tracker_to_save = object_dict[obj_id]['tracker']

                        if obj_id not in object_dict:
                            object_dict[obj_id] = {'crop': crop_resized, 'last_seen': now, 'first_seen': now, 'name': obj_name, 'visible': True, 'tracker': tracker_to_save, 'bbox': bbox_cv}
                        else:
                            object_dict[obj_id].update({'crop': crop_resized, 'last_seen': now, 'visible': True, 'tracker': tracker_to_save, 'bbox': bbox_cv})
                        current_ids.add(obj_id)

        for oid in object_dict:
            if oid not in current_ids: object_dict[oid]['visible'] = False

    # --- 4. Cleanup and Display Update ---
    remove_ids = [oid for oid, info in object_dict.items() if not info['visible'] and now - info['last_seen'] > REMOVE_DELAY]
    for oid in remove_ids:
        del object_dict[oid]
        remove_slot(oid)
    
    if selected_obj_id is not None and not is_cv_tracking and isinstance(selected_obj_id, int):
        if selected_obj_id not in object_dict:
            selected_obj_id = None

    if now - last_panel_update > REFRESH_INTERVAL:
        slot_crops = [np.zeros((CROP_SIZE+24, CROP_SIZE, 3), dtype=np.uint8) for _ in range(MAX_OBJECTS)]
        for obj_id, info in object_dict.items():
            if now - info['first_seen'] >= APPEAR_THRESHOLD:
                slot_idx = assign_slot(obj_id)
                if slot_idx is not None:
                    label_img = np.zeros((24, CROP_SIZE, 3), dtype=np.uint8)
                    cv2.putText(label_img, f"ID:{obj_id}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                    crop_with_label = np.vstack([label_img, info['crop']])
                    color = (0, 255, 0) if info['visible'] else (0, 165, 255)
                    cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), color, 2)
                    if obj_id == selected_obj_id:
                        if is_cv_tracking: cv2.rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (255, 0, 0), 2)
                        else: draw_dashed_rectangle(crop_with_label, (0, 24), (CROP_SIZE-1, CROP_SIZE+23), (0, 255, 255), 2, 5)
                    slot_crops[slot_idx] = crop_with_label
        rows = [np.hstack(slot_crops[i:i+OBJECTS_PER_ROW]) for i in range(0, MAX_OBJECTS, OBJECTS_PER_ROW)]
        cv2.imshow(objects_window_name, np.vstack(rows))
        last_panel_update = now

    # Use single frame for UI
    annotated_frame = frame.copy()
    
    # Draw all tracked objects for smooth high-frequency visualization
    for obj_id, info in object_dict.items():
        if info.get('visible', False) and 'bbox' in info:
            x, y, w, h = map(int, info['bbox'])
            is_sel = (obj_id == selected_obj_id)
            color = (255, 0, 0) if is_sel else get_color(obj_id)
            thick = 4 if is_sel else 2 # 選中加粗
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thick)
            label = f"ID:{obj_id} {info['name']}"
            cv2.putText(annotated_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Create UI Panel
    frame_h, frame_w = annotated_frame.shape[:2]
    ui_panel = np.zeros((UI_HEIGHT, frame_w, 3), dtype=np.uint8)

    # Draw UI labels and buttons
    yolo_color = (0, 255, 0) if app_mode == 'YOLO' else (150, 150, 150)
    cv2.rectangle(ui_panel, (MODE_BTNS['YOLO'][0], MODE_BTNS['YOLO'][1]), (MODE_BTNS['YOLO'][2], MODE_BTNS['YOLO'][3]), yolo_color, -1 if app_mode == 'YOLO' else 2)
    cv2.putText(ui_panel, "YOLO", (MODE_BTNS['YOLO'][0] + 15, MODE_BTNS['YOLO'][1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if app_mode == 'YOLO' else yolo_color, 2)

    manual_color = (0, 165, 255) if app_mode == 'MANUAL' else (150, 150, 150)
    cv2.rectangle(ui_panel, (MODE_BTNS['MANUAL'][0], MODE_BTNS['MANUAL'][1]), (MODE_BTNS['MANUAL'][2], MODE_BTNS['MANUAL'][3]), manual_color, -1 if app_mode == 'MANUAL' else 2)
    cv2.putText(ui_panel, "Manual", (MODE_BTNS['MANUAL'][0] + 15, MODE_BTNS['MANUAL'][1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if app_mode == 'MANUAL' else manual_color, 2)

    # BG Tracking Toggle Button
    bg_track_color = (0, 255, 0) if use_bg_tracking else (0, 0, 255)
    cv2.rectangle(ui_panel, (BG_TRACK_TOGGLE_BTN[0], BG_TRACK_TOGGLE_BTN[1]), (BG_TRACK_TOGGLE_BTN[2], BG_TRACK_TOGGLE_BTN[3]), bg_track_color, 2)
    cv2.putText(ui_panel, f"BG Track: {'ON' if use_bg_tracking else 'OFF'}", (BG_TRACK_TOGGLE_BTN[0] + 10, BG_TRACK_TOGGLE_BTN[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bg_track_color, 1)

    cv2.putText(ui_panel, "Mode:", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(ui_panel, "Tracker:", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(ui_panel, "Performance:", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw Tracker Buttons
    for t_name, coords in TRACKER_BTNS.items():
        color = (0, 255, 255) if CV_TRACKER_TYPE == t_name else (150, 150, 150)
        cv2.rectangle(ui_panel, (coords[0], coords[1]), (coords[2], coords[3]), color, -1 if CV_TRACKER_TYPE == t_name else 1)
        short_name = t_name if t_name != 'DASIAMRPN' else 'DaSiam'
        short_name = short_name if short_name != 'NANOTRACK' else 'NANO'
        cv2.putText(ui_panel, short_name, (coords[0] + 5, coords[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if CV_TRACKER_TYPE == t_name else color, 1)

    # Draw Performance Info
    perf_y = 85
    det_info = f"Det: {det_calc_time:.1f}ms ({det_actual_fps:.1f}Hz)"
    trk_info = f"Trk: {track_calc_time:.1f}ms ({track_actual_fps:.1f}Hz)"
    cv2.putText(ui_panel, det_info, (130, perf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(ui_panel, trk_info, (330, perf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    final_frame = np.vstack((ui_panel, annotated_frame))

    # --- 5. Draw manual drawing preview rectangle ---
    if is_drawing and drawing_start and drawing_current:
        # Note: mouse coordinates are already relative to frame in callback (after UI_HEIGHT subtraction)
        # So we add UI_HEIGHT back for drawing on final_frame
        p1 = (drawing_start[0], drawing_start[1] + UI_HEIGHT)
        p2 = (drawing_current[0], drawing_current[1] + UI_HEIGHT)
        cv2.rectangle(final_frame, p1, p2, (0, 255, 255), 2) # Yellow preview box

    # Selection Highlight (Legacy/Manual Selection)
    if tracker_bbox is not None:
        x, y, w, h = map(int, tracker_bbox)
        p1 = (x, y + UI_HEIGHT)
        p2 = (x + w, y + UI_HEIGHT)
        cv2.rectangle(final_frame, p1, (x + w, y + h + UI_HEIGHT), (255, 0, 0), 4) # 藍/紅加粗
        cv2.putText(final_frame, f"SELECT: {selected_obj_id}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow(main_window_name, final_frame)
    
    # Calculate wait time to match video source FPS
    if not is_camera:
        elapsed_time = time.time() - now
        target_delay = 1.0 / source_fps
        wait_time_ms = int((target_delay - elapsed_time) * 1000)
        if wait_time_ms < 1:
            wait_time_ms = 1
    else:
        wait_time_ms = 1

    if cv2.waitKey(wait_time_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Experiment finished.")
