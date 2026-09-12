# YOLO 物件偵測與追蹤展示 (YOLO Object Detection & Tracking Demo)

本專案提供基於 **Python** 與 **C++** 的高性能即時物件偵測與追蹤系統。透過整合 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 的精準偵測與多種 OpenCV 單目標追蹤器（Single-object trackers），實現了強大的目標鎖定與流暢的動態表現。

## 核心功能特色 (Python 版本)

*   **混合偵測與追蹤系統 (Hybrid System)**:
    *   **低頻偵測，高頻追蹤**：允許 YOLO 以較低頻率運行（節省 CPU/GPU 資源），同時使用 OpenCV 特徵追蹤器以相機的全速 FPS 填補間隔。
    *   **BG Track 開關**：設有專用的背景追蹤切換開關。開啟 (**ON**) 時，所有偵測到的物件都會平滑追蹤；關閉 (**OFF**) 時，背景物件僅隨 YOLO 偵測更新，但**選中目標 (Selected Target)** 仍保持高頻平滑追蹤。
*   **視覺穩定性與回饋**:
    *   **EMA 平滑化 (Exponential Moving Average)**：引入座標平滑化演算法，大幅減少偵測校正瞬間產生的視覺跳動，讓框線移動極致平滑。
    *   **選中強化**：選中的物件（透過點擊或手動劃框）會以**加粗邊框 (Thickness=4)** 標示，並優先進行每一幀的追蹤更新。
*   **整合控制面板 (Control Panel)**:
    *   **模型即時聯動**：切換追蹤器類型（例如從 NanoTrack 換成 CSRT）時，會立即以新演算法重新初始化所有運作中的追蹤器。
    *   **實時效能統計**：實時顯示偵測 (Det) 與追蹤 (Trk) 循環的耗時 (ms) 與實際執行頻率 (Hz)。
*   **雙視窗監控**:
    *   **主視窗 (Main)**：顯示即時影像流、平滑的邊界框以及 UI 控制項。
    *   **物件視窗 (Objects)**：以網格形式顯示所有偵測目標的即時裁切預覽 (Crop)，方便辨識與快速選取。

## 目錄結構

```plaintext
├── python/
│   ├── objdet.py              # 主實作程式 (偵測 + 追蹤 + UI)
│   ├── tracking_example.py    # 簡化的純 OpenCV 追蹤範例
│   ├── bytetrack.yaml         # YOLO 內部追蹤配置
│   └── *.onnx / *.pt          # 追蹤器與 YOLO 的權重檔案
│
└── cpp/
    ├── main.cpp               # C++ 偵測 Demo (OpenCV DNN)
    ├── yolomodel_pt2onmx.py   # 模型轉換工具 (.pt -> .onnx)
    └── yolov8n.onnx           # C++ 版本使用的 ONNX 模型
```

## Python 實作使用指南

### 環境需求

*   Python 3.10+
*   `pip install ultralytics opencv-contrib-python numpy`

### 快速啟動

1.  **基本執行**:
    ```bash
    cd python
    python objdet.py
    ```
    *若未指定參數，系統會列出所有可用的 UVC 相機模式供互動式選取。*

2.  **混合效能模式 (推薦測試)**:
    ```bash
    # 以 1Hz (低耗能) 進行偵測，同時以 30Hz (高流暢度) 進行追蹤
    python objdet.py --det-freq 1 --track-freq 30 --conf 0.5
    ```

### CLI 參數說明

| 參數 | 說明 | 範例 |
| :--- | :--- | :--- |
| `--source` | 相機索引或影片路徑 | `0` 或 `video.mp4` |
| `--source_cfg` | 指定硬體解析度與 FPS | `1280x720@30` |
| `--model` | YOLO 模型檔案路徑 | `yolov8n.pt` |
| `--conf` | 偵測信心門檻 (0~1) | `--conf 0.4` |
| `--det-freq` | 限制 YOLO 偵測頻率 (Hz) | `--det-freq 2.0` |
| `--track-freq` | 限制 OpenCV 追蹤頻率 (Hz) | `--track-freq 30.0` |

### 介面操作指南

*   **Integrated/BG Track 按鈕**: 切換是否要追蹤背景物件，或僅追蹤選中目標。
*   **YOLO Mode**: 在主畫面點擊物件框，或在 `Objects` 視窗點擊裁切預覽即可鎖定目標。
*   **Manual Mode**: **按住左鍵拖曳**可劃定自定義區域。拖曳時會顯示黃色預覽框。
*   **Engine 選取**: 點擊 `NANO`, `DaSiam`, `CSRT`, `KCF` 或 `MIL` 可立即為所有物件切換追蹤演算法。
*   **鍵盤快捷鍵**:
    *   `q`: 退出程式。

## 重要注意事項

*   **平滑係數**: 物件框的跟隨反應速度可透過修改 `objdet.py` 內的 `SMOOTH_ALPHA` 進行調整（預設為 0.2）。
*   **Wayland 支援**: 在使用 Wayland 的 Linux 環境（如 Ubuntu/Gnome）下，程式會自動設定 `QT_QPA_PLATFORM=xcb` 以確保視窗字體與渲染正常。

## 授權聲明

本專案採用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 授權。
請尊重第三方庫與模型（如 Ultralytics YOLO, OpenCV）的個別授權條款。
