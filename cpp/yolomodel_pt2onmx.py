import argparse
from ultralytics import YOLO

def main():
    # 建立一個參數解析器
    parser = argparse.ArgumentParser(description="Convert a YOLO .pt model to .onnx format.")
    # 新增 --input 參數，設為必填
    parser.add_argument('--input', type=str, required=True, help='Input .pt model file name (e.g., yolov8n.pt)')
    args = parser.parse_args()

    print(f"Loading model from: {args.input}")
    model = YOLO(args.input)
    
    print("Exporting model to ONNX format...")
    # Add opset=12 for better compatibility with older OpenCV versions (like 4.5.4)
    # This can help resolve errors related to Reshape operations in the model's head.
    model.export(format='onnx', dynamic=False, simplify=True, opset=12)
    print(f"Export complete. The ONNX model is saved as {args.input.replace('.pt', '.onnx')}")

if __name__ == "__main__":
    main()