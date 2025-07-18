from ultralytics.utils.benchmarks import benchmark
import pandas as pd

results = []
for format in (
    "torchscript",
    "onnx",
    "openvino",
    "saved_model",
    "tflite",
    # "engine",  # GPU only,
):
    results.append(
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format=format)
    )

print(pd.concat(results))
