from ultralytics.utils.benchmarks import benchmark
import pandas as pd

results = []
for model in (
    "yolo11n.pt",
    "yolo11m.pt",
    "yolo12n.pt",
    "yolo12m.pt",
):
    for data in ("coco8.yaml",):
        for format in (
            "torchscript",
            "onnx",
            "openvino",
            "saved_model",
            "tflite",
            # "engine",  # GPU only,
        ):
            df = benchmark(model=model, data=data, imgsz=640, format=format)
            df["Model"] = model
            # df["Data"] = data
            results.append(df)

print(pd.concat(results))
