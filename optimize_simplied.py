import onnx
import onnxoptimizer
import onnxruntime as ort


model = onnx.load("./ckpt/transweather.onnx")
model_opt = onnxoptimizer.optimize(model)
onnx.save(model_opt, "./ckpt/transweather_opt.onnx")
print("[INFO] finished.")
