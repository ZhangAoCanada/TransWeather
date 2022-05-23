import onnx
import onnxoptimizer
import onnxruntime as ort


model = onnx.load("./ckpt/transweather_sim.onnx")
model_opt = onnxoptimizer.optimize(model)
onnx.save(model_opt, "./ckpt/transweather.onnx")
print("[INFO] finished.")
