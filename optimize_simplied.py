# import onnx
# import onnxoptimizer
# import onnxruntime as ort
# model = onnx.load("./ckpt/transweather_sim.onnx")
# model_opt = onnxoptimizer.optimize(model)
# onnx.save(model_opt, "./ckpt/transweather.onnx")
# print("[INFO] finished.")

import onnx
from onnxsim import simplify
import onnxoptimizer
onnx_model = onnx.load("./ckpt/transweather.onnx")
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
new_model = onnxoptimizer.optimize(model_simp)
onnx.save(new_model, "./ckpt/transweather_simp_opt.onnx")
