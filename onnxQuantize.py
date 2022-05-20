# import onnx
# from onnxruntime.quantization import quantize_dynamic, QuantType

# model_fp32 = './ckpt/transweather.onnx'
# model_quant = './ckpt/transweather.quant.onnx'
# quantized_model = quantize_dynamic(model_fp32, model_quant)

# model = onnx.load("./ckpt/transweather.quant.onnx")
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))


import onnx
from onnxruntime.quantization import quantize, QuantizationMode

model = onnx.load('./ckpt/transweather.onnx')
quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
onnx.save(quantized_model, './ckpt/transweather.quant.onnx')

model = onnx.load("./ckpt/transweather.quant.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
