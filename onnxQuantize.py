""" Dynamic Quantization """
# import onnx
# from onnxruntime.quantization import quantize, QuantizationMode

# model = onnx.load('./ckpt/transweather.onnx')
# quantized_model = quantize(model, quantization_mode=QuantizationMode.IntegerOps)
# onnx.save(quantized_model, './ckpt/transweather.quant.onnx')

# model = onnx.load("./ckpt/transweather.quant.onnx")
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))


""" Static Quantization """
import os
import numpy as np
import cv2
import onnx 
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


def preprocess_image(image_path):
    # Resizing image in the multiple of 16"
    input_img = cv2.imread(image_path)
    input_img = cv2.resize(input_img, (640, 360))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    wd_new, ht_new, _ = input_img.shape
    if ht_new>wd_new and ht_new>2048:
        wd_new = int(np.ceil(wd_new*2048/ht_new))
        ht_new = 2048
    elif ht_new<=wd_new and wd_new>2048:
        ht_new = int(np.ceil(ht_new*2048/wd_new))
        wd_new = 2048
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    input_img = cv2.resize(input_img, (ht_new, wd_new), interpolation=cv2.INTER_AREA)

    input_img = input_img.astype(np.float32) / 255.
    input_img = np.expand_dims(input_img, axis=0)
    return input_img


def preprocess_func(images_folder, size_limit=0):
    image_names = os.listdir(images_folder)
    print("[INFO] num of images: ", len(image_names))
    print("[INFO] image name sample: ", image_names[0])
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        # image_filepath = images_folder + '/' + image_name
        image_filepath = os.path.join(images_folder, image_name)
        image_data = preprocess_image(image_filepath)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class TransQuantDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, size_limit=10)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)




if __name__ == "__main__":
    calibration_image_folder = "/mnt/d/DATASET/DATA_2070/test/rain_L"
    dr = TransQuantDataReader(calibration_image_folder)

    quantize_static("./ckpt/transweather.onnx", "./ckpt/transweather.quant.onnx", dr)
    # quantize_static("./ckpt/transweather.onnx", "./ckpt/transweather.quant.onnx", dr, extra_options={'ActivationSymmetric ': True, 'WeightSymmetric': True})

    print('ONNX full precision model size (MB):', os.path.getsize("./ckpt/transweather.onnx")/(1024*1024))
    print('ONNX quantized model size (MB):', os.path.getsize("./ckpt/transweather.quant.onnx")/(1024*1024))
