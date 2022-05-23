from plistlib import InvalidFileException
import cv2
import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort

from torchvision.transforms import Compose, ToTensor, Normalize

from skimage import img_as_ubyte


def preprocessImage(input_img):
    # Resizing image in the multiple of 16"
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
    # input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
    # input_img = cv2.resize(input_img, (wd_new, ht_new), interpolation=cv2.INTER_AREA)
    input_img = cv2.resize(input_img, (ht_new, wd_new), interpolation=cv2.INTER_AREA)
    # input_img = input_img.transpose((2, 0, 1))

    # input_img = input_img.astype(np.float32) / 255.0
    # input_img = (input_img - 0.5) / 0.5

    input_img = input_img.astype(np.float32) / 255.
    # input_img = (input_img - 0.5*255) / (0.5 * 255)

    input_img = np.expand_dims(input_img, axis=0)
    return input_img


model = onnx.load("./ckpt/transweather.onnx")
onnx.checker.check_model(model)

video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
cap = cv2.VideoCapture(video_path)

# ort_session = ort.InferenceSession("./ckpt/transweather.onnx")
# ort_session = ort.InferenceSession("./ckpt/transweather_quant.onnx")
ort_session = ort.InferenceSession("./ckpt/transweather.quant.onnx")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (960, 540))
    frame = cv2.resize(frame, (640, 360))
    input_img = preprocessImage(frame)
    outputs = ort_session.run(
        None,
        {"input": input_img},
    )
    pred = outputs[0][0]
    # pred = pred.transpose((1, 2, 0))
    print("[INFO--raw pred] ", pred.shape)
    # pred = pred * 255.0
    # pred = pred.astype(np.uint8)
    pred = img_as_ubyte(pred)
    pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
    print("[INFO] ", pred.shape)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    img_show = np.hstack((frame, pred))
    # img_show = cv2.resize(img_show, None, fx=0.5, fy=0.5)
    cv2.imshow("pred", img_show)
    if cv2.waitKey(1) == 27:
        break
