
import numpy as np
import argparse
import cv2
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import logging
# from net.rexnetv1 import ReXNetV1
from net.resnet18_ACGP import Model
from torchsummary import summary
width=224

#检测硬件环境，自动选择CPU或GPU
if torch.cuda.is_available():
    device=torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    #   获取GPU相关参数
    p = torch.cuda.get_device_properties(0)
    pp = '{},GPU:{},total_memory:{}G'.format(torch.__version__, p.name, p.total_memory / 1024 ** 3)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(pp)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.FloatTensor')


def face_dec(frame,net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return  detections,h, w

def get_face_region(i,frame,detections,w,h):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)
    face = frame[startY:endY, startX:endX]
    return face,(startX,startY),(endX,endY)

def live_dec(model,face):
    image = cv2.resize(face, (width, width)).astype(np.float32)
    # image = torch.from_numpy(image/255.0).permute(2, 0, 1)
    ####
    cv2.imshow('target', image / 255.0)
    image -= (104.0, 117.0, 123.0)  # BRG
    image /= 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    ####
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        model.eval()
        predict = model(Variable(image.to(device)))
        live_score, result = torch.max(F.softmax(predict, dim=1), dim=1)
        live_score = live_score.data.cpu().numpy()[0]
        result = result.data.cpu().numpy()[0]
    return  live_score,result

def draw_rectangle(frame,result,live_score,startX, startY,endX, endY):
    if result == 0:
        cv2.putText(frame, 'fake_{}'.format(str(live_score)), (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (200, 0, 200), 2)
    else:
        cv2.putText(frame, 'real_{}'.format(str(live_score)), (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (200, 200, 0), 2)

def pre_precessing(args):
    print("人脸检测器载入中...")
    protoPath = os.path.sep.join([args.detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([args.detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    time.sleep(3)

    print('活体检测模型载入中...')
    # model = ReXNetV1(classes=2, width_mult=1.7, depth_mult=1.6).to(device)
    model = Model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    summary(model, (3,width, width))
    time.sleep(2)

    print("视频流读取中...")
    camera = cv2.VideoCapture(0)
    assert camera.isOpened(), '摄像头打不开'
    time.sleep(1)

    # 设定视频窗口大小
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 输出视频流相关参数
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS) % 100
    # logging.info('w:{}  h:{}  fps:{}'.format(w, h, fps))


    # 是否设定录制视频
    out=None
    if args.video:
        print('开启视频录制...')
        now = time.strftime("%m-%d %H-%M-%S", time.localtime())
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('{}.avi'.format(now), fourcc, 15, (1280, 720))
    return net,model,camera,out

def main(args):
    net, model, camera, out=pre_precessing(args)
    #模型正向推理
    print('正向推理进行中...')
    while True:
        re, frame = camera.read()
        if not re:
            print('无法获取视频帧!!!')
            exit()
        frame2=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
        cv2.imshow("HSV frame", cv2.resize(frame2,(544,306)))
        # frame3=frame.astype(np.float32)-np.array((104, 117, 123), dtype=np.float32)
        frame3 = frame - (104.0, 117.0, 123.0)
        cv2.imshow("Precessed frame", cv2.resize(frame3, (544, 306)))
        st = time.time()
        detections, h, w=face_dec(frame,net)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.face_confidence:
                face, (startX, startY), (endX, endY)=get_face_region(i,frame,detections,w,h)
                if len(face) != int(0):
                    live_score, result=live_dec(model,face)
                    if live_score > args.live_confidence:
                        draw_rectangle(frame,result,live_score,startX, startY,endX, endY)
        en = time.time()
        cv2.putText(frame, '{:.1f}fps'.format(1 / (en - st)), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if args.video:
            out.write(frame)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    return camera,out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default=r'face_detector',help='人脸检测器存放文件夹')
    parser.add_argument("--face_confidence", type=float, default=0.8,help='人脸检测器阈值')
    parser.add_argument('--model_path', default=r'model/resACGP_0.25_face.pt',help='活体检测模型路径')
    parser.add_argument("--live_confidence", type=float, default=0.75,help='活体检测阈值')
    parser.add_argument("--video",  default=False,help='视频录制')
    arguments = parser.parse_args()

    camera,out=main(arguments)
    camera.release()
    out.release()
    cv2.destroyAllWindows()

