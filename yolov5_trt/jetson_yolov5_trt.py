"""
uses TensorRT's Python api to make inferences on Jetson Nano.
"""
import math
import ctypes
import os
import shutil
import random
import sys
import threading
import time

import tkinter
from tkinter import ttk
from PIL import Image, ImageTk  # 图像控件

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        """
        root:当前正在遍历的这个文件夹本身的地址
        dir:一个list,指该文件夹中所有的目录的名字（不包含子目录）
        files:一个list,指该文件中所有文件的名字（不包含子目录）
        """
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    # 包括若干batch,每个batch里包含batch_size张图片
    return ret


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
               'F0F8FF', 'FAEBD7', '00FFFF', '7FFFD4', 'F0FFFF', '8A2BE2', 'A52A2A', 'DEB887', 'FF7F50', '20B2AA',
               '6495ED', 'A9A9A9', 'E9967A', '483D8B', 'B0C4DE', '32CD32', 'FFF0F5', 'E6E6FA', '90EE90', 'FFFF00')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


# 定义画框函数
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img（BGR）,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2] 预测边界框坐标
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness 定义画框线的宽度
    color = color or [random.randint(0, 255) for _ in range(3)]  # 画框线颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))  # (x1,y1),(x2,y2)
    # 画预测边界框
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果要给框打标签 label为要打标签的名字，如"potato"
    if label:
        tf = max(tl - 1, 1)  # font thickness 字体的宽度
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # 根据标签的名字确定标签所在框的大小
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        """
        c1,c2:标签框的对角线坐标
        cv2.LINE_AA 线型
        """
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # 为预测边界框打标签
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # Logger()为Builder/ICudaEngine/Runtime对象提供logger
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers  分配主机和设备缓冲区
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings. 将设备缓冲区追加到设备绑定
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        for i, image_raw in enumerate(raw_image_generator):
            # image, image_raw, h, w
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            # print(">>>>>>>image_raw:", image_raw.shape)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        batch_input_image = np.ascontiguousarray(batch_input_image)
        # print(">>>>>>>batch_input_image:", batch_input_image.shape)

        # Copy input image to host buffer .ravel()多维数组变成一维数组
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference. 运行推理
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess 后处理
        batch_detected_classid = []  # 一个batch中，各图片检测到的种类
        batch_num_classid = []  # 一个batch中，各图片中各种类的数量
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i]
            )
            # 获取每张图片中 各种类目标的数量
            image_detected_classid = []  # 每张图片检测到的种类
            image_num_classid = []  # 每张图片中各种类的数量
            sorted_classid = np.unique(result_classid)
            for c in sorted_classid:
                n = (c == result_classid).sum()
                image_detected_classid.append(c)
                image_num_classid.append(n)

            batch_detected_classid.append(image_detected_classid)
            batch_num_classid.append(image_num_classid)

            # Draw rectangles and labels on the original image
            for j in range(len(result_boxes)):
                color = colors(result_classid[j])
                box = result_boxes[j]
                plot_one_box(
                    box,
                    batch_image_raw[i],
                    color=color,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[j])], result_scores[j]
                    ),
                )
        return batch_image_raw, batch_detected_classid, batch_num_classid, end - start

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def get_raw_image_camera(self,cap):
        """
        description:Ready data for infer
        """
        for _ in range(self.batch_size):
            _, raw_image = cap.read()
            yield raw_image

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        # 按照小的缩放比调整原始图像为符合输入网络的图像尺寸(input_w,input_h)
        # 不够的地方在图像上下两边（r_h > r_w）或者左右两边（r_w > r_h）填充相应宽度的边框
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        # 为image在上、下、左、右填充 ty1,ty2,tx1,tx2宽度的边框，边框的像素值为(128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        # 将图像类转化为数组类，数据类型float32
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        # np.ascontiguousarray()函数将一个内存不连续存储的数组转换为内存连续存储（地址连续）的数组，使得运行速度更快。
        # 行优先的顺序（Row-major Order)，即内存中同行的元素存在一起
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])  # [i*6001:(i+1)*6001]
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]  # 每张图输出1000个预测目标
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def bbox_ious(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-16):
        """
        description: compute the GIoU/DIoU/CIoU of two bounding boxes  box1 is 4, box2 is nx4
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        """
        np.clip(a, a_min, a_max, out=None)
        a: 输入的数组
        a_min: 限定的最小值 也可以是数组 如果为数组时 shape必须和a一样
        a_max: 限定的最大值 也可以是数组 shape和a一样
        out：剪裁后的数组存入的数组
        """
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        w1, h1 = b1_x2 - b1_x1 + 1, b1_y2 - b1_y1 + 1
        w2, h2 = b2_x2 - b2_x1 + 1, b2_y2 - b2_y1 + 1
        union_area = w1 * h1 + w2 * h2 - inter_area + eps

        iou = inter_area / union_area
        if GIoU or DIoU or CIoU:
            # 求最小外接矩形的宽和高
            cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1) + 1
            ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1) + 1
            if CIoU or DIoU:
                c2 = cw ** 2 + ch ** 2 + eps  # 最小外接矩形对角线的平方
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # 两框中心点距离的平方
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:
                    v = (4 / math.pi ** 2) * np.power(np.arctan(w2 / h2) - np.arctan(w1 / h1), 2)
                    alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:
                c_area = cw * ch + eps
                return iou - (c_area - union_area) / c_area  # GIoU
        else:
            return iou  # IoU

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs 按置信度降序排列
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            # 用DIoU_NMS
            #large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            large_overlap = self.bbox_ious(np.expand_dims(boxes[0, :4], 0), boxes[:, :4], DIoU=True) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


class inferThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, yolov5_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.image_path_batch = image_path_batch

    def run(self):  # 线程在创建后会直接运行run()函数
        batch_image_raw, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class warmUpThread(threading.Thread):
    def __init__(self, yolov5_wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper

    def run(self):
        batch_image_raw, _, _, use_time = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


def detect(cap, yolov5_wrapper):
    if (cap.isOpened() == False):   
        cap.open(0)  # 'test_radio.mp4'

    while cap.isOpened():
        # 清空结果显示列表
        items = tree.get_children()
        for item in items:
            tree.delete(item)

        # 载入一张图片
        raw_imgs = yolov5_wrapper.get_raw_image_camera(cap)
        # raw_img = cv2.flip(img0, 1)  # flip left-right

        # 预测推理 batch_image_raw（BGR）, batch_detected_classid, batch_num_classid, end - start
        images_result, detected_classid, num_classid, use_time = yolov5_wrapper.infer(raw_imgs)

        for i,img_result in enumerate(images_result):
            # 将检测后的图片显示在画布中
            img = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(img)  # 将array转化成Image
            tkImage = ImageTk.PhotoImage(image=pilImage)  # 一个与tkinter兼容的照片图像
            canvas.create_image(0, 0, anchor='nw', image=tkImage)  # 显示于画布
            # 将检测到的图片中的各种类名称和数量显示在表格中
            for j in range(min(len(detected_classid[i]), len(num_classid[i]))):
                tree.insert('', j, values=(categories[int(detected_classid[i][j])], int(num_classid[i][j])))

        # 显示检测时间
        text = f'{use_time * 1000:.5} ms'
        time = tkinter.Label(root_win, text=text)
        time.place(x=740, y=450)

        # 更新窗口
        root_win.update()
        root_win.after(1)


def stop_detect(cap):
    cap.release()
    cv2.destroyAllWindows()


def root_win_quit(yolov5_wrapper):
    root_win.destroy()
    yolov5_wrapper.destroy()


if __name__ == "__main__":
    # #############################################load custom plugin and engine########################################
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/best.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # ###########################################load food labels######################################################

    categories = ["baby cabbage", "bacon", "basa fish", "bass", "batata", "cabbage", "chicken", "chicken wing", "chips",
                  "clam", "corn", "crab", "drumette", "drumstick", "duck", "duck leg", "dumpling", "egg", "eggplant",
                  "fish head", "fried bread stick", "gass carp", "lamb chop", "lemon", "lettuces", "needle mushroom",
                  "oyster", "pettitoes", "pleurotus eryngii", "popcorn chicken", "pork ribs", "potato", "prawn",
                  "salmon", "saury", "scallop", "squid", "steak", "streaky pork", "twistbread"]

    # ###########################################YoLov5TRT instance#####################################################
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    
    print('batch size is', yolov5_wrapper.batch_size)
# ######################################################## warm up #################################################
    for i in range(10):
        # create a new thread to do warm_up
        thread1 = warmUpThread(yolov5_wrapper)
        thread1.start()  # 启动线程
        thread1.join()  # 等待至线程终止

    # ###################################################用户界面设计####################################################
    camera = cv2.VideoCapture(0)  # 'test_radio.mp4'  # 打开usb摄像头
    canvas_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    root_win = tkinter.Tk()  # 创建主窗口
    root_win.title(string="烤箱环境下的食材检测系统")
    root_win.geometry('960x600')  # 设置窗口大小
    # 设置主窗口的背景颜色,颜色值可以是英文单词，或者颜色值的16进制数,除此之外还可以使用tkinter内置的颜色常量
    # root_win["background"] = ""
    canvas = tkinter.Canvas(root_win, bg='white', width=canvas_w, height=canvas_h)  # 创建画布来实时显示检测结果
    canvas.place(x=10, y=10)  # 画布放置位置
    # 创建表格中文显示检测结果
    """
    Treeview 组件是 ttk 模块的组件之一
    它既可以作为树结构使用，也可以作为表格展示数据(tkinter 并没有表格控件)
    """
    columns = ("class", "number")
    # 设置表格高度，"headings"表示将tree用作表格
    tree = ttk.Treeview(root_win, height=18, show="headings", columns=columns)
    # 设置每一列的表头宽度，内容居中
    tree.column("class", width=100, anchor='center')
    tree.column("number", width=100, anchor='center')
    # 显示表头
    tree.heading('class', text="食材种类")
    tree.heading('number', text="数量")
    tree.place(x=700, y=10)
    # 显示时间
    label_time = tkinter.Label(root_win, text="时间 ：")
    label_time.place(x=700, y=450)
    # 按键设计
    """
    通过用户点击按钮的行为来执行回调函数，是 Button 控件的主要功用。
    首先自定义一个函数或者方法，然后将函数与按钮关联起来，最后，当用户按下这个按钮时，Tkinter 就会自动调用相关函数。

    """
    button_start = ttk.Button(root_win, text="开始检测", state='normal', command=lambda: detect(camera, yolov5_wrapper))
    button_start.place(x=180, y=520)
    button_exit = ttk.Button(root_win, text="停止检测", state='normal', command=lambda: stop_detect(camera))
    button_exit.place(x=450, y=520)
    button_quit = ttk.Button(root_win, text="关闭", state='normal', command=lambda: root_win_quit(yolov5_wrapper))
    button_quit.place(x=700, y=520)

    root_win.mainloop()




