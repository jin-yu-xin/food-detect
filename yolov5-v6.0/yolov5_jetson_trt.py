import numpy as np
import os
import sys
from pathlib import Path
import tkinter
from tkinter import ttk
from PIL import Image, ImageTk  # 图像控件
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
def LoadImage(cap, img_size=640, stride=32):
    assert cap.isOpened(), f'Failed to open camera!'  # 判断摄像头是否打开
    if cv2.waitKey(1) == ord('q'):  # q to quit
        cap.release()
        cv2.destroyAllWindows()
    # 读取一帧图片
    ret_val, img0 = cap.read()
    img0 = cv2.flip(img0, 1)  # flip left-right
    # Padded resize
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    return img, img0


# 推理不更新梯度
def infer(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
          source=0,  # file/dir/URL/glob, 0 for webcam
          imgsz=[640],  # inference size (pixels)
          conf_thres=0.25,  # confidence threshold
          iou_thres=0.45,  # NMS IOU threshold
          max_det=1000,  # maximum detections per image
          device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
          view_img=False,  # show results 是否展示预测之后的图片或视频
          classes=None,  # filter by class: --class 0, or --class 0 2 3
          agnostic_nms=False,  # class-agnostic NMS 进行nms是否也去除不同类别之间的框,默认False
          augment=False,  # augmented inference
          visualize=False,  # visualize features
          line_thickness=3,  # bounding box thickness (pixels) 边界框的线条粗细 默认3个像素点
          half=False,  # use FP16 half-precision inference
          dnn=False,  # use OpenCV DNN for ONNX inference
          ):
    # 将输入的待推理文件路径变为字符串
    source = str(source)
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand [640,640]
    # .isnumeric()是否是由数字组成
    is_webcam = source.isnumeric()

    # Load model 载入模型
    # 获取设备 CUDA/CPU
    device = select_device(device)
    # 根据权重文件类型检测推理使用的深度学习框架 PyTorch/TorchScript/TensorFlow/CoreML/ONNX Runtime/ONNX OpenCV DNN
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # 确保输入图片大小能整除stride==64,如果不能整除则调整为可整除的图像尺寸
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    # 如果不用CPU而是PyTorch on CUDA并且half==True 那么可以使用16位半精度推理，推理速度更快
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # 使用摄像头
    if is_webcam:
        view_img = check_imshow()  # Check if environment supports image displays
        pipe = eval(source)
        p = 'Camera'
        cap = cv2.VideoCapture(pipe)  # 打开摄像头
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cudnn.benchmark = True  # set True to speed up constant image size inference

    # 用户交互界面
    root_win = tkinter.Tk()  # 创建主窗口
    root_win.title(string="烤箱环境下的食材检测系统")
    root_win.geometry('960x600')  # 设置窗口大小
    # 设置主窗口的背景颜色,颜色值可以是英文单词，或者颜色值的16进制数,除此之外还可以使用tkinter内置的颜色常量
    # root_win["background"] = ""
    # 创建画布来实时显示检测结果
    canvas = tkinter.Canvas(root_win, bg='white', width=w, height=h)
    canvas.place(x=10, y=10)  # 画布放置位置
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

    # Run inference 运行推理
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    count = 0
    while True:
        items = tree.get_children()
        for item in items:
            tree.delete(item)
        dt = [0.0, 0.0, 0.0]  # 记录各部分运行时间
        # for path, im, im0s, s in dataset:
        s = f'Camera:'
        # 载入数据
        im, im0 = LoadImage(cap=cap, img_size=imgsz, stride=stride)
        t1 = time_sync()
        # 转化到GPU上
        im = torch.from_numpy(im).to(device)
        # 是否使用半精度
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # 像素值归一化
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim 增加一个batch维度
        t2 = time_sync()
        dt[0] += t2 - t1  # pre-process time

        # Inference
        # 执行forward()前向推理
        """
        pred.shape == (1, num_boxes, 5+num_classes)
        h和w为输入图片的长和宽，分别在stride=8,16,32三个尺度上进行多尺度预测
        num_boxes == h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[...,0:4]为预测边界框坐标(x,y,w,h)
        pred[...,4]为objectness置信度
        pred[...,5:-1]为类别概率
        """
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        # inference time
        dt[1] += t3 - t2

        # ############################################# NMS 非极大值抑制###############################################
        """
        pred:网络输出结果 边界框+置信度+类别索引
        conf_thres:置信度阙值 默认为0.25
        iou_thres:iou阙值
        classes:是否只保留特定类别
        agnostic_nms:进行nms是否也去除不同类别之间的框
        max_det:保留的最多边界框数量

        Return：list of detections, on (num_boxes,6) tensor per image [xyxy, conf, cls]
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为xyxy+conf+cls 其中：
        num_boxes 经过nms处理后的最终得到的预测边界框数量 例如bus.jpg中的预测边界框数量为5
        tensor[0:4]:x1,y1,x2,y2
        tensor[4]:conf
        tensor[5]:cls表示预测边界框类别的索引,例如bus.jpg中的预测到的类别为bus的边界框cls为5
        """
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3  # nms-process time

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # ############################# 5.Process predictions 对预测结果进行显示打印保存####################################
        # det.shape == torch.Size([num_boxes,6])  pred是一个列表 num_boxes:一张图片中预测边界框的数量
        for i, det in enumerate(pred):  # per image
            count += 1
            # 输入源来自于LoadVebcam
            img0 = im0.copy()
            s += f'{count}: '

            # 将输入图片的wxh加入打印信息中
            s += '%gx%g ' % im.shape[2:]  # print string
            # 设置在原图中标注预测结果的边界框线条粗细，字体大小，字体类别等信息
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图上
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                # 打印预测到的类别数量
                # det[:, -1]为类别概率 .unique()会去除一维数组或列表中的重复元素，并按照由小到大的顺序返回一个新的数组或列表
                number = []
                detected_class = []
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    number.append(n)
                    detected_class.append(c)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # reversed()返回一个反转的迭代器
                for *xyxy, conf, cls in reversed(det):  # per box
                    if view_img:  # Add bbox to image
                        # 在原图上画出边界框,打上标签和置信度
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            # im0为将预测结果标注在原图上的结果图
            img0 = annotator.result()
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGBA)
            pilImage = Image.fromarray(img)  # 将array转化成Image
            tkImage = ImageTk.PhotoImage(image=pilImage)  # 一个与tkinter兼容的照片图像
            canvas.create_image(0, 0, anchor='nw', image=tkImage)
            for j in range(min(len(detected_class), len(number))):
                tree.insert('', j, values=(names[int(detected_class[j])], int(number[j].item())))
            root_win.update()
            root_win.after(1)
            # # 显示图片
            # if view_img:
            #     cv2.imshow(str(p), img0)  # str(p) 显示窗口的名字 如果是摄像头则为0
            #     cv2.waitKey(1)  # 1 millisecond
    root_win.mainloop()


if __name__ == "__main__":
    check_requirements(exclude=('tensorboard', 'thop'))
    infer()

