# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# 推理不更新梯度
@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results 是否展示预测之后的图片或视频
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS 进行nms是否也去除不同类别之间的框,默认False
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels) 边界框的线条粗细 默认3个像素点
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # ######################################### 1. 初始化配置 ###########################################################
    # 将输入的待推理文件路径变为字符串
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 获取文件后缀名，判断其是否为img和video格式
    # Path()提取文件名 例如：p = Path('./data/images/bus.jpg') p.name->bus.jpg p.parent->./data/images p.suffix->.jpg
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # .lower()转化为小写 .upper()转化为大写 .title()转化为首字母大写其余小写 .startswich()检查字符串是否是以指定子字符串开头，返回True/False
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # .isnumeric()是否是由数字组成
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories 创建存储检测结果的文件夹
    # 查看./runs/detect/exp是否存在，不存在就新建，存在就按照exp递增新建，如exp1,exp2...
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 如果参数save_txt为True 就创建./runs/detect/exp/labels 否则就创建./runs/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model 载入模型
    # 获取设备 CUDA/CPU
    device = select_device(device)
    # 根据权重文件类型检测推理使用的深度学习框架 PyTorch/TorchScript/TensorFlow/CoreML/ONNX Runtime/ONNX OpenCV DNN
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # 确保输入图片大小能整除stride==64,如果不能整除则调整为可整除的图像尺寸
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    # 如果不用CPU而是PyTorch on CUDA并且half==True 那么可以使用16位半精度推理，推理速度更快
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # #####################################################2.加载数据 ##################################################
    # Dataloader 载入待推理数据
    # 使用摄像头或者网页
    if webcam:
        view_img = check_imshow()  # Check if environment supports image displays
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # return self.sources, img, img0, None, ''
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        # return path, img, img0, self.cap, s
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ####################################################3.网络推理预测#################################################
    # Run inference 运行推理
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
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
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
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

        # ############################################# 4.NMS 非极大值抑制###############################################
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
            seen += 1
            if webcam:  # batch_size >= 1
                # 输入源来自于LoadStreams
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # 输入源来自LoadImages 读取本地文件中的图片或者视频，batch_size==1
                # p:当前图片或者视频的绝对路径
                # im0: 未经过letterbox(resize+pad成stride倍数)之前的原始图片
                # frame: 视频流的第几帧
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path  ./data/images/
            # 图片或者视频保存路径 如：./runs/detect/exp/bus.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            # 预测框坐标的.txt文件保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 将输入图片的wxh加入打印信息中
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 设置在原图中标注预测结果的边界框线条粗细，字体大小，字体类别等信息
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图上
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 打印预测到的类别数量
                # det[:, -1]为类别概率 .unique()会去除一维数组或列表中的重复元素，并按照由小到大的顺序返回一个新的数组或列表
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # reversed()返回一个反转的迭代器
                for *xyxy, conf, cls in reversed(det):  # per box
                    if save_txt:  # Write to file
                        # 将每个图片的预测信息分别存入./runs/detect/exp/labels文件夹下的xxx.txt文件中
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 在原图上画出边界框,打上标签和置信度
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            # 将预测出来的目标剪切出来，保存成图片.jpg 保存在./runs/detect/exp/crops/类别名/文件夹下
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            # im0为将预测结果标注在原图上的结果图
            im0 = annotator.result()
            # 显示图片
            if view_img:
                cv2.imshow(str(p), im0)  # str(p) 显示窗口的名字 如果是摄像头则为0
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video 保存处理后的视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream 保存摄像机拍摄的视频流处理结果
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # 打印预测全过程中各部分处理速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)  # 打印所有参数
    return opt


def main(opt):
    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    check_requirements(exclude=('tensorboard', 'thop'))
    # 按照参数执行run()函数
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
