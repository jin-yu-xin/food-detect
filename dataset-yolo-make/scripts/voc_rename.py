# 如果此时已经标注好VOC数据集，并且将.jpg和.xml文件分别放置在了两个文件夹下
# Note:是分别放置在两个文件下，与labeled文件夹将.jpg和.xml放置在同一文件夹下不同
# 可以自行利用一下函数重名.jpg和.xml文件
# 重命名格式 000000.jpg or .xml

import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import argparse
import yaml
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='config/custom.yaml')
    args = parser.parse_args()
    if not os.path.exists(args.conf):
        raise FileNotFoundError('%s "%s" not found' % (chr(128561),args.conf))
    return args


ARGS = parse_args()
CONFIG_FILE: str = ARGS.conf

configs=None
with open(CONFIG_FILE,'r') as f:
    configs=yaml.full_load(f)
    if 'dataset' not in configs:
        raise KeyError("%s key 'dataset' is not defined in '%s'" % (chr(128561),CONFIG_FILE))
    DATASET = configs['dataset']
    print(chr(128640), 'Process dataset: %s' % DATASET, os.linesep)


# 获取文件夹中图片的数量
def getDirImageNum(img_dir):
    bmpDirImagesNum = 0
    for bmpfile in os.listdir(img_dir):
        if os.path.splitext(bmpfile)[1] == '.jpg':
            bmpDirImagesNum += 1
    return bmpDirImagesNum


# 获取文件夹中xml文件的数量
def getDirXmlNum(xml_dir):
    xmlDirXmlNum = 0
    for xmlfile in os.listdir(xml_dir):
        if os.path.splitext(xmlfile)[1] == '.xml':
            xmlDirXmlNum += 1
    return xmlDirXmlNum


# 创建VOC文件目录
def make_voc_dir(voc_dir):
    voc_img_path = voc_dir + "/JPEGImage"
    voc_xml_path = voc_dir + "/Annotations"
    if not os.path.exists(voc_dir):
        os.makedirs(voc_img_path)
        os.makedirs(voc_xml_path)
    else:
        print('directory exists!')
    return voc_img_path, voc_xml_path


def voc_rename(img_dir, xml_dir, voc_dir):
    # 创建保存更名后文件的文件夹
    output_img_path, output_xml_path = make_voc_dir(voc_dir=voc_dir)
    # 获取.jpg和.xml文件数
    image_num = getDirImageNum(img_dir)
    xml_num = getDirXmlNum(xml_dir)
    print('图片数: %d  标签数: %d' % (image_num, xml_num))
    if image_num != xml_num:
        print('待更名的图片数和标签数不相等！请检查！')
    # 获取.xml文件列表
    xml_file_list = os.listdir(xml_dir)
    # 可排序,避免乱序
    xml_file_list.sort()
    # 获取输出文件夹中已有的.xml文件数量,方便之后按编号顺序添加更名后的文件
    j = getDirXmlNum(output_xml_path)
    error = []
    for i, xml_file in enumerate(xml_file_list):
        print(i + 1)
        src_img_path = os.path.join(img_dir, xml_file.split('.')[0] + '.jpg')
        src_xml_path = os.path.join(xml_dir, xml_file)
        if os.path.exists(src_img_path) and os.path.exists(src_xml_path):
            j = j + 1
            # 更改.jpg文件名 如000000.jpg
            new_jpg_name = '0' + format(str(j), '0>5s') + '.jpg'
            new_jpg_filename = os.path.join(os.path.abspath(output_img_path), new_jpg_name)
            os.rename(src_img_path, new_jpg_filename)
            # 更改.xml文件名 如000000.xml
            new_xml_name = '0' + format(str(j), '0>5s') + '.xml'
            new_xml_filename = os.path.join(os.path.abspath(output_xml_path), new_xml_name)
            # 更改.xml文件中标签对filename和path的内容
            try:
                tree = ET.parse(src_xml_path)
                root = tree.getroot()
                filename = root.find('filename')
                filename.text = new_jpg_name
                path = root.find('path')
                path.text = os.path.abspath(output_img_path) + '\\' + new_jpg_name
                tree.write(src_xml_path)
                os.rename(src_xml_path, new_xml_filename)
                print('converting %s to %s ...' % (src_xml_path, new_xml_filename))
            except:
                error.append(new_jpg_name)
                continue
        else:
            print('.jpg an .xml are not exist!')
    # 如果有出错的文件，error++
    print(len(error))


IMAGE_DIR = os.path.join(DATASET, 'image')
ANNOTATION_DIR = os.path.join(DATASET, 'anno')
VOC_DIR = './VOC'
voc_rename(IMAGE_DIR, ANNOTATION_DIR, VOC_DIR)
