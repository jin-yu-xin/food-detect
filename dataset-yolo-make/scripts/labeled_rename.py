# 如果拿到该工程时已经按类别标注好了数据集，用labeled_rename.py来重命名.jpg和.xml
# 重命名方式：类别名-序列号 如potato-00001.jpg or.xml

import os
import xml.etree.ElementTree as ET
import argparse
import yaml
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='config/custom.yaml')
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    if not os.path.exists(args.conf):
        raise FileNotFoundError('"%s" not found' % args.conf)
    return args


ARGS = parse_args()
CONFIG_FILE: str = ARGS.conf

configs=None
with open(CONFIG_FILE, 'r') as f:
    configs = yaml.full_load(f)
    if 'dataset' not in configs:
        raise KeyError("%s key 'dataset' is not defined in '%s'" % (chr(128561), CONFIG_FILE))
    DATASET = configs['dataset']
    print(chr(128640), 'Process dataset: %s' % DATASET, os.linesep)

LABELED_DIR = os.path.join(DATASET, 'labeled')  # 获取labeled文件夹路径

for class_name in os.listdir(LABELED_DIR):  # 访问labeled文件夹下的所有类别文件夹
    if not os.path.isdir(os.path.join(LABELED_DIR, class_name)):
        continue
    print()
    print(' -- Rename Class: %s' % class_name)
    class_dir = os.path.join(LABELED_DIR, class_name)

    pbar = tqdm.tqdm(os.listdir(class_dir))  # os.listdir()返回指定的文件夹包含的文件或文件夹的名字的列表
    n_jpg_rename = 0
    n_xml_rename = 0
    for file in pbar:
        src_file_path = os.path.join(class_dir, file)
        fename = os.path.splitext(src_file_path)[1]  # 获取文件扩展名
        if fename == '.jpg':
            new_file_name = '%s-%05d.jpg' % (class_name, n_jpg_rename)
            n_jpg_rename += 1
        elif fename == '.xml':
            new_file_name = '%s-%05d.xml' % (class_name, n_xml_rename)
            # 修改.xml文件中标签对filename之间的值
            # 可有可无，因为后续的voc转yolo没有用到标签对filename和path之间的内容
            tree = ET.parse(src_file_path)
            root = tree.getroot()
            filename = root.find('filename')
            new_filename = '%s-%05d.jpg' % (class_name, n_xml_rename)
            filename.text = new_filename
            path = root.find('path')
            path.text = os.path.abspath(class_dir) + '\\' + new_filename
            tree.write(src_file_path)
            n_xml_rename += 1
        else:
            print('Error file format!')
        dst_file_path = os.path.join(class_dir, new_file_name)
        os.rename(src_file_path, dst_file_path)
        pbar.set_description('[%s] %s' % (class_name, file))


