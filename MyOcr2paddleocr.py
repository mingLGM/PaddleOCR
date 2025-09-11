# OCR抠图并生成PaddleOCR训练标签
from pathlib import Path
import cv2 as cv
import os, json
import PIL.Image as Image
import numpy as np
import re
import shutil
import random

def translate_points(dx, dy, points):
    points[0::2] += dx
    points[1::2] += dy

def roate_points(rad, points):
    xs = points[0::2]
    ys = points[1::2]
    _xs = xs * np.cos(rad) - ys * np.sin(rad)
    _ys = xs * np.sin(rad) + ys * np.cos(rad)
    points[0::2] = _xs
    points[1::2] = _ys
    return points

def get_cornerpoints(points, rad):
    """
    获取旋转矩形的四个角点
    """
    x1, y1, x2, y2 = points
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    translate_points(-cx, -cy, points)
    roate_points(-rad, points)
    
    x1, y1, x2, y2 = points
    width, height = abs(x2-x1), abs(y2-y1)
    
    conners = np.array([-width/2, -height/2, width / 2, -height / 2, width/2, height/2, -width/2, height/2])
    roate_points(rad, conners)
    translate_points(cx, cy, conners)
    
    return conners, (cx, cy), width, height

def crop_rotated_rectangle(image, points):
    """
    从图像中抠取旋转矩形区域
    :param image: 原始图像
    :param points: 旋转矩形的四个角点，格式为np.array([x1,y1, x2,y2, x3,y3, x4,y4])
    :return: 抠取后的矩形图像
    """
    # 将一维点数组转换为二维点数组 (4x2)
    points = points.reshape(4, 2).astype(np.float32)
    
    # 计算旋转矩形的宽度和高度
    width = int(np.linalg.norm(points[0] - points[1]))
    height = int(np.linalg.norm(points[1] - points[2]))
    
    # 如果计算出的宽度或高度为0，则使用替代方法
    if width == 0 or height == 0:
        rect = cv.minAreaRect(points)
        width, height = int(rect[1][0]), int(rect[1][1])
    
    # 定义目标矩形的四个点（正矩形）
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # 确保点按顺时针或逆时针顺序排列（凸包）
    hull = cv.convexHull(points.reshape(-1, 1, 2))
    hull_points = hull.reshape(4, 2).astype(np.float32)
    
    # 计算透视变换矩阵
    M = cv.getPerspectiveTransform(hull_points, dst_points)
    
    # 应用透视变换
    cropped = cv.warpPerspective(image, M, (width, height))
    
    return cropped

def draw_points(img, points, color):
    nLen = len(points)
    for i in range(0, nLen, 2):
        x = int(points[i])
        y = int(points[i + 1])
        cv.drawMarker(img, (x, y), color, cv.MARKER_CROSS, 20, 2) 
    return img

def trans_points_to_cropImage(center, rad, points, w, h):
    """
    将点转换到抠图坐标系下
    """
    cx, cy = center
    translate_points(-cx, -cy, points)
    roate_points(-rad, points)
    translate_points(w/2, h/2, points)
    return points

def get_new_number_filename(fileName:str, idx: int):
    idxStr = str(idx)
    for i in range(5):
        if i >= len(idxStr):
            idxStr = f'0{idxStr}'
    idxStr = f'{idxStr}_'
    fileName = re.sub("^\d[\d]{4} *", idxStr, fileName)
    return fileName

def rename_filename(filename):
    """将文件名中的空格替换为下划线"""
    return filename.replace(" ", "_")

def convert_to_paddleocr_format(annotations, image_path):
    """
    将标注转换为PaddleOCR格式
    """
    paddle_annotations = []
    for ann in annotations:
        # 获取旋转矩形的四个角点
        points = ann['points']
        rad = ann['angle']
        corners, _, _, _ = get_cornerpoints(np.asarray(points), rad)
        
        # 转换为PaddleOCR需要的点格式
        points_list = corners.reshape(4, 2).astype(int).tolist()
        
        # 创建PaddleOCR标注对象
        paddle_ann = {
            "transcription": ann['transcription'],
            "points": points_list
        }
        paddle_annotations.append(paddle_ann)
    
    # 创建PaddleOCR格式的行
    return f"{image_path}\t{json.dumps(paddle_annotations)}\n"


def split_dataset_labels(label_path, split_ratio=0.9, random_seed=42):
    """
    划分数据集标签为训练集和验证集
    
    参数:
    label_path: 标签文件路径
    split_ratio: 训练集比例，默认0.9 (9:1)
    random_seed: 随机种子，确保结果可重现
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 处理检测标签文件
    print("处理标签文件...")
    with open(label_path, 'r', encoding='utf-8') as f:
        label_lines = f.readlines()
    
    # 打乱检测标签顺序[2,3](@ref)
    random.shuffle(label_lines)
    
    # 划分检测数据集
    split_idx = int(len(label_lines) * split_ratio)
    label_train = label_lines[:split_idx]
    label_val = label_lines[split_idx:]
    
    # 保存划分后的检测标签
    label_train_path = label_path.replace('.txt', '_train.txt')
    label_val_path = label_path.replace('.txt', '_val.txt')
    
    with open(label_train_path, 'w', encoding='utf-8') as f:
        f.writelines(label_train)
    with open(label_val_path, 'w', encoding='utf-8') as f:
        f.writelines(label_val)
    
    print(f"训练集: {len(label_train)} 条, 保存于: {label_train_path}")
    print(f"验证集: {len(label_val)} 条, 保存于: {label_val_path}")
    print(f"划分比例: {split_ratio*100}% 训练, {(1-split_ratio)*100}% 验证")

def doOcrCropImages(prjPath:str):
    prjPath = Path(prjPath)
    if not prjPath.exists() or not os.path.join(prjPath, "inputImages") or not os.path.join(prjPath, "labelInfo"):
        print(f"项目路径不存在: {prjPath}")
        return
    
    # 创建新的PaddleOCR数据集目录结构
    paddleocr_datasets_path = os.path.join(prjPath, "paddleocr_dataset")
    os.makedirs(paddleocr_datasets_path, exist_ok=True)
    
    # 创建det目录结构
    det_path = os.path.join(paddleocr_datasets_path, "det")
    det_images_path = os.path.join(det_path, "images")
    os.makedirs(det_images_path, exist_ok=True)
    
    # 创建rec目录结构
    rec_path = os.path.join(paddleocr_datasets_path, "rec")
    rec_images_path = os.path.join(rec_path, "images")
    os.makedirs(rec_images_path, exist_ok=True)
    
    # 创建标签文件
    det_label_path = os.path.join(det_path, "det_label.txt")
    rec_label_path = os.path.join(rec_path, "rec_label.txt")
    det_label_file = open(det_label_path, "w", encoding="utf-8")
    rec_label_file = open(rec_label_path, "w", encoding="utf-8")

    labelInfoRoot = os.path.join(prjPath, "labelInfo")
    inputImagesRoot = os.path.join(prjPath, "inputImages")
    infoList = os.listdir(labelInfoRoot)

    nLen = len(infoList)
    for i, file in enumerate(infoList, 1):
        filePath = os.path.join(labelInfoRoot, file)
        with open(filePath, 'r', encoding='utf8') as f:
            content = json.load(f)
            if 'images' not in content:
                continue

            img_info = content['images'][0]
            img_file_name = img_info['file_name']
            
            original_img_path = os.path.join(inputImagesRoot, img_file_name)
            
            # 复制原始图像到det/images目录
            det_sanitized_name = rename_filename(img_file_name)
            det_image_path = os.path.join(det_images_path, det_sanitized_name)
            shutil.copy2(original_img_path, det_image_path)
            det_relative_path = os.path.join("images", det_sanitized_name)
            
            # 生成PaddleOCR检测标签
            if 'annotations' in content:
                # 使用det目录中的图像路径
                det_label_line = convert_to_paddleocr_format(content['annotations'], det_relative_path)
                det_label_file.write(det_label_line)

            # 获取抠图
            img = Image.open(original_img_path)
            mat = np.asarray(img) 
            if len(mat.shape) == 3:
                mat = cv.cvtColor(mat, cv.COLOR_RGB2BGR)
            
            _content = {
                "annotations": content.get('annotations', []),
                "images": content['images']
            }

            if 'annotations' in content:
                for k, obj in enumerate(list(content['annotations']), 0):

                    # 获取原始图像上的标注坐标
                    dividepoints = obj['dividepoints']
                    points = obj['points']
                    rad = obj['angle']

                    # 获取四个角点
                    corners, center, w, h = get_cornerpoints(np.asarray(points), rad)

                    cropped = crop_rotated_rectangle(mat, corners)

                    # 将点转换到抠图坐标上
                    _dividepoints = trans_points_to_cropImage(center, rad, np.asarray(dividepoints), w, h)
                    _points = trans_points_to_cropImage(center, rad, np.asarray(points), w, h)
                    _corners = trans_points_to_cropImage(center, rad, corners, w, h)
 
                    obj['angle'] = 0
                    obj['area'] = w * h
                    obj['dividepoints'] = list(_dividepoints)
                    obj['points'] = list(_points)
                    _content[k] = obj
                
                # 生成新的文件名（带序号）
                img_file_name = get_new_number_filename(img_file_name, i - 1)
                
                # 保存裁剪后的图像到rec/images目录
                rec_sanitized_name = rename_filename(img_file_name)
                cropped_img_path = os.path.join(rec_images_path, rec_sanitized_name)
                cv.imwrite(cropped_img_path, cropped)
                rec_relative_path = os.path.join("images", rec_sanitized_name)
                
                # 生成PaddleOCR识别标签
                transcription = obj.get('transcription', '')
                if transcription:
                    # 使用rec目录中的图像路径
                    rec_label_line = f"{rec_relative_path}\t{transcription}\n"
                    rec_label_file.write(rec_label_line)

            print(f'{file} be processed finished  {i}/{nLen} ')
    
    # 关闭标签文件
    det_label_file.close()
    rec_label_file.close()
    
    split_dataset_labels(det_label_path, split_ratio=0.9)
    split_dataset_labels(rec_label_path, split_ratio=0.9)
    
    print(f"PaddleOCR数据集已生成在: {paddleocr_datasets_path}")
    print(f"检测数据位置: {det_path}")
    print(f"识别数据位置: {rec_path}")




if __name__ == '__main__':
    # doOcrCropImages(r'/home/hc/work/lzm/datasets/gangban/')
    
    det_label_path = r'/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/det_1280/det_label_1280.txt'
    split_dataset_labels(det_label_path, split_ratio=0.9)