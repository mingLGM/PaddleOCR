from pathlib import Path
import cv2 as cv
import os, json
import PIL.Image as Image
import numpy as np
import re
import shutil
import random
import math

def translate_points(dx, dy, points):
    """平移点集"""
    points[0::2] += dx
    points[1::2] += dy

def rotate_points(rad, points):
    """旋转点集（修复版）"""
    xs = points[0::2].copy()
    ys = points[1::2].copy()
    _xs = xs * np.cos(rad) - ys * np.sin(rad)
    _ys = xs * np.sin(rad) + ys * np.cos(rad)
    points[0::2] = _xs
    points[1::2] = _ys
    return points

def get_cornerpoints(points, rad):
    """
    获取旋转矩形的四个角点（修复版）
    """
    x1, y1, x2, y2 = points
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # 创建点的副本以避免修改原始数据
    points_copy = np.array([x1, y1, x2, y2], dtype=np.float64)
    
    # 平移点到中心
    translate_points(-cx, -cy, points_copy)
    
    # 旋转点
    rotate_points(-rad, points_copy)
    
    x1, y1, x2, y2 = points_copy
    width, height = abs(x2 - x1), abs(y2 - y1)
    
    # 计算四个角点
    conners = np.array([
        -width/2, -height/2, 
        width/2, -height/2, 
        width/2, height/2, 
        -width/2, height/2
    ], dtype=np.float64)
    
    # 旋转回原方向并平移回原位置
    rotate_points(rad, conners)
    translate_points(cx, cy, conners)
    
    return conners, (cx, cy), width, height

def crop_rotated_rectangle(image, points):
    """
    从图像中抠取旋转矩形区域（修复版）
    """
    # 将一维点数组转换为二维点数组 (4x2)
    points = points.reshape(4, 2).astype(np.float32)
    
    # 计算旋转矩形的宽度和高度
    width = int(np.linalg.norm(points[0] - points[1]))
    height = int(np.linalg.norm(points[1] - points[2]))
    
    # 确保宽度和高度为正数
    width = max(1, width)
    height = max(1, height)
    
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
    """在图像上绘制点"""
    nLen = len(points)
    for i in range(0, nLen, 2):
        x = int(points[i])
        y = int(points[i + 1])
        cv.drawMarker(img, (x, y), color, cv.MARKER_CROSS, 20, 2) 
    return img

def trans_points_to_cropImage(center, rad, points, w, h):
    """
    将点转换到抠图坐标系下（修复版）
    """
    cx, cy = center
    points_copy = points.copy()
    
    # 平移点到中心
    translate_points(-cx, -cy, points_copy)
    
    # 旋转点
    rotate_points(-rad, points_copy)
    
    # 平移到裁剪图像中心
    translate_points(w/2, h/2, points_copy)
    
    # 确保坐标在图像范围内[7](@ref)
    points_copy[0::2] = np.clip(points_copy[0::2], 0, w - 1)
    points_copy[1::2] = np.clip(points_copy[1::2], 0, h - 1)
    
    return points_copy

def get_new_number_filename(fileName:str, idx: int):
    """生成带序号的新文件名"""
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
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 处理检测标签文件
    print("处理标签文件...")
    with open(label_path, 'r', encoding='utf-8') as f:
        label_lines = f.readlines()
    
    # 打乱检测标签顺序
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

def datasetConvert_HC2PaddleOCR(prjPath:str):
    """
    将 HC 格式的OCR标注数据集转换为 PaddleOCR 官方所需的两任务格式：
        1. 文本检测(det)任务：提供原图 + 检测标签(det_label.txt)
        2. 文本识别(rec)任务：提供按检测框裁剪后的单行文本图 + 识别标签(rec_label.txt)
    同时自动完成训练/验证集划分(9:1)。

    目录结构生成如下：
    prjPath/paddleocr_dataset/
    ├── det/
    │   ├── images/           # 检测任务原图
    │   └── det_label.txt     # 检测标签，每行：图像相对路径\t[{"points":[[x1,y1],...],"transcription":"text"},...]
    └── rec/
        ├── images/           # 识别任务裁剪图
        └── rec_label.txt     # 识别标签，每行：图像相对路径\t文本

    参数
    ----
    prjPath : str
        HC 项目根目录，其下必须存在
        - inputImages/   原图文件夹
        - labelInfo/     标注 json 文件夹（每张图对应一个 json, 内含 images/annotations 字段）
    """
    prjPath = Path(prjPath)
    if not prjPath.exists() or not os.path.exists(os.path.join(prjPath, "inputImages")) or not os.path.exists(os.path.join(prjPath, "labelInfo")):
        print(f"项目路径不存在或缺少必要目录: {prjPath}")
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
            
            # 检查图像文件是否存在
            if not os.path.exists(original_img_path):
                print(f"警告：图像文件 {original_img_path} 不存在，跳过")
                continue
            
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
                    try:
                        # 获取原始图像上的标注坐标
                        dividepoints = obj['dividepoints']
                        points = obj['points']
                        rad = obj['angle']

                        # 获取四个角点
                        corners, center, w, h = get_cornerpoints(np.asarray(points), rad)
                        
                        # 确保宽度和高度为正
                        w, h = max(1, int(w)), max(1, int(h))
                        
                        cropped = crop_rotated_rectangle(mat, corners)
                        
                        if cropped is None or cropped.size == 0:
                            print(f"警告: 无法裁剪图像 {img_file_name} 的标注 {k}")
                            continue
                        
                        # 将点转换到抠图坐标上
                        _dividepoints = trans_points_to_cropImage(center, rad, np.asarray(dividepoints), w, h)
                        _points = trans_points_to_cropImage(center, rad, np.asarray(points), w, h)
                        
                        obj['angle'] = 0
                        obj['area'] = w * h
                        obj['dividepoints'] = list(_dividepoints)
                        obj['points'] = list(_points)
                        _content['annotations'][k] = obj
                        
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
                            
                    except Exception as e:
                        print(f"处理图像 {img_file_name} 的标注 {k} 时出错: {str(e)}")
                        continue

            print(f'{file} be processed finished  {i}/{nLen} ')
    
    # 关闭标签文件
    det_label_file.close()
    rec_label_file.close()
    
    split_dataset_labels(det_label_path, split_ratio=0.9)
    split_dataset_labels(rec_label_path, split_ratio=0.9)
    
    print(f"PaddleOCR数据集已生成在: {paddleocr_datasets_path}")
    print(f"检测数据位置: {det_path}")
    print(f"识别数据位置: {rec_path}")


def create_inverted_images_and_labels(original_label_file_path, output_base_dir='./train_data/cls'):
    """
    根据原有的正向分类数据集，创建180度反转的图像和对应的标签文件。
    目录结构将符合PaddleOCR示例：
        images/0/ - 原始图像（0度）
        images/180/ - 反转图像（180度）
    
    Args:
        original_label_file_path (str): 原有正向数据标签文件的路径。
        output_base_dir (str): 新生成的数据集存放的根目录。
    """

    # 定义目录路径
    original_image_dir = os.path.dirname(original_label_file_path)  # 原始图像所在目录
    images_dir = os.path.join(output_base_dir, 'images')  # 图像存储目录
    images_0_dir = os.path.join(images_dir, '0')  # 0度图像目录
    images_180_dir = os.path.join(images_dir, '180')  # 180度图像目录
    inverted_label_file_path = os.path.join(output_base_dir, 'cls_label.txt')  # 最终合并后的标签文件

    # 创建输出目录
    os.makedirs(images_0_dir, exist_ok=True)
    os.makedirs(images_180_dir, exist_ok=True)
    os.makedirs(output_base_dir, exist_ok=True)

    # 读取原始标签文件
    original_lines = []
    with open(original_label_file_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    # 处理原始标签行并生成反转图像和标签
    new_label_lines = []  # 用于存储所有标签行（包括原始和新的）

    for line in original_lines:
        # 清理行尾的换行符并按制表符分割
        parts = line.strip().split('\t')
        if len(parts) < 2:
            print(f"Skipping invalid line: {line}")
            continue

        # 获取图像路径和标签
        original_image_rel_path = parts[0]  # 原始图片的相对路径
        label = parts[1]  # 标签

        # 构建原始图像的绝对路径
        original_image_path = os.path.join(original_image_dir, original_image_rel_path)

        # 检查原始图像是否存在
        if not os.path.exists(original_image_path):
            print(f"Original image not found: {original_image_path}, skipping.")
            continue

        # 获取图像文件名
        image_filename = os.path.basename(original_image_rel_path)
        
        # 复制原始图像到 images/0 目录
        dest_image_0_path = os.path.join(images_0_dir, image_filename)
        if not os.path.exists(dest_image_0_path):
            shutil.copy2(original_image_path, dest_image_0_path)
            print(f"Copied original image to: {dest_image_0_path}")
        
        # 为原始图像添加标签行（0度）
        new_image_0_rel_path = os.path.join('images', '0', image_filename)
        new_label_line_0 = f"{new_image_0_rel_path}\t0\n"
        new_label_lines.append(new_label_line_0)

        # 创建反转图像并保存到 images/180 目录
        name_part, ext_part = os.path.splitext(image_filename)
        inverted_image_filename = f"{name_part}_inv{ext_part}"
        inverted_image_path = os.path.join(images_180_dir, inverted_image_filename)

        try:
            with Image.open(original_image_path) as img:
                inverted_img = img.rotate(180)  # 旋转180度
                inverted_img.save(inverted_image_path)
                print(f"Created inverted image: {inverted_image_path}")
        except Exception as e:
            print(f"Failed to process image {original_image_path}: {e}")
            continue

        # 为反转图像添加标签行（180度）
        new_image_180_rel_path = os.path.join('images', '180', inverted_image_filename)
        new_label_line_180 = f"{new_image_180_rel_path}\t1\n"
        new_label_lines.append(new_label_line_180)

    # 将所有的标签行写入新的标签文件
    with open(inverted_label_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_label_lines)

    print(f"\nAll done!")
    print(f"Original images are saved in: {images_0_dir}")
    print(f"Inverted images are saved in: {images_180_dir}")
    print(f"Merged label file is saved as: {inverted_label_file_path}")
    print(f"Directory structure:")
    print(f"  {output_base_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── 0/      # Original images (0 degrees)")
    print(f"    │   └── 180/    # Inverted images (180 degrees)")
    print(f"    └── cls_label.txt  # Label file")



if __name__ == '__main__':
    # 使用示例
    datasetConvert_HC2PaddleOCR(r'/home/hc/work/lzm/datasets/gangban')
    
    # det_label_path = r'/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/cls/cls_label.txt'
    # create_inverted_images_and_labels(det_label_path, r'/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/cls/')
    # split_dataset_labels(det_label_path, split_ratio=0.9)