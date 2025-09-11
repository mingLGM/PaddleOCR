import os
import cv2
import numpy as np
import json
import argparse

def letterbox_image(img, intermediate_size=640, target_size=(1280, 1280), color=(114, 114, 114)):
    """
    将图像先按比例缩放到指定中间尺寸，再填充到目标尺寸。
    
    Args:
        img: 输入图像 (numpy array, BGR格式).
        intermediate_size: 中间缩放尺寸（长边目标值）.
        target_size: 最终目标尺寸 (宽, 高).
        color: 填充颜色 (B, G, R).
        
    Returns:
        padded_img: 处理后的图像.
        ratio: 缩放比例 (w_ratio, h_ratio).
        pad: 填充量 (top, bottom, left, right).
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # 1. 计算缩放比例，将长边缩放到 intermediate_size
    scale = intermediate_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 2. 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. 计算填充量，使缩放后的图像在目标画布中居中
    dw = target_w - new_w
    dh = target_h - new_h
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    
    # 4. 创建目标画布并填充
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    
    return padded_img, (scale, scale), (top, bottom, left, right)

def update_annotation_points(points, ratio, pad):
    """
    更新标注点的坐标以匹配letterbox处理后的图像。
    
    Args:
        points: 原始标注点列表 [[x1, y1], [x2, y2], ...].
        ratio: 缩放比例 (w_ratio, h_ratio).
        pad: 填充量 (top, bottom, left, right).
        
    Returns:
        new_points: 更新后的点坐标.
    """
    top, bottom, left, right = pad
    w_ratio, h_ratio = ratio
    
    new_points = []
    for point in points:
        # 应用缩放
        x = point[0] * w_ratio
        y = point[1] * h_ratio
        # 应用填充偏移
        x += left
        y += top
        new_points.append([int(round(x)), int(round(y))])
    
    return new_points

def process_dataset(image_dir, label_file_path, output_image_dir, output_label_path, target_size=(1280, 1280)):
    """
    处理整个数据集：调整图像大小并更新标注。
    
    Args:
        image_dir: 原始图像目录.
        label_file_path: 原始标注文件路径.
        output_image_dir: 输出图像目录.
        output_label_path: 输出标注文件路径.
        target_size: 目标尺寸 (宽, 高).
    """
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, 'images'), exist_ok=True)
    
    # 读取原始标注
    with open(label_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
            
        img_path = parts[0]
        annotation_str = parts[1]
        
        # 构建完整的图像路径
        full_img_path = os.path.join(image_dir, img_path)
        # print(full_img_path)
        
        # 检查图像文件是否存在
        if not os.path.exists(full_img_path):
            print(f"警告：图像文件 {full_img_path} 不存在，跳过")
            continue
        
        # 读取图像
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"警告：无法读取图像 {full_img_path}，跳过")
            continue
        
        # 应用letterbox
        # processed_img, ratio, pad = letterbox_image(img, target_size)
        processed_img, ratio, pad = letterbox_image(img, intermediate_size=640, target_size=target_size)
        
        # 保存处理后的图像
        output_img_path = os.path.join(output_image_dir, img_path)
        # print(output_img_path)
        cv2.imwrite(output_img_path, processed_img)
        
        # 解析并更新标注
        try:
            annotations = json.loads(annotation_str)
            updated_annotations = []
            
            for ann in annotations:
                if 'points' in ann:
                    updated_points = update_annotation_points(ann['points'], ratio, pad)
                    updated_ann = ann.copy()
                    updated_ann['points'] = updated_points
                    updated_annotations.append(updated_ann)
                else:
                    updated_annotations.append(ann)
            
            # 构建新的标注行
            new_line = f"{img_path}\t{json.dumps(updated_annotations, ensure_ascii=False)}\n"
            new_lines.append(new_line)
            
        except json.JSONDecodeError:
            print(f"警告：无法解析标注行 {annotation_str}，跳过")
            continue
    
    # 写入更新后的标注文件
    with open(output_label_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"处理完成！共处理 {len(new_lines)} 个样本。")
    print(f"处理后的图像保存在: {output_image_dir}")
    print(f"更新后的标注文件: {output_label_path}")

if __name__ == "__main__":
    ####使用LetterBox处理数据集图像和标注
    
    image_dir        = r"/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/det"          # 原始图像目录
    label_file       = r"/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/det/det_label.txt"      # 原始标注文件
    output_image_dir = r"/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/det_1280"     # 输出图像目录
    output_label_file= r"/home/hc/work/lzm/datasets/gangban/paddleocr_dataset/det_1280/det_label_1280.txt" # 输出标注文件
    target_size      = (1280, 1280)            # 目标尺寸 (宽, 高)
    
    process_dataset(image_dir, label_file, output_image_dir, output_label_file, target_size)