import argparse
import json
import os

import cv2
from tqdm import tqdm


def yolov5_to_coco(image_root, label_root, class_list, json_path):
    """
    修改自：
    https://github.com/Weifeng-Chen/DL_tools/blob/main/yolo2coco.py
    """
    image_names = sorted(os.listdir(image_root))

    dataset_info = {"categories": [], "annotations": [], "images": []}

    for i, class_name in enumerate(class_list, 0):
        dataset_info["categories"].append(
            {"id": i, "name": class_name, "supercategory": "mark"}
        )

    # 标注的id
    ann_id_cnt = 0
    for k, name in enumerate(tqdm(image_names)):
        # 支持 png jpg 格式的图片。
        label_path = os.path.join(
            label_root, name.replace(".jpg", ".txt").replace(".png", ".txt")
        )
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(image_root, name))
        height, width, _ = im.shape
        # 添加图像的信息
        dataset_info["images"].append(
            {"file_name": name, "id": k, "width": width, "height": height}
        )
        if not os.path.exists(label_path):
            # 如没标签，跳过，只保留图片信息。
            continue

        with open(label_path, "r") as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset_info["annotations"].append(
                    {
                        "area": width * height,
                        "bbox": [x1, y1, width, height],
                        "category_id": cls_id,
                        "id": ann_id_cnt,
                        "image_id": k,
                        "iscrowd": 0,
                        # mask, 矩形是从左上角点按顺时针的四个顶点
                        "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                    }
                )
                ann_id_cnt += 1

    # 保存结果
    with open(json_path, encoding="utf-8", mode="w") as f:
        json.dump(dataset_info, f, indent=2)
    print("Save annotation to {}".format(json_path))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--class-list", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.label_root)
    assert os.path.isdir(args.image_root)
    assert os.path.isfile(args.class_list)
    if not os.path.exists(os.path.dirname(args.json_path)):
        os.makedirs(os.path.dirname(args.json_path))
    return args


def load_list_from_txt(path):
    with open(path, encoding="utf-8", mode="r") as f:
        class_list = f.read().strip()
    return class_list


if __name__ == "__main__":
    args = get_args()
    yolov5_to_coco(
        image_root=args.image_root,
        label_root=args.label_root,
        class_list=load_list_from_txt(args.class_list),
        json_path=args.json_path,
    )
