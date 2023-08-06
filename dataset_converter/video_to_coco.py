import copy
import datetime
import json
import os
from itertools import groupby

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as pyccocomask

INFO = {
    "description": "YouTube-VOS",
    "url": "https://youtube-vos.org/home",
    "version": "1.0",
    "year": 2018,
    "contributor": "me",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "id": 1,
        "name": "Creative Commons Attribution 4.0 License",
    }
]

# 根据自己的需要添加种类
CATEGORIES = [{"id": 1, "name": "person", "supercategory": "object", }]


class NumpyEncoder(json.JSONEncoder):
    """
    Custom encoder for numpy data types
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """

    def default(self, obj):
        if isinstance(
                obj,
                (
                        np.int_,
                        np.intc,
                        np.intp,
                        np.int8,
                        np.int16,
                        np.int32,
                        np.int64,
                        np.uint8,
                        np.uint16,
                        np.uint32,
                        np.uint64,
                ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def binary_mask_to_rle(binary_mask):
    """
    convert binary_mask to rle format
    from: lib/python3.6/site-packages/pycococreatortools/pycococreatortools.py
    """
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def convert_instance_mask_to_coco_format(jpeg_root, anno_root):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "videos": [],
        "categories": CATEGORIES,
        "annotations": [],
    }
    video_info_template_dict = {
        "date_captured": "2019-04-11 00:55:41.903902",
        "coco_url": "",
        "license": 1,
        "flickr_url": "",
        # 仅有后面五项需要修改，前面的直接使用复制的值即可
        "id": None,
        "length": None,
        "height": None,
        "width": None,
        "file_names": [],
    }
    inst_id_in_all = 0
    anno_info_total = []

    video_name_list = sorted(os.listdir(jpeg_root))
    for video_id, video_name in enumerate(video_name_list, start=1):
        video_info = copy.deepcopy(video_info_template_dict)

        video_info["id"] = video_id

        video_path = os.path.join(jpeg_root, video_name)
        frame_name_list_per_video = sorted(os.listdir(video_path))
        video_info["length"] = len(frame_name_list_per_video)

        first_frame_path = os.path.join(video_path, frame_name_list_per_video[0])
        image = cv2.imread(filename=first_frame_path, flags=1)
        height, width, _ = image.shape
        video_info["height"], video_info["width"] = height, width

        # annotations的每一个字典表示特定实例(不同类别、目标)，
        # 每个字典中的segmentations表示该视频(video_id指定)中各帧的属于该实例的标注
        # 每帧上的标注使用RLE格式
        inst_anno_template_dict = {
            "height": height,
            "width": width,
            "length": 1,
            "category_id": 1,
            "video_id": video_id,
            "iscrowd": 1,
            # 只有后四项需要修改，其他项直接复制即可
            "id": None,  # 后面需要修改
            "segmentations": [],  # 后面使用需要赋值
            "bboxes": [],  # 需要赋值
            "areas": [],  # 需要赋值
        }

        inst_id_from_video_first_frame = inst_id_in_all
        # 对于一个新的视频，实际上应该先遍历这个视频确定包含的总的实例数量，和一些其他的信息，
        # 不然不好应对目标中间消失或者更复杂的情况
        inst_set_curr_video = [0]
        for image_name in frame_name_list_per_video:
            if not image_name.endswith(".jpg"):
                continue
            video_info["file_names"].append(video_name + "/" + image_name)
            anno_name = image_name[:-4] + ".png"
            anno_video_path = os.path.join(anno_root, video_name, anno_name)
            anno_inst_array = np.array(Image.open(anno_video_path).convert("P"))
            anno_inst_list = np.unique(anno_inst_array)
            inst_set_curr_video = set(anno_inst_list).union(set(inst_set_curr_video))

        inst_list_curr_video = sorted(list(inst_set_curr_video))
        print(
            f" ==>> processing {video_id}/{len(video_name_list)}: {video_name}, and it has instances {inst_list_curr_video}"
        )
        for inst_idx_curr_video in inst_list_curr_video[1:]:
            new_inst_info = copy.deepcopy(inst_anno_template_dict)
            new_inst_info["id"] = inst_id_from_video_first_frame + inst_idx_curr_video
            anno_info_total.append(new_inst_info)
            inst_id_in_all += 1

        # 所有的实例字典已经创建完毕，接下来只需要更新内部的segmentations, bboxes, areas
        for image_id, image_name in enumerate(frame_name_list_per_video):
            if not image_name.endswith(".jpg"):
                continue
            anno_name = image_name[:-4] + ".png"
            anno_video_path = os.path.join(anno_root, video_name, anno_name)
            anno_inst_array = np.array(Image.open(anno_video_path).convert("P")).astype(np.uint8)
            anno_inst_list = sorted(np.unique(anno_inst_array))

            # 这里直接默认都是None，对于其他实例，我们直接对其更新即可
            for inst_idx_curr_video in inst_list_curr_video[1:]:
                curr_inst_id = inst_id_from_video_first_frame + inst_idx_curr_video
                anno_info_total[curr_inst_id - 1]["segmentations"].append(None)
                anno_info_total[curr_inst_id - 1]["bboxes"].append(None)
                anno_info_total[curr_inst_id - 1]["areas"].append(None)

            if len(anno_inst_list) > 1:
                # 此时包含前景实例
                for idx in anno_inst_list[1:]:  # 排除背景的实例编号
                    # 遍历从注释中找到的实例
                    inst_binary_mask = np.where(anno_inst_array == idx, 255, 0).astype(np.uint8)
                    encoded_inst_binary_mask = pyccocomask.encode(np.asfortranarray(inst_binary_mask))
                    area_for_inst_in_anno = pyccocomask.area(encoded_inst_binary_mask)
                    assert area_for_inst_in_anno >= 1
                    bbox_for_inst_in_anno = pyccocomask.toBbox(encoded_inst_binary_mask)
                    seg_for_inst_in_anno = binary_mask_to_rle(inst_binary_mask)
                    curr_inst_id = inst_id_from_video_first_frame + idx
                    anno_info_total[curr_inst_id - 1]["segmentations"][image_id] = seg_for_inst_in_anno
                    anno_info_total[curr_inst_id - 1]["bboxes"][image_id] = bbox_for_inst_in_anno
                    anno_info_total[curr_inst_id - 1]["areas"][image_id] = area_for_inst_in_anno

        coco_output["videos"].append(video_info)
    coco_output["annotations"] = anno_info_total

    with open(f"{os.path.dirname(jpeg_root)}/data.json", encoding="utf-8", mode="w") as output_json_file:
        json.dump(coco_output, output_json_file, cls=NumpyEncoder)


def main():
    video_image_root = "JPEGImages"
    video_anno_root = "Annotations"
    convert_instance_mask_to_coco_format(jpeg_root=video_image_root, anno_root=video_anno_root)


if __name__ == "__main__":
    main()
