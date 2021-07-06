import argparse
import os

from PIL import Image


def yolov5_to_voc(label_root, image_root, class_list, xml_root):
    """
    修改自：
    https://blog.csdn.net/qq_38109843/article/details/90783347
    """
    for file in os.listdir(label_root):
        img_path = os.path.join(image_root, file.replace(".txt", ".jpg"))
        img_file = Image.open(img_path)
        txt_file = open(os.path.join(label_root, file)).read().splitlines()
        xml_file = open(os.path.join(xml_root, file.replace(".txt", ".xml")), "w")
        width, height = img_file.size
        xml_file.write("<annotation>\n")
        xml_file.write("\t<folder>simple</folder>\n")
        xml_file.write("\t<filename>" + str(file) + "</filename>\n")
        xml_file.write("\t<size>\n")
        xml_file.write("\t\t<width>" + str(width) + " </width>\n")
        xml_file.write("\t\t<height>" + str(height) + "</height>\n")
        xml_file.write("\t\t<depth>" + str(3) + "</depth>\n")
        xml_file.write("\t</size>\n")

        for line in txt_file:
            print(line)
            line_split = line.strip().split()
            x_center = float(line_split[1])
            y_center = float(line_split[2])
            w = float(line_split[3])
            h = float(line_split[4])
            xmax = int((2 * x_center * width + w * width) / 2)
            xmin = int((2 * x_center * width - w * width) / 2)
            ymax = int((2 * y_center * height + h * height) / 2)
            ymin = int((2 * y_center * height - h * height) / 2)

            xml_file.write("\t<object>\n")
            xml_file.write("\t\t<name>" + class_list[int(line_split[0])] + "</name>\n")
            xml_file.write("\t\t<pose>Unspecified</pose>\n")
            xml_file.write("\t\t<truncated>0</truncated>\n")
            xml_file.write("\t\t<difficult>0</difficult>\n")
            xml_file.write("\t\t<bndbox>\n")
            xml_file.write("\t\t\t<xmin>" + str(xmin) + "</xmin>\n")
            xml_file.write("\t\t\t<ymin>" + str(ymin) + "</ymin>\n")
            xml_file.write("\t\t\t<xmax>" + str(xmax) + "</xmax>\n")
            xml_file.write("\t\t\t<ymax>" + str(ymax) + "</ymax>\n")
            xml_file.write("\t\t</bndbox>\n")
            xml_file.write("\t</object>\n")
        xml_file.write("</annotation>")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--class-list", type=str, required=True)
    parser.add_argument("--xml-root", type=str, required=True)
    args = parser.parse_args()

    assert os.path.isdir(args.label_root)
    assert os.path.isdir(args.image_root)
    assert os.path.isfile(args.class_list)
    if not os.path.exists(args.xml_root):
        os.makedirs(args.xml_root)
    return args


def load_list_from_txt(path):
    with open(path, encoding="utf-8", mode="r") as f:
        class_list = f.read().strip()
    return class_list


if __name__ == "__main__":
    args = get_args()
    yolov5_to_voc(
        label_root=args.label_root,
        image_root=args.image_root,
        class_list=load_list_from_txt(args.class_list),
        xml_root=args.xml_root,
    )
