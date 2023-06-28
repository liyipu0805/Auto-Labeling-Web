import os
from flask import Flask, render_template, request, url_for, redirect, send_from_directory, jsonify, send_file
print(os.getcwd())
from PIL import Image
from tools.ImageLabel import ImageAutoLabeling
import numpy as np
from ultralytics import YOLO
import zipfile
import io
from xml.etree import ElementTree as ET
coco_cls = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt") 

CURRENT = os.getcwd()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['LABEL_FOLDER'] = 'xmlLabel'
app.config['LABEL_INFO'] = {}


# LABEL_STATUS = 0

# IAL = ImageAutoLabeling("sam_vit_b_01ec64.pth", "vit_b", "cuda")

def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path) # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])
if not os.path.exists(app.config['LABEL_FOLDER']):
    os.makedirs(app.config['LABEL_FOLDER'])

del_files(app.config['UPLOAD_FOLDER'])
del_files(app.config['OUTPUT_FOLDER'])
del_files(app.config['LABEL_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    global LABEL_STATUS
    if request.method == 'POST':
        # 获取上传的文件

        # LABEL_STATUS = 0
        files = request.files.getlist('file')
        print(files)
        # 创建用于保存文件的目录（如果不存在）
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        

        # 处理上传的文件
        for file in files:
            # 保存文件到服务器
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return redirect(url_for('index'))

    # 获取已上传的图片文件列表
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            images.append(url_for('uploaded_file', filename=filename))
    boxes = []
    label_image = []
    for filename in images:
        if filename.endswith(('.jpg', '.jpeg', '.png')) and len(filename) > 0:
            # print(os.path.join(os.getcwd(), filename[1:]).replace("\\", "/"))
            '''
            img, masks = IAL.run(os.path.join(os.getcwd(), filename[1:]).replace("\\", "/"))
            img = np.uint8(img[: ,: ,:3] * 255)
            img = Image.fromarray(img)
            
            # 生成标注文件
            masks = IAL.write_masks_to_folder(masks)
            # result = cv2.addWeighted(image, 1, img, 0.5, 0)'''
            img_path = os.path.join(os.getcwd(), filename[1:]).replace("\\", "/")
            results = model.predict(img_path, imgsz=640, conf=0.3)
            annotated_frame = results[0].plot()
            W, H = annotated_frame.shape[1], annotated_frame.shape[0]
            box = results[0].boxes
            box = ToDict(box)
            boxes.append(box)

            # 生成标注信息
            
            img_label_info = generate_label(results[0].boxes, img_path, W, H)
            xml_info = create_xml(img_label_info)
            tree = ET.ElementTree(xml_info)
            
            

            # 保存到全局
            app.config['LABEL_INFO'][filename.split("/")[-1]] = img_label_info

            img = Image.fromarray(np.uint8(annotated_frame[:, :, ::-1]))
            ex, ed = filename.split("/")[-1].split(".")

            tree.write(os.path.join(app.config["LABEL_FOLDER"], ex  + ".xml" ))
            img.save(os.path.join(app.config['OUTPUT_FOLDER'], ex + "_labeled" + "." + ed))
            label_image.append(url_for('output_file', filename=ex + "_labeled" + "." + ed))

    
    LABEL_STATUS = 1
    NUM = len(os.listdir(app.config['UPLOAD_FOLDER']))
    
    
    return render_template('index.html', num=NUM, images=images, labels=label_image, box=boxes)

# img1: {"img_path":path, "width":width, 
# "height":height, "boxes": [{"cls": cls, "xyxy": [xmin, ymin, xmax, ymax]}, ...]}
def generate_label(boxes, img_path, width, height):
    def generate_box(box):
        cls = box.cls
        print(box.xyxy[0].tolist())
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        return {"cls": int(cls), "xyxy": [int(xmin), int(ymin), int(xmax), int(ymax)]}
    img_path = img_path
    width = width
    height = height
    boxes_info = []
    for i in boxes:
        box_info = generate_box(i)
        boxes_info.append(box_info)
    return {        
            "img_path": img_path, 
            "width": width, 
            "height": height, 
            "boxes": boxes_info
            }


    

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/xmlPath/<filename>')
def label_file(filename):
    return send_from_directory(app.config['LABEL_FOLDER'], filename, as_attachment=True)

@app.route('/zipdownload')
def label_export():
    
    if len(os.listdir(app.config['LABEL_FOLDER'])) == 0:
        return jsonify({"message": "未上传文件！"})
    else:

    # 创建一个空的内存文件，用于存储压缩文件
        in_memory_zip = io.BytesIO()

        # 创建一个ZipFile对象，将压缩文件写入内存文件
        with zipfile.ZipFile(in_memory_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # 遍历目录下的所有文件，并将文件添加到zip文件中
            for root, dirs, files in os.walk(app.config['LABEL_FOLDER']):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 将文件添加到zip文件中，并指定压缩文件中的文件名为文件相对路径
                    zf.write(file_path, arcname=file)

        # 将内存文件指针移动到文件开头
        in_memory_zip.seek(0)
        
        # 返回压缩文件，指定文件名为download.zip
        return send_file(in_memory_zip, download_name='labels.zip', as_attachment=True)
    

@app.route('/status')
def get_label_status():
    print(LABEL_STATUS)
    return jsonify({'labelStatus': LABEL_STATUS})

@app.route('/export', methods=['POST'])
def exportXML():
    exp_img_name = request.form["expID"].split("/")[-1]
    
    img_path = app.config['LABEL_INFO'][exp_img_name]['img_path']

    filename = img_path.split("/")[-1].split(".")[0] + ".xml"
    
    return jsonify({"filename": filename})


def ToDict(box):
    return {"cls": [int(i) for i in list(box.cls)], "num": len(list(box.cls))}

def create_xml(img_label_info):
    annotation = create_tree(img_label_info)
    create_object(annotation, img_label_info)
    return annotation

def create_object(root, img_label_info):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # 创建一级分支object
    boxes = img_label_info["boxes"]
    for box in boxes:
        obj_name = coco_cls[int(box["cls"])]
        
        xi, yi, xa, ya = box["xyxy"]

        _object = ET.SubElement(root, 'object')
        # 创建二级分支
        name = ET.SubElement(_object, 'name')
        # print(obj_name)
        name.text = str(obj_name)
        pose = ET.SubElement(_object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(_object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(_object, 'difficult')
        difficult.text = '0'
        # 创建bndbox
        bndbox = ET.SubElement(_object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = '%s' % xi
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = '%s' % yi
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = '%s' % xa
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = '%s' % ya
 
 
# 创建xml文件的函数
def create_tree(img_label_info):
    imgpath = img_label_info["img_path"]
    imgdir = imgpath.split("/")[-2]
    image_name = imgpath.split("/")[-1]

    w = img_label_info["width"]
    h = img_label_info["height"]
    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    # folder = ET.SubElement(annotation, 'folder')
    # # 添加folder标签内容
    # folder.text = (imgdir)
 
    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name
 
    # 创建一级分支path
    # path = ET.SubElement(annotation, 'path')
 
    # path.text = imgpath  # 用于返回当前工作目录
 
    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
 
    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
 
    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    return annotation



if __name__ == '__main__':
    app.run(debug=True)
