import cv2,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random,colorsys
from PIL import Image
from easydict import EasyDict as edict
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
output = os.path.join(BASE_DIR,'detections/')
weights = (os.path.join(BASE_DIR,'weights/'))
__C = edict()
cfg = __C
__C.YOLO = edict()
__C.YOLO.CLASSES = os.path.join(BASE_DIR,'data/classes.names')
__C.YOLO.ANCHORS_TINY =[23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES_TINY =[16, 32]
__C.YOLO.XYSCALE_TINY =[1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE =3
__C.YOLO.IOU_LOSS_THRESH =0.5
__C.YOLO.ANCHORS= [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.STRIDES= [8, 16, 32]
__C.YOLO.XYSCALE= [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE= 3
__C.YOLO.IOU_LOSS_THRESH= 0.5
__C.YOLO.ANCHORS= [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.STRIDES= [8, 16, 32]
__C.YOLO.XYSCALE= [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE= 3
__C.YOLO.IOU_LOSS_THRESH= 0.5
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
def load_config():
    STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
    ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY)
    XYSCALE = cfg.YOLO.XYSCALE_TINY
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))
    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE
def get_anchors(anchors_path):
    anchors = np.array(anchors_path)
    return anchors.reshape(2, 3, 2)
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes
def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label = True, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values()), read_text = False):
    classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name not in allowed_classes:
            continue
        else:
            if read_text:
                height_ratio = int(image_h / 25)
                img_text = None
                if img_text != None:
                    cv2.putText(image, img_text, (int(coor[0]), int(coor[1]-height_ratio)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,0), 2)
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)
                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio
    return image
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data
    counts = dict()
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)
        for i in range(num_objects):
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue
    else:
        counts['total object'] = num_objects
    return counts
def detect_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    counts = dict()
    for i in range(num_objects):
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            xmin, ymin, xmax, ymax = boxes[i]
            cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            cv2.imwrite(img_path, cropped_img)
        else:
            continue
config = ConfigProto()
STRIDES, ANCHORS, NUM_CLASSES, XYSCALE = load_config()
input_size = 416
saved_model_load = tf.saved_model.load(weights, tags = [tag_constants.SERVING])
infer = saved_model_load.signatures['serving_default']
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        return image
def gen(camera):
    while True:
        frame = camera.get_frame()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0 : 4]
            pred_conf = value[:, :, 4:]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),max_output_size_per_class = 10,max_total_size = 50,iou_threshold = 0.45,score_threshold = 0.6)
        original_h, original_w, _ = frame.shape
        bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        class_names = read_class_names(cfg.YOLO.CLASSES)
        allowed_classes = list(class_names.values())
        counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
        image = draw_bbox(frame,pred_bbox,counted_classes,allowed_classes = allowed_classes,read_text = False)
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', result)
        b=jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b + b'\r\n\r\n')
