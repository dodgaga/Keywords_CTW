import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
#matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '/home/wudao/ctw/ctw-baseline-master/ssd/caffe'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

#print os.getcwd()

import caffe
print os.getcwd()
caffe.set_device(5)
caffe.set_mode_gpu()
print os.getcwd()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/wudao/ctw/ctw-baseline-master/ssd/caffe/data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

model_def = 'models/VGGNet/VOC0712/SSD_300x300_ft/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel'

print os.getcwd()

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

dirname = '/home/wudao/imagesnocrop/'
for maindir, subdir, file_name_list in os.walk(dirname):
    for filename in file_name_list:
        image_path = os.path.join(maindir, filename)
        image = caffe.io.load_image(image_path)
        #plt.imshow(image)
        print os.getcwd()
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        
        
        num_region = top_conf.shape[0]
        print("num_region",num_region)
        if(num_region == 0):
            continue
        
        top_label_indices = det_label[top_indices].tolist()
        top_animal_indices = [i for i, label in enumerate (top_label_indices) if (label == 3
        or label == 8 or label ==10 or label==12 or label==13 or label==17)]
        
        print("animal_indices",top_animal_indices)
       # top_labels = get_labelname(labelmap, top_label_indices)
        if (len(top_animal_indices) == 0):
            continue
        
        top_xmin = det_xmin[top_indices][top_animal_indices]
        top_ymin = det_ymin[top_indices][top_animal_indices]
        top_xmax = det_xmax[top_indices][top_animal_indices]
        top_ymax = det_ymax[top_indices][top_animal_indices]



        xmin = np.array(top_xmin * image.shape[1])
        ymin = np.array(top_ymin * image.shape[0])
        xmax = np.array(top_xmax * image.shape[1])
        ymax = np.array(top_ymax * image.shape[0])
        print("detection_box",xmin,ymin,xmax,ymax)
        #choose the max region

        if(num_region > 1):
            width = np.array(xmax - xmin)
            height = np.array(ymax - ymin)
            size = width*height
            max_index = np.where(size == np.max(size))
            xmin = xmin[max_index]
            ymin = ymin[max_index]
            xmax = xmax[max_index]
            ymax = ymax[max_index]
        
        #im = cv2.imread(image_path)
        im = Image.open(image_path)
        print("top_xmin",top_xmin)
        
        #box = (xmin,ymin,xmax,ymax)
        
        width = xmax - xmin
        height = ymax - ymin
        center_x = [xmin + (xmax - xmin)/2]
        center_y = [ymin + (ymax - ymin)/2]
        
        board = max(width, height)
        
        box = (max(0, center_x - board/2),
        max(0, center_y - board/2),
        min(image.shape[1], center_x + board/2),
        min(image.shape[0], center_y + board/2))
        print("box",box)
        crop_image = im.crop(box)
        resize_image = crop_image.resize((256,256))
        #crop_image = im[box[1]:box[3], box[0]:box[2]]
        #resize_image = cv2.resize(crop_image, (256,256), interpolation=cv2.INTER_CUBIC)
        
        crop_image_path = os.path.join('/home/wudao/imagescrop', filename)
        resize_image.save(crop_image_path)
        #cv2.imwrite(crop_image_path,resize_image)


#colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

#plt.imshow(image)
