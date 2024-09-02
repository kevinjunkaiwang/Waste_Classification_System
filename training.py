from pycocotools.coco import COCO
from PIL import Image, ExifTags
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
# from tensorflow.python import keras

import colorsys
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import tensorflow as tf


num_classes = 60
width = 256
height = 256

orientation_code = 274


def exif_orientation_to_rotations(img_file_path, orientation_code=orientation_code):
    image = Image.open(img_file_path)
    
    try:
        if image._getexif():
            exif = dict(image._getexif().items())
        
        if orientation_code in exif:
            orientation = exif[orientation_code]
            if orientation == 3:
                return 2  # 180 degrees rotation
            elif orientation == 6:
                return 3  # 270 degrees rotation (or 90 degrees counter-clockwise)
            elif orientation == 8:
                return 1  # 90 degrees rotation
        return 0  # No rotation
    except:
        return 0
            

def load_and_process_image(image_path, bbox, label, scale, num_rotations):
    """ Image """
    # Load the image
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_data, channels=3)
    
    # Apply EXIF rotation using tf.image.rot90
    image = tf.image.rot90(image, k=num_rotations)

    # Resize the image
    image = tf.image.resize(image, [height, width])
    image = image / 255.0  # Normalize the image to [0, 1]

    ''' Bounding Box '''
    x, y, w, h = bbox
    scale_x, scale_y = scale

    # Adjust x, y, w, h based on the scaling factors
    x = x * scale_x
    y = y * scale_y
    w = w * scale_x
    h = h * scale_y

    # x1, y1 = x - w/2, y - h/2
    # x2, y2 = x + w/2, y + h/2
    
    norm_bbox = [x, y, w, h]

    # norm_bbox = [norm_x1, norm_y1, norm_x2, norm_y2]

    ''' Label '''
    # Convert string label to integer using label_map
    label_index = label_map[label.numpy().decode('utf-8')]  # Decode bytes to string and map to integer

    # Apply one-hot encoding
    one_hot_label = tf.one_hot(label_index, num_classes)
    
    return image, norm_bbox, one_hot_label


def tf_dataset(images, bboxes, labels, scales, rotations, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices((images, bboxes, labels, scales, rotations))

    # Parse dataset
    ds = ds.map(lambda img, bbox, label, scale, rotation: tf.py_function(
        func=load_and_process_image,
        inp=[img, bbox, label, scale, rotation],
        Tout=[tf.float32, tf.float32, tf.float32]
    ))

    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache().batch(batch_size).prefetch(AUTOTUNE)

    return ds


# Load dataset 
dataset_path = os.path.join('TACO', 'data')
anns_file_path = os.path.join(dataset_path, 'annotations.json')

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

categories = dataset['categories']
annotations = dataset['annotations']
imgs = dataset['images']
nr_cats = len(categories)
nr_annotations = len(annotations)
nr_images = len(imgs)

# Load categories and super categories
cat_names = []
super_cat_names = []
super_cat_ids = {}
last_super_cat_name = ''
nr_super_cats = 0
for cat_it in categories:
    cat_names.append(cat_it['name']) 
    super_cat_name = cat_it['supercategory']
    
    # Adding new supercat
    if super_cat_name != last_super_cat_name:
        super_cat_names.append(super_cat_name)
        super_cat_ids[super_cat_name] = nr_super_cats
        last_super_cat_name = super_cat_name
        nr_super_cats += 1

print('Number of super categories:', nr_super_cats)
print('Number of categories:', nr_cats)
print('Number of annotations:', nr_annotations)
print('Number of images:', nr_images)

# for img in imgs:
#     image_file_path = os.path.join(dataset_path, img['file_name']).replace('/', '\\')
#     image = Image.open(image_file_path)

#     if image._getexif():
#         exif = dict(image._getexif().items())

#     # Obtain Exif orientation tag code
#     for orientation in ExifTags.TAGS.keys():
#         if ExifTags.TAGS[orientation] == 'Orientation':
#             break
    
#     if orientation != orientation_code:
#         print("check")

# # Visualizing Sample Images with Bounding Boxes
# # Loads dataset as a coco object
# coco = COCO(anns_file_path)

# for i in range(88, 89):
#     img = imgs[i]
#     img_id = img['id']
#     img_file_name = img['file_name']
#     img_file_path = os.path.join(dataset_path, img_file_name)
    
#     exif_orientation_to_rotations(img_file_path)
#     I = Image.open(img_file_path)
    
#     # Show image
#     fig,ax = plt.subplots(1)
#     plt.axis('off')
#     plt.imshow(I)

#     # Load mask ids
#     annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
#     anns_sel = coco.loadAnns(annIds)
    
#     # Show annotations
#     for ann in anns_sel:
#         color = colorsys.hsv_to_rgb(np.random.random(),1,1)
        
#         for seg in ann['segmentation']:
#             poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
#             p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
#             ax.add_collection(p)
#             p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
#             ax.add_collection(p)
        
#         [x, y, w, h] = ann['bbox']
#         rect = Rectangle((x,y),w,h,linewidth=2,edgecolor=color,
#                          facecolor='none', alpha=0.7, linestyle = '--')
#         ax.add_patch(rect)
        
#         cat = coco.loadCats(ann['category_id'])[0]
#         ax.annotate(cat['name'], (x, y), color=color)

#     plt.show()

# Loads dataset as a coco object
coco = COCO(anns_file_path)

images = []
bboxes = []
labels = []
scales = []
rotations = []

for ann in annotations:
    img_info = coco.loadImgs(ann["image_id"])[0]
    image_file_path = os.path.join(dataset_path, img_info['file_name']).replace('/', '\\')
    images.append(image_file_path)
    print(image_file_path)
    
    rotation = exif_orientation_to_rotations(image_file_path)
    rotations.append(rotation)

    img_width = img_info['width']
    img_height = img_info['height']
    scale_x = width / img_width
    scale_y = height / img_height
    scales.append([scale_x, scale_y])
    
    bboxes.append(ann['bbox'])

    cat_info = coco.loadCats(ann['category_id'])[0]
    labels.append(cat_info['name'])

label_map = {label: idx for idx, label in enumerate(cat_names)}

# mid=3
# end=40

# train_images = images[:mid]
# train_bboxes = bboxes[:mid]
# train_labels = cat_names[:mid]

# valid_images = images[mid:end]
# valid_bboxes = bboxes[mid:end]
# valid_labels = cat_names[mid:end]

# test_images = images[end:]
# test_bboxes = bboxes[end:]
# test_labels = cat_names[end:]

# X_train = train_images
# y_train = np.array((train_bboxes, train_labels), dtype="object")
# X_test = test_images
# y_test = np.array((test_bboxes, test_labels), dtype="object")

for i in range(229,230):
    test_image = images[i]
    test_bbox = bboxes[i]
    test_label = tf.constant(labels[i], dtype=tf.string)
    test_scale = scales[i]
    test_rotation = rotations[i]

    # Call the function
    processed_image, processed_bbox, processed_label = load_and_process_image(test_image, test_bbox, test_label, test_scale, test_rotation)

    # Print outputs to verify
    print("Processed Image Shape:", processed_image.shape)
    print("Processed BBox:", processed_bbox)
    print("Processed Label:", processed_label.numpy())  # Convert to NumPy to view

    # Plot the processed image
    plt.imshow(processed_image.numpy())
    plt.axis('off')

    # Overlay the bounding box
    x, y, w, h = processed_bbox
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', lw=2))

    # Show the plot
    plt.show()


spit = math.floor(0.8*len(images))

train_ds = tf_dataset(images[:spit], bboxes[:spit], labels[:spit], scales[:spit], rotations[:spit])
valid_ds = tf_dataset(images[spit:], bboxes[spit:], labels[spit:], scales[spit:], rotations[spit:])

# Example: Iterate through the dataset
for batch_images, batch_bboxes, batch_labels in train_ds.take(1):
    print(batch_images.shape)  # (batch_size, height, width, 3)
    print(batch_bboxes.shape)  # (batch_size, 4)
    print(batch_labels.shape)  # (batch_size, num_classes)