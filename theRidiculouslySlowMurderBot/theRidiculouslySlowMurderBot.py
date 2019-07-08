## THIS CODE WAS ORIGINALLY FOR A RACOON DETECTOR. ( lol ikr ? )

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

## mss is supposed to be v'fast. I have no complaints with it either.
from mss import mss
import cv2
import keys as k
import time
keys = k.Keys() ## HELPS WITH MOUSE INPUT ETC. THIS IS A DIRECT INPUT


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util




# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.

## I USED THIS TO GET THE LABELS FOR " PEOPLE "
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model




# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

## SCREEN DEFINING FOR MSS
sct = mss()
monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

## MOVEMENT WALA PART

'''def movement_jugaad(mid_x, mid_y,width=800, height=600):
    x_move = 0.5-mid_x
    y_move = 0.5-mid_y
    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, -1*int(x_move*width), -1*int(y_move*height)))'''

def movement_jugaad(mid_x, mid_y, width = monitor['width'], height = monitor['height'] - 15): # 753 to compensate for the title bar
    x_move = mid_x - 0.5 # banda ka x coord kaha hai minus the centre of the screen gives us the distance to move in the x direction
    y_move = mid_y - 0.5 # ^ same
    hm_x = x_move # kitna move karne ka hai in the x axis
    hm_y = y_move # ^ same :)


    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, int(hm_x*800), int(hm_y*600)))




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.15)





with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:


          image_np = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGR2RGB)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

          # Visualization of the results of a detection.

          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=4)

          for i,b in enumerate(boxes[0]):
                  if classes[0][i] == 1:# and scores[0][i]>=SCORE:    
                      if scores[0][i] >= 0.7:
                          mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2  ## DETERMINING THE CENTRE OF THE PERSON - X
                          mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2  ## ^ SAME - Y
                          movement_jugaad(mid_x=mid_x, mid_y=mid_y)  
                    

                            #movement_jugaad(mid_x=banda_pasand[0], mid_y=banda_pasand[1])
                           


                          keys.directMouse(buttons=keys.mouse_rb_press)
                          keys.directMouse(buttons=keys.mouse_lb_press)
                          #keys.directMouse(buttons=keys.mouse_lb_release)
                            #keys.directMouse(buttons=keys.mouse_rb_release)
                          # print(f'{banda_pasand} - ye wala')

                      else:
                               keys.directMouse(buttons=keys.mouse_rb_release)
                               keys.directMouse(buttons=keys.mouse_lb_release)
                            # keys.directKey('w')
                             

                  

                      #cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                      if cv2.waitKey(25) & 0xFF == ord('q'):
                          cv2.destroyAllWindows()
                          break

