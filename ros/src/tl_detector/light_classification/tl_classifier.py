from styx_msgs.msg import TrafficLight
import os
import numpy as np
import tensorflow as tf
import cv2
COLOR_MAP = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 255, 0), 4: (0, 0, 255)}
COLOR_NAME = {1: 'Green', 2: 'Red', 3: 'Yellow', 4: 'Off'}
#SSD_INCEPTION_MODEL_FILE = '../../Traffic-Light-Classification/models/ssd_sim/frozen_inference_graph.pb'
SSD_INCEPTION_MODEL_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../../../models/ssd_sim/frozen_inference_graph.pb')
DEBUG = True

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.detection_graph = self.load_graph(SSD_INCEPTION_MODEL_FILE)
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.save_img_idx = 0

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].
        
        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        
        return box_coords

    def draw_boxes(self, image, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            cv2.rectangle(image, (left, top), (right, bot), COLOR_MAP[class_id], thickness)
            score_str = '%s=%f' %(COLOR_NAME[class_id], scores[i])
            cv2.putText(image, score_str, (left, bot), cv2.FONT_HERSHEY_SIMPLEX,  1, COLOR_MAP[class_id], 2)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.detection_graph) as sess:                
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes,
                self.detection_scores, self.detection_classes], 
                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            boxes, scores, classes = self.filter_boxes(0.1, boxes, scores, classes)
            #To save image for debugging:
            if DEBUG:
                print boxes
                print scores
                print classes
                self.save_img_idx += 1
                img_file_path = '/tmp/img/img%d.png' %(self.save_img_idx)
                height, width = image.shape[0], image.shape[1]
                box_coords = self.to_image_coords(boxes, height, width)

                # Each class with be represented by a differently colored box
                self.draw_boxes(image, box_coords, classes, scores)
                cv2.imwrite(img_file_path, image)

            if len(scores) == 0:
                return TrafficLight.UNKNOWN
            else:
                scoreboard = {}
                for idx in range(scores.size):
                    if classes[idx] not in scoreboard:
                        scoreboard[classes[idx]] = scores[idx]
                    else:
                        scoreboard[classes[idx]] += scores[idx]
                label = max(scoreboard.iterkeys(), key=(lambda key: scoreboard[key])) 
                if label == 1:
                    return TrafficLight.GREEN
                elif label == 2:
                    return TrafficLight.RED
                elif label == 3:
                    return TrafficLight.YELLOW
                else:
                    return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN
