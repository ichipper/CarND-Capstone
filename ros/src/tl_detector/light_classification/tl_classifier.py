from styx_msgs.msg import TrafficLight
import os
import numpy as np
import tensorflow as tf
#SSD_INCEPTION_MODEL_FILE = '../../Traffic-Light-Classification/models/ssd_sim/frozen_inference_graph.pb'
SSD_INCEPTION_MODEL_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../../../../models/ssd_sim/frozen_inference_graph.pb')

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
            if len(scores) == 0:
                return TrafficLight.UNKNOWN
            else:
                label = classes[0]
                if label == 1:
                    return TrafficLight.GREEN
                elif label == 2:
                    return TrafficLight.RED
                elif label == 3:
                    return TrafficLight.YELLOW
                else:
                    return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN
