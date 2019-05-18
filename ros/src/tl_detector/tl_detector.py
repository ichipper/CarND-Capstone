#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

from scipy.spatial import KDTree
import numpy as np

STATE_COUNT_THRESHOLD = 2
SAVE_IMAGE_INTERVAL = 2
SAVE_IMAGE_START = 0

state_dict = {TrafficLight.RED:'Red', TrafficLight.GREEN:'Green',
        TrafficLight.YELLOW:'Yellow', TrafficLight.UNKNOWN:'Unknown'}

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.camera_image = None
        self.lights = []
        self.image_counter = 0
        self.save_image_counter = SAVE_IMAGE_START

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config['is_site']
    

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        #rospy.spin()
        self.ros_spin()

    def ros_spin(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.pose is not None and self.waypoints is not None and self.camera_image is not None:
                self.image_counter += 1

                ########################
                
                if self.image_counter == 4:
                    light_wp, state = self.process_traffic_lights()
                    self.image_counter = 0
                else:
                    rate.sleep()
                    continue

                '''
                Publish upcoming red lights at camera frequency.
                Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''
                if self.state != state:
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD:
                    #if self.last_state == TrafficLight.RED and self.state == TrafficLight.GREEN:
                        #self.image_counter = -200
                    self.last_state = self.state
                    light_wp = light_wp if state == TrafficLight.RED else -1
                    self.last_wp = light_wp
                    self.upcoming_red_light_pub.publish(Int32(light_wp))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1

                #if light_wp >= 0:
                    #rospy.logwarn('Red light!')
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                waypoint.pose.pose.position.y] for waypoint in
                waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
    def save_camera_images(self, camera_msg, state):
        self.save_image_counter += 1
        if self.save_image_counter % SAVE_IMAGE_INTERVAL != 0:
            return
        # Decode to cv2 image and store
        cv2_img = self.bridge.imgmsg_to_cv2(camera_msg, "rgb8")
        #cv2_img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        img_file_path = "/tmp/camera_img/img%d.png" %(self.save_image_counter/SAVE_IMAGE_INTERVAL)
        cv2.imwrite(img_file_path, cv2_img)
        rospy.loginfo("Saved to: " + img_file_path)
        light_wp, state = self.process_traffic_lights()
        if state == TrafficLight.RED:
            label = 'red'
        elif state == TrafficLight.YELLOW:
            label = 'yellow'
        elif state == TrafficLight.GREEN:
            label = 'green'
        else:
            label = 'unknown'
        with open('/tmp/camera_img_label', 'a') as fo:
            fo.write(img_file_path+', '+label+'\n' )
        
        
    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        x = pose.position.x
        y = pose.position.y
        #rospy.logwarn('Current position is: (%f, %f)', x, y)

        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        closest_point = self.waypoints_2d[closest_idx]
        prev_point = self.waypoints_2d[closest_idx-1]

        cl_vect = np.array(closest_point)
        prev_vect = np.array(prev_point)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #self.save_camera_images(self.camera_image, light.state)
        #return light.state
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        light_state = self.light_classifier.get_classification(cv_image)
        if not self.is_site:
            rospy.loginfo('Predicted state is %s, and real state is %s',
                    state_dict[light_state], state_dict[light.state]  )
            if light_state != light.state:
                rospy.logwarn('Predicted state is %s, and real state is %s',
                        state_dict[light_state], state_dict[light.state]  )
        else:
            rospy.loginfo('Predicted state is %s', state_dict[light_state] )
        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        closest_light = None
        light_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for idx, light in enumerate(self.lights):
                stop_line = stop_line_positions[idx]
                stop_line_pose = Pose()
                stop_line_pose.position.x = stop_line[0]
                stop_line_pose.position.y = stop_line[1]
                tmp_wp_idx = self.get_closest_waypoint(stop_line_pose)

                d = tmp_wp_idx - car_position
                if d > 0 and d < diff:
                    light_wp_idx = tmp_wp_idx
                    closest_light = light
                    diff = d

        if closest_light and diff < 50:
            state = self.get_light_state(closest_light)
            return light_wp_idx, state
        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
