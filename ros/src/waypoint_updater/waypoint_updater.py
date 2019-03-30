#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
from scipy.spatial import KDTree
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
#LOOKAHEAD_REDLIGHT = 20

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.waypoint_tree = None
        self.pose = None

        self.stopline_wp_idx = -1
        self.max_decel = abs(rospy.get_param('~decel_limit', -5))

        self.loop()
        #rospy.spin()

    def loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints_2d:
                closest_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.position.x
        y = self.pose.position.y
        #rospy.logwarn('Current position is: (%f, %f)', x, y)

        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        closest_point = self.base_waypoints_2d[closest_idx]
        prev_point = self.base_waypoints_2d[closest_idx-1]

        cl_vect = np.array(closest_point)
        prev_vect = np.array(prev_point)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.base_waypoints_2d)
        return closest_idx
        
    def publish_waypoints(self, closest_idx):
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx +
                #LOOKAHEAD_WPS]
        #rospy.logwarn('The closest idx is %d', closest_idx)
        #rospy.logwarn('The first two way points are (%f, %f), (%f, %f)',
                #lane.waypoints[0].pose.pose.position.x,
                #lane.waypoints[0].pose.pose.position.y, 
                #lane.waypoints[1].pose.pose.position.x, 
                #lane.waypoints[1].pose.pose.position.y)
        lane = self.generate_lane(closest_idx)
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        final_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or farthest_idx < self.stopline_wp_idx:
            lane.waypoints = final_waypoints
        else:
            #rospy.logwarn('Start decelerating=====================')
            lane.waypoints = self.decelerate_waypoints(final_waypoints,
                    closest_idx) 
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        final_waypoints = []

        stop_idx = max(self.stopline_wp_idx-closest_idx-8, 0)
        for idx,  wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            dist = self.distance(waypoints, idx, stop_idx)
            vel = math.sqrt(2*self.max_decel*dist)
            if vel < 1.:
                vel = 0
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            #rospy.loginfo('Set linear velocity to %f', p.twist.twist.linear.x)
            final_waypoints.append(p)
            if idx > stop_idx:
                break

        return final_waypoints

            


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        rospy.loginfo('Receiving Base Waypoints')
        self.base_waypoints = waypoints
        if not self.base_waypoints_2d:
            self.base_waypoints_2d = [[waypoint.pose.pose.position.x,
              waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            for data in self.base_waypoints_2d:
                rospy.loginfo('x=%f, y=%f', data[0], data[1])
            rospy.loginfo('Constructing KD Tree')
            self.waypoint_tree = KDTree(self.base_waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
