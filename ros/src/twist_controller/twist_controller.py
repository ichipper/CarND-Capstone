import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1,
                max_lat_accel, max_steer_angle)
        #kp = 0.7
        #ki = 0.
        #kd = 0.
        #self.steer_controller = PID(kp, ki, kd)

        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.
        mx = 0.2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

        

    def control(self, linear_vel, angular_vel, current_vel, current_ang_vel, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            #self.steer_controller.reset()
            self.last_time = rospy.get_time()
            return 0., 0., 0

        #rospy.loginfo('linear_vel=%f', linear_vel)
        #rospy.loginfo('angular_vel=%f', angular_vel)
        #rospy.loginfo('current_vel=%f', current_vel)
        steering = self.yaw_controller.get_steering(linear_vel,
                angular_vel, current_vel)
            
        current_vel = self.vel_lpf.filt(current_vel)
        time_now = rospy.get_time()
        time_elapsed = time_now - self.last_time
        self.last_time = time_now
        
        #ang_error = angular_vel - current_ang_vel
        #rospy.logwarn('angular velocity=%f, current angular velocity=%f, angular \
        #error=%f', angular_vel, curren_ang_vel, ang_error)
        #steering = self.steer_controller.step(ang_error, time_elapsed)

        vel_error = linear_vel - current_vel
        #rospy.loginfo('linear velocity=%f, current  velocity=%f, velocity \
        #error=%f', linear_vel, current_vel, vel_error)
        throttle = self.throttle_controller.step(vel_error, time_elapsed)

        rospy.loginfo('target linear_vel=%f, target angular_vel=%f,\
                current_vel=%f, current_ang_vel=%f, steering=%f, throttle=%f', linear_vel, angular_vel,
                current_vel, current_ang_vel, steering, throttle)

        brake = 0


        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius

        return throttle, brake, steering 
