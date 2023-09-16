#!/usr/bin/env python

# Enable robot before running test
# rosrun baxter_tools enable_robot.py -e

import tf
import os
import math
import time
import json
import rospy
import socket
import struct
import signal
import threading
import baxter_interface

from commands import *
from std_srvs.srv import Empty
from reflex_msgs.msg import PoseCommand
from geometry_msgs.msg import Twist


b_running = True
port = 3000

def timer_callback(event):
    global twist_output
    pub.publish(twist_output)

def base_move(twist, duration):
    global twist_output, zero_twist
    #rospy.loginfo("base_move", twist, duration)
    twist_output = twist
    rospy.sleep(duration)
    #twist_output = zero_twist
    #rospy.sleep(1.0)

def deg2rad(deg):
    return ((deg*math.pi)/180.0)
    
def rad2deg(rad):
    return ((rad/math.pi)*180.0)

def handler(signum, frame):
    rospy.loginfo("STOP signal was received! (Ctr+C)")
    global b_running
    b_running = False

def move_thread(limb, name, joint_pos):
    rospy.loginfo(name, ' move to ', joint_pos)
    limb.move_to_joint_positions(joint_pos)
    rospy.loginfo(name, " .Done!")

#Client has commands: GetPosLeft:NoData,GetPosRight:NoData, 
# SetPosL:DictionaryData, SetPosR:DictionaryData

def zero_hands_thread(pos_pub_left, pos_pub_right):
    pos_pub_left.publish(PoseCommand(0, 0, 0, 0)),
    pos_pub_right.publish(PoseCommand(0, 0, 0, 0))

def cmd_thread(cmd):
    cmd_str = ""
    if cmd == ENBABLE_ROBOT:
        cmd_str = "rosrun baxter_tools enable_robot.py -e"
    if cmd == DISABLE_ROBOT:
        cmd_str = "rosrun baxter_tools enable_robot.py -d"
    if cmd == TUCK_ROBOT:
        cmd_str = "rosrun baxter_tools tuck_arms.py -t"
    if cmd == UNTUCK_ROBOT:
        cmd_str = "rosrun baxter_tools tuck_arms.py -u"
    rospy.loginfo("Thread CMD = %s", cmd_str)
    os.system(cmd_str)

def client_thread_func(sock, left_limb, right_limb):
    rospy.loginfo("client_thread_func")
    calibrate_fingers_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_fingers', Empty)
    calibrate_tactile_hand1 = rospy.ServiceProxy('/reflex_takktile/calibrate_tactile', Empty)
    calibrate_fingers_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_fingers', Empty)
    calibrate_tactile_hand2 = rospy.ServiceProxy('/hand2/reflex_takktile/calibrate_tactile', Empty)
    pos_pub_right = rospy.Publisher('/reflex_takktile/command_position', PoseCommand, queue_size=1)
    pos_pub_left = rospy.Publisher('/hand2/reflex_takktile/command_position', PoseCommand, queue_size=1)
    
    #Hand Pose
    close_angle = 125
    preshape_angle = 0  #old value: 60
    p_open =  PoseCommand(deg2rad(0), deg2rad(0), deg2rad(0), deg2rad(preshape_angle))
    p_close = PoseCommand(deg2rad(close_angle), deg2rad(close_angle), deg2rad(close_angle), deg2rad(preshape_angle))
    #p_close = PoseCommand(0, 0, 0, 0)
    
    #Base     
    global twist_output, zero_twist
    
    speed = 0.15
    dist = 0.15
    vel = Twist()
    
    #Transformation
    listener = tf.TransformListener()
    global b_running    
    while b_running:        
        # receive data stream. it won't accept data packet greater than 1024 bytes
        cmd_bytes = sock.recv(4)
        cmd = struct.unpack('<I', cmd_bytes)[0]
        #rospy.loginfo("cmd = ", cmd)
        #sock.send("ACK".encode("ascii"))

        source_link = "/world"
        if cmd == GET_WLPOS:
            try:
                (trans,rot) = listener.lookupTransform(source_link, '/left_hand',  rospy.Time())
            except Exception as e:
                print('Exeption: GET_WLPOS ', e)
                trans = [0, 0, 0]
                rot = [0, 0, 0, 0]
            euler = tf.transformations.euler_from_quaternion(rot)
            data = trans + rot + list(euler) #Merge two lists
            json_data = json.dumps(data)
            l = int(len(json_data))
            sock.send(struct.pack('<I', l))
            sock.send(json_data.encode("ascii"))
        elif cmd == GET_WRPOS:
            try:
                (trans,rot) = listener.lookupTransform(source_link,'/right_hand',  rospy.Time())
            except Exception as e:
                print('Exeption: GET_WRPOS ', e)
                trans = [0, 0, 0]
                rot = [0, 0, 0, 0]
            euler = tf.transformations.euler_from_quaternion(rot)
            data = trans + rot + list(euler) #Merge two lists
            json_data = json.dumps(data)
            l = int(len(json_data))
            sock.send(struct.pack('<I', l))
            sock.send(json_data.encode("ascii"))
            
        elif cmd == GET_LPOS:
            #rospy.loginfo("GET_LPOS")
            angles = left_limb.joint_angles()
            json_angles = json.dumps(angles)
            l = int(len(json_angles))
            #rospy.loginfo(" >>> left l = ", l)
            sock.send(struct.pack('<I', l))
            #rospy.loginfo("Sending Left Arm Data")
            sock.send(json_angles.encode("ascii"))  # send data to the client
        
        elif cmd == GET_RPOS:
            #rospy.loginfo("GET_RPOS")
            angles = right_limb.joint_angles()
            json_angles = json.dumps(angles)
            l = int(len(json_angles))
            #rospy.loginfo(" >>> left l = ", l)
            sock.send(struct.pack('<I', l))
            #rospy.loginfo("Sending Left Arm Data")
            sock.send(json_angles.encode("ascii"))  # send data to the client
        
        elif cmd == ENBABLE_ROBOT or cmd == DISABLE_ROBOT or cmd == TUCK_ROBOT or cmd == UNTUCK_ROBOT:
            client_thread = threading.Thread(
                target=cmd_thread,
                args=(cmd,)
            )
            client_thread.start()
            
            
        elif cmd == CALIBRATE_HANDS:
            rospy.loginfo("CALIBRATE_HANDS")
            calibrate_fingers_hand1()
            calibrate_tactile_hand1()
            calibrate_fingers_hand2()
            calibrate_tactile_hand2()
            pass
        
        elif cmd == OPEN_LEFT_HAND:
            pos_pub_left.publish(p_open)
            rospy.loginfo("OPEN_LEFT_HAND")
            pass
            
        elif cmd == CLOSE_LEFT_HAND:
            pos_pub_left.publish(p_close)
            rospy.loginfo("CLOSE_LEFT_HAND")
            pass
            
        elif cmd == OPEN_RIGHT_HAND:
            pos_pub_right.publish(p_open)
            rospy.loginfo("OPEN_RIGHT_HAND")
            pass
            
        elif cmd == CLOSE_RIGHT_HAND:
            pos_pub_right.publish(p_close)
            rospy.loginfo("CLOSE_RIGHT_HAND")
            pass
        
        elif cmd == ZERO_HANDS:
            #pos_pub_left.publish(PoseCommand(0, 0, 0, 0))
            #pos_pub_right.publish(PoseCommand(0, 0, 0, 0))
            
            zero_hands_thread = threading.Thread(
                target=lambda :( 
                   pos_pub_left.publish(PoseCommand(0, 0, 0, 0)),
                   pos_pub_right.publish(PoseCommand(0, 0, 0, 0))
                ),
                args=()
            )
            zero_hands_thread.start()
            rospy.loginfo("ZERO_HANDS")
            pass
            
        #Jogging
        elif LEFT_S0_NAG <= cmd and cmd <= RIGHT_W2_POS:
            left = left_limb
            right = right_limb
            lj = left.joint_names()
            rj = right.joint_names()
            def set_j(limb, joint_name, delta):
                current_position = limb.joint_angle(joint_name)
                joint_command = {joint_name: current_position + delta}
                limb.set_joint_positions(joint_command)
            bindings = {
                #key: (function, args, description)
                21: (set_j, [left, lj[0], 0.1], "left_s0 increase"),
                22: (set_j, [left, lj[0], -0.1], "left_s0 decrease"),
                23: (set_j, [left, lj[1], 0.1], "left_s1 increase"),
                24: (set_j, [left, lj[1], -0.1], "left_s1 decrease"),
                25: (set_j, [left, lj[2], 0.1], "left_e0 increase"),
                26: (set_j, [left, lj[2], -0.1], "left_e0 decrease"),
                27: (set_j, [left, lj[3], 0.1], "left_e1 increase"),
                28: (set_j, [left, lj[3], -0.1], "left_e1 decrease"),
                29: (set_j, [left, lj[4], 0.1], "left_w0 increase"),
                30: (set_j, [left, lj[4], -0.1], "left_w0 decrease"),
                31: (set_j, [left, lj[5], 0.1], "left_w1 increase"),
                32: (set_j, [left, lj[5], -0.1], "left_w1 decrease"),
                33: (set_j, [left, lj[6], 0.1], "left_w2 increase"),
                34: (set_j, [left, lj[6], -0.1], "left_w2 decrease"),
                41: (set_j, [right, rj[0], 0.1], "right_s0 increase"),
                42: (set_j, [right, rj[0], -0.1], "right_s0 decrease"),
                43: (set_j, [right, rj[1], 0.1], "right_s1 increase"),
                44: (set_j, [right, rj[1], -0.1], "right_s1 decrease"),
                45: (set_j, [right, rj[2], 0.1], "right_e0 increase"),
                46: (set_j, [right, rj[2], -0.1], "right_e0 decrease"),
                47: (set_j, [right, rj[3], 0.1], "right_e1 increase"),
                48: (set_j, [right, rj[3], -0.1], "right_e1 decrease"),
                49: (set_j, [right, rj[4], 0.1], "right_w0 increase"),
                50: (set_j, [right, rj[4], -0.1], "right_w0 decrease"),
                51: (set_j, [right, rj[5], 0.1], "right_w1 increase"),
                52: (set_j, [right, rj[5], -0.1], "right_w1 decrease"),
                53: (set_j, [right, rj[6], 0.1], "right_w2 increase"),
                54: (set_j, [right, rj[6], -0.1], "right_w2 decrease"),
            }
            
            jcmd = bindings[cmd]
            jcmd[0](*jcmd[1])
            rospy.loginfo("command: %s" % (jcmd[2],))
        
        elif cmd == BASE_MOVE_FORWARD:
            rospy.loginfo("BASE_MOVE_FORWARD")
            vel.linear.x = speed
            vel.linear.y = 0
            vel.angular.z = 0
            base_move(vel, dist/speed)
           
        
        elif cmd == BASE_MOVE_BACKWARD:
            rospy.loginfo("BASE_MOVE_BACKWARD")
            vel.linear.x = -speed
            vel.linear.y = 0
            vel.angular.z = 0
            base_move(vel, dist/speed)
            
            
        elif cmd == BASE_MOVE_LEFT:
            rospy.loginfo("BASE_MOVE_LEFT")
            vel.linear.x = 0
            vel.linear.y = speed
            vel.angular.z = 0
            base_move(vel, dist/speed)
            
            
        elif cmd == BASE_MOVE_RIGHT:
            rospy.loginfo("BASE_MOVE_RIGHT")
            vel.linear.x = 0
            vel.linear.y = -speed
            vel.angular.z = 0
            base_move(vel, dist/speed)
            
        elif cmd == BASE_TURN_LEFT:
            rospy.loginfo("BASE_TURN_LEFT")
            vel.linear.x = 0
            vel.linear.y = 0
            vel.angular.z = speed
            base_move(vel, dist/speed)
            
        elif cmd == BASE_TURN_RIGHT:
            rospy.loginfo("BASE_TURN_RIGHT")
            vel.linear.x = 0
            vel.linear.y = 0
            vel.angular.z = -speed
            base_move(vel, dist/speed)
        
        elif cmd == BASE_STOP:
            rospy.loginfo("BASE_STOP")
            vel.linear.x = 0
            vel.linear.y = 0
            vel.angular.z = 0
            base_move(vel, dist/speed)
        
    rospy.loginfo("Disconnect from the client")
    
def init_main():

    global b_running
    
    #Arms
    rospy.init_node('Baxter_Move_Server')
    left_limb = baxter_interface.Limb('left')
    right_limb = baxter_interface.Limb('right')
    
    #Hands
    # Services can automatically call hand calibration
 
 
    #Base init
    global pub
    pub = rospy.Publisher('/mobility_base/cmd_vel', Twist, queue_size=1)
    #rospy.init_node('motion_demo');
    rospy.Timer(rospy.Duration(0.1), timer_callback)

    global zero_twist
    zero_twist = Twist()
    zero_twist.linear.x = 0
    zero_twist.linear.y = 0
    zero_twist.angular.z = 0
    
    global twist_output
    twist_output = Twist()
    
    #Ctr+C to escape program
    signal.signal(signal.SIGINT, handler)
    
    #Create and start a server
    host = socket.gethostname()
    rospy.loginfo("Listening at %s:%d" %(host, port))
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    
    server_socket.setblocking(1)
    while b_running:
        try:
            sock, address = server_socket.accept()
            rospy.loginfo("Connection from: " + str(address))
            client_thread = threading.Thread(
                target=client_thread_func,
                args=(sock, left_limb, right_limb)
            )
            client_thread.start()
            time.sleep(0.5)
        except:
            rospy.loginfo("Listening time out")

###################
init_main()
