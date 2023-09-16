#! usr/bin/python3

import time
import yaml
import socket

# Safe load configuration yaml file
with open("config/config.yaml", 'r') as file:
    hyperparams = yaml.safe_load(file) # safe_load is recommended to use

BASE_MOVE_FORWARD = 61
BASE_MOVE_BACKWARD = 62
BASE_MOVE_LEFT = 63
BASE_MOVE_RIGHT = 64
BASE_TURN_LEFT = 65
BASE_TURN_RIGHT = 66
BASE_STOP = 67
STOP_TIME = hyperparams["base_stop_time"]
CAPTURE_FOLDER = hyperparams["capture_folder"]

def send_cmd(conn, command, duration):
    if conn != None:
        conn.send(command.to_bytes(4, 'little'))
        time.sleep(duration)
        conn.send(BASE_STOP.to_bytes(4, 'little'))
        time.sleep(STOP_TIME)

conn = socket.socket()
conn.setblocking(1)
ip_address, port_no = hyperparams["ip_address"], hyperparams["port_no"]
conn.connect((ip_address, port_no))

send_cmd(conn, BASE_TURN_RIGHT, 17.0)
send_cmd(conn, BASE_MOVE_FORWARD, 10.0)
time.sleep(4.0)
conn.close()
