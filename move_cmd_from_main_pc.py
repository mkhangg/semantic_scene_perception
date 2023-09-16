#! usr/bin/python3

import time
import yaml
import socket

# Safe load configuration yaml file
with open("config/config.yaml", 'r') as file:
    hyperparams = yaml.safe_load(file) 

BASE_MOVE_FORWARD = 61
BASE_MOVE_BACKWARD = 62
BASE_MOVE_LEFT = 63
BASE_MOVE_RIGHT = 64
BASE_TURN_LEFT = 65
BASE_TURN_RIGHT = 66
BASE_STOP = 67
STOP_TIME = hyperparams["base_stop_time"]


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
print(f">> Connect to {ip_address}.")

send_cmd(conn, BASE_MOVE_LEFT, hyperparams["move_left_time"])
send_cmd(conn, BASE_TURN_RIGHT, hyperparams["turn_right_time"])
send_cmd(conn, BASE_MOVE_FORWARD, hyperparams["base_stop_time"])

conn.close()
print(">> Done execution.")
