#! usr/bin/python3

import numpy as np
import datetime as dt
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

pipeline.start(config)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs, gx, gy, gz = [], [], [], []

def plot_gyro_realtime(_, xs, gx, gy, gz):
    imu_data = pipeline.wait_for_frames()
    gyro_frame = imu_data[1].as_motion_frame().get_motion_data()
    gyro = np.asarray([gyro_frame.x, gyro_frame.y, gyro_frame.z])

    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    gx.append(gyro[0])
    gy.append(gyro[1])
    gz.append(gyro[2])

    xs, gx, gy, gz = xs[-25:], gx[-25:], gy[-25:], gz[-25:]
    ax.clear()
    ax.plot(xs, gx)
    ax.plot(xs, gy)
    ax.plot(xs, gz)

    plt.grid()
    plt.ylim(-0.10, 0.10)
    plt.legend(['Pitch', 'Yaw', 'Roll'])
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.ylabel('Gyrometer (deg)')
    plt.title('Orientation of RGB-D Camera')

try:
    while True: 
        gyro_plot = animation.FuncAnimation(fig, plot_gyro_realtime, fargs=(xs, gx, gy, gz), interval=10)
        plt.show()
finally:
    pipeline.stop()
