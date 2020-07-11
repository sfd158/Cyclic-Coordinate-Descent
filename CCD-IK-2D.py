import numpy as np
import matplotlib.pyplot as plt
import math

link = np.array([50, 40, 35, 30])  # Robot Link Length Parameter
angle = np.zeros((link.size,))  # Robot Initial Joint Values (degree)
init_pos = np.array([[0, 0]] + [[mlen, 0] for mlen in link])
pos = np.zeros((link.size + 1, 2))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


def rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def local_pos(idx):
    rot = rot_mat(angle[idx-1])
    return np.matmul(rot, init_pos[idx])


def forward_kinematics(idx):  # idx > 0
    for i in range(idx, link.size+1):
        pos[i] = pos[i-1] + local_pos(i)


def ik_update_angle(target, idx):
    now_to_end, now_to_tar = pos[-1] - pos[idx], target - pos[idx]
    len_now_to_end, len_now_to_tar = np.linalg.norm(now_to_end), np.linalg.norm(now_to_tar)
    norml = len_now_to_end * len_now_to_tar
    if norml < 0.0001:
        theta = 0
    else:
        sin_rot_ang = (now_to_end[0] * now_to_tar[1] - now_to_end[1] * now_to_tar[0]) / norml
        theta = math.asin(sin_rot_ang)
    angle[idx] += np.clip(-1, theta, 1)
    if angle[idx] > np.pi:
        angle[idx] -= 2 * np.pi
    elif angle[idx] < -np.pi:
        angle[idx] += 2 * np.pi


def ik(target, max_iter=60, eps=0.01):
    for miter in range(0, max_iter):
        for idx in range(link.size-1, -1, -1):
            ik_update_angle(target, idx)
            forward_kinematics(idx+1)
            norml = np.linalg.norm(pos[-1] - target)
            if norml < eps:
                print("Iteration %d, eps = %f" % (miter, norml))
                return
    print("IK failed.")


def mydraw(target):
    plt.cla()
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    ax.scatter(target[0], target[1])
    for i in range(1, link.size + 1):
        ax.plot(pos[i-1:i+1, 0], pos[i-1:i+1, 1], '-o')
    plt.show()


def onclick(event):
    forward_kinematics(1)
    target = np.array([event.xdata, event.ydata])
    ik(target)
    mydraw(target)


if __name__ == "__main__":
    fig.canvas.mpl_connect('button_press_event', onclick)
    forward_kinematics(1)
    mydraw([0, 0])
