import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pyquaternion import Quaternion

link = np.array([50, 40, 35, 30])  # Robot Link Length Parameter
link = link / float(np.sum(link))
quats = [Quaternion() for i in range(link.size)]
init_pos = np.array([[0, 0, 0]] + [[mlen, 0, 0] for mlen in link])
pos = np.array(init_pos.copy())

fig = plt.figure()
ax = fig.gca(projection='3d')


def local_pos(idx):
    quats[idx-1] = quats[idx-1].normalised
    return quats[idx-1].rotate(init_pos[idx])


def forward_kinematics(idx):  # idx > 0
    for i in range(idx, link.size+1):
        pos[i] = pos[i-1] + local_pos(i)


def ik_update_quat(target, idx: int):
    now_to_end, now_to_tar = np.array(pos[-1] - pos[idx]), np.array(target - pos[idx])
    len_now_to_end, len_now_to_tar = np.linalg.norm(now_to_end), np.linalg.norm(now_to_tar)
    now_to_end, now_to_tar = now_to_end / len_now_to_end, now_to_tar / len_now_to_tar
    rot_axis = np.cross(now_to_end, now_to_tar)
    len_rot_axis = np.linalg.norm(rot_axis)
    rot_angle = math.asin(len_rot_axis)
    delta_quat = Quaternion(axis=rot_axis / len_rot_axis, angle=rot_angle)
    quats[idx] = (delta_quat * quats[idx]).normalised


def ik(target, max_iter=60, eps=0.001):
    for miter in range(0, max_iter):
        for idx in range(link.size-1, -1, -1):
            ik_update_quat(target, idx)
            forward_kinematics(idx+1)
            norml = np.linalg.norm(pos[-1] - target)
            if norml < eps:
                print("Iteration %d, eps = %f" % (miter, norml))
                return
    print("IK failed.")


def mydraw(target):
    ik(target)
    plt.cla()
    for i in range(link.size):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], "-o")
        print(np.linalg.norm(pos[i+1] - pos[i]) - link[i])
    ax.scatter(target[0], target[1], target[2])
    plt.show()


def rand_pos3(radius, min_ratio=0.3, max_ratio=0.7) -> np.ndarray:
    _r = np.random.uniform(min_ratio * radius, max_ratio * radius, 1)
    _theta = np.random.uniform(0, np.pi, 1)
    _phi = np.random.uniform(0, 2 * np.pi, 1)
    return np.array([_r * np.sin(_theta) * np.cos(_phi),
                     _r * np.sin(_theta) * np.sin(_phi),
                     _r * np.cos(_theta)]).squeeze()


if __name__ == "__main__":
    forward_kinematics(1)
    mydraw(rand_pos3(1))
