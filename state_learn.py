import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import time
import math

import matplotlib.pyplot as plt

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


import copy
import time

#### ENVIRONMENT INIT

tau = 0.005
target_update_interval = 1
iteration = 5
gamma = 0.99
capacity = 100000
num_iterations = 100
batch_size = 500
policy_delay = 2

CONNECT = 'GUI'

time_step = 1.0/500.0
runtime = 4.0
run_steps = round(runtime/time_step)
pb = None

if CONNECT == 'GUI':
    pb = bc.BulletClient(connection_mode=p.GUI)
else:
    pb = bc.BulletClient(connection_mode=p.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.setGravity(0,0,-9.8)
pb.setRealTimeSimulation(0)
pb.setTimeStep(time_step)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
print(torch.cuda.get_device_name(device))


def f(x):
    return x

class Robot:
    def __init__(self, pb):
        self.arm = pb.loadURDF("gen3.urdf", [0,0,0], useFixedBase=True)
        self.link_cnt = int(pb.getNumJoints(self.arm))-1
        self.pb = pb
        self.unlock()

    def unlock(self):
        maxForce = 0
        mode = p.VELOCITY_CONTROL
        for i in range(pb.getNumJoints(self.arm)):
                self.pb.setJointMotorControl2(self.arm,i,targetVelocity=0,controlMode=mode,force=maxForce)

    def joint_state(self):
        state = self.pb.getJointStates(self.arm, [i for i in range(self.link_cnt)])
        pos = []
        vel = []
        for l in range(self.link_cnt):
            pos.append(state[l][0])
            vel.append(state[l][1])
        return pos, vel

    def set_joint_state(self, states):
        self.acc_err = 0.0
        for l in range(self.link_cnt):
            self.pb.resetJointState(self.arm, l, states[l], states[self.link_cnt+l])

    def ik(self, x):
        return self.pb.calculateInverseKinematics(self.arm,
                                            self.link_cnt,
                                            x)


    def apply_torques(self, taus):
        self.pb.setJointMotorControlArray(self.arm,
                                [i for i in range(self.link_cnt)],
                                controlMode=p.TORQUE_CONTROL,
                                forces=taus)
    def step(self):
        self.pb.stepSimulation()
    

class Network(nn.Module):
    def __init__(self, layers, nodes, out_act, opt, lr=0.001):
        super(Network, self).__init__()
        self.layers = []
        for l in range(layers):
            self.layers.append(nn.Linear(nodes[l],nodes[l+1]).to(device))
        self.opt = opt(self.parameters(),lr=lr)
        self.out_act = out_act

    def parameters(self):
        p = []
        for l in self.layers:
            p.append(l.weight)
            p.append(l.bias)
        return p

    def forward(self, x):
        out = x
        for l in self.layers[:-1]:
            out = F.leaky_relu(l.to(device)(out))
        out = self.out_act(self.layers[-1].to(device)(out))
        return out



robot = Robot(pb)


# not necessary but helps conceptualize whats happening
state_types = ["position", "velocity"]


EPOCHS = 20000
DATA_SIZE = 20000000
TRAIN_SIZE = 50000

joints = robot.link_cnt
states = int(joints*len(state_types))


model = Network(4, [states,250,300,350,int(states**2)+joints],f,opt.AdamW)



x = torch.empty((DATA_SIZE,states)).to(device)
tu = torch.empty((DATA_SIZE,joints)).to(device)
y = torch.empty((DATA_SIZE,states)).to(device)
## collecting dynamics data

RUNTIME = 50000

for i in range(int(DATA_SIZE/RUNTIME)):

    print(f'\rcurrent run: {i}', end='')

    start_state = []
    for s in range(states):
        start_state.append(random.uniform(-10,10))
        if s >= joints:
            start_state[s] = start_state[s]*0.1
    robot.set_joint_state(start_state)
    robot.unlock()

    taus = []
    for s in range(joints):
        # if i % 2 == 1:
                # taus.append((-0.15*float(s)+1.6)*np.random.normal(0,1.25))
        # else:
        taus.append(0.0)

    for j in range(RUNTIME):


        # for s in range(joints):
                # if i % 2 == 1:
                        # taus[s] += np.random.normal(0,0.05)
        robot.apply_torques(taus)

        tu[(i*RUNTIME)+j,0:joints] = torch.tensor(taus)

        pos, vel = robot.joint_state()
        x[(i*RUNTIME)+j,0:joints] = torch.tensor(pos)
        x[(i*RUNTIME)+j,joints:states] = torch.tensor(vel)

        robot.step()

        pos_p, vel_p = robot.joint_state()
        d_pos = (np.array(pos_p)-np.array(pos))/time_step
        d_vel = (np.array(vel_p)-np.array(vel))/time_step
        y[(i*RUNTIME)+j,0:joints] = torch.tensor(d_pos)
        y[(i*RUNTIME)+j,joints:states] = torch.tensor(d_vel)



print('')
lr = 0.0005
opt = optim.Adam(model.param_list(),lr=lr)


for e in range(EPOCHS):

    v = torch.randperm(TRAIN_SIZE)

    x_t = x[v,:]
    tu_t = tu[v,:]
    y_t = y[v,:]

    AB = model.forward(x_t)
    A, B = torch.split(AB, [states**2,joints],dim=1)
    
    y_hat = (A.view(TRAIN_SIZE,states,states) @ x_t.unsqueeze(2)).squeeze(2)
    y_hat[:,joints:states] += B * tu_t

    opt.zero_grad()
    loss = model.loss(y_hat, y_t) + 0.00001*(model.l1_val())
    print(f'\repoch: {e}, loss: {loss.item()}',end='')
    loss.backward()
    opt.step()


print("")
input("enter to continue")

TEST_STEPS = 10000

start_state = []
for s in range(states):
    start_state.append(random.uniform(-10,10))
    if s >= joints:
        start_state[s] = start_state[s]*0.1
robot.set_joint_state(start_state)

for steps in range(TEST_STEPS):

    with torch.no_grad():
        AB = model.forward(torch.tensor(start_state).unsqueeze(0).to(device))
        A, B = torch.split(AB, [states**2,joints],dim=1)

        y_hat = (A.view(1,states,states) @ torch.tensor(start_state).unsqueeze(0).unsqueeze(2).to(device)).squeeze(2)
        # y_hat[:,joints:states] += B * tu_t

        for j in range(states):
            if j < joints:
                start_state[j] += y_hat[0,j]*time_step
            else:
                start_state[j] += y_hat[0,j]*time_step

            robot.set_joint_state(start_state)


pb.disconnect()


