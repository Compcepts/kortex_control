from pathlib import Path
import sys
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

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

from multi_network import MultiNetwork

from scipy.interpolate import CubicSpline

import copy
import time

#### ENVIRONMENT INIT

tau = 0.005
target_update_interval = 1
iteration = 5
gamma = 0.99
capacity = 200000
num_iterations = 500
batch_size = 256
policy_delay = 10

ROBOTS = 15


time_step = 1.0/150.0
runtime = 3.0
run_steps = round(runtime/time_step)

bc = bc.BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0,0,-9.8)
bc.setRealTimeSimulation(0)
bc.setTimeStep(time_step)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
print(torch.cuda.get_device_name(device))

replay_buffer = []
buff_ptr = 0
max_size = capacity




def f(x):
        return x

def f_act(x):
        return torch.tanh(x)*75.0

class Robot:
        def __init__(self, pb, loc=[0,0,0]):
                self.arm = pb.loadURDF("gen3.urdf", loc, useFixedBase=True)
                self.link_cnt = int(pb.getNumJoints(self.arm))-1
                self.pb = pb
                self.acc_err = 0.0
                self.unlock()

        def unlock(self):
                maxForce = 0
                mode = p.VELOCITY_CONTROL
                for i in range(self.pb.getNumJoints(self.arm)):
                        self.pb.setJointMotorControl2(self.arm,i,targetVelocity=0,controlMode=mode,force=maxForce)

        def joint_state(self):
                state = self.pb.getJointStates(self.arm, [i for i in range(self.link_cnt)])
                pos = []
                vel = []
                for l in range(self.link_cnt):
                        pos.append(state[l][0])
                        vel.append(state[l][1])
                return np.array([pos, vel]).flatten()

        def set_joint_state(self, states):
                self.acc_err = 0.0
                for l in range(self.link_cnt):
                        self.pb.resetJointState(self.arm, l, states[l], states[self.link_cnt+l])

        def ik(self, x):
                return self.pb.calculateInverseKinematics(self.arm,
                                                    self.link_cnt,
                                                    x)

        def pid_control(self, xd, xd_dot, Kp, Kd, Ki):
                curr_state = self.joint_state()
                pos_err = curr_state[0:self.link_cnt]-xd
                vel_err = curr_state[self.link_cnt:self.link_cnt*2]-xd_dot
                self.acc_err += pos_err
                taus = -Kp*(pos_err) - Kd*(vel_err) - Ki*(self.acc_err)
                taus = np.clip(taus, -50, 50)
                self.apply_torques(taus)

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

        @torch.no_grad()
        def forward_ng(self, x):
                out = x
                for l in self.layers[:-1]:
                        out = F.leaky_relu(l(out))
                out = self.out_act(self.layers[-1](out))
                return out

class ReplayBuffer:
        def __init__(self, buff, ptr, max_size):
                self.storage = buff
                self.max_size = max_size
                self.ptr = ptr

        def push(self, data):
                if len(self.storage) == self.max_size:
                        self.storage[int(self.ptr)] = data
                        self.ptr = (self.ptr + 1) % self.max_size
                else:
                        self.storage.append(data)

        def sample(self, batch_size):
                ind = np.random.randint(0, len(self.storage), size=batch_size)
                x, y, u, r, d = [], [], [], [], []

                for i in ind:
                        X, Y, U, R, D = self.storage[i]
                        x.append(np.array(X, copy=False))
                        y.append(np.array(Y, copy=False))
                        u.append(np.array(U, copy=False))
                        r.append(np.array(R, copy=False))
                        d.append(np.array(D, copy=False))

                return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

REPLAY_BUFFER = ReplayBuffer(replay_buffer,buff_ptr,max_size)

class TD3:
        def __init__(self, state_dim, action_dim):

                self.actor_target = Network(4,[state_dim,75,75,75,action_dim],f_act,opt.AdamW).to(device)
                self.critic_1 = Network(4,[state_dim+action_dim,75,75,75,1],f,opt.AdamW).to(device)
                self.critic_1_target = Network(4,[state_dim+action_dim,75,75,75,1],f,opt.AdamW).to(device)
                self.critic_2 = Network(4,[state_dim+action_dim,75,75,75,1],f,opt.AdamW).to(device)
                self.critic_2_target = Network(4,[state_dim+action_dim,75,75,75,1],f,opt.AdamW).to(device)

                # self.actor_optimizer = self.actor.opt
                self.critic_1_optimizer = self.critic_1.opt
                self.critic_2_optimizer = self.critic_2.opt

                # self.memory = Replay_buffer(capacity)
                # self.memory = replay
                self.num_critic_update_iteration = 0
                self.num_actor_update_iteration = 0
                self.num_training = 0

        def select_action(self, state):
                state = torch.tensor(state.reshape(1, -1)).float().to(device)
                return self.actor(state).cpu().data.numpy().flatten()

        def update(self, actor, num_iteration, best_index):

                for i in range(num_iteration):
                        # x, y, u, r, d = self.memory.sample(batch_size)
                        x, y, u, r, d = REPLAY_BUFFER.sample(batch_size)
                        state = torch.FloatTensor(x).to(device)
                        action = torch.FloatTensor(u).to(device)
                        next_state = torch.FloatTensor(y).to(device)
                        done = torch.FloatTensor(d).to(device)
                        reward = torch.FloatTensor(r).to(device)

                        next_action = (self.actor_target(next_state))# + noise)

                        # Compute target Q-value:
                        target_Q1 = self.critic_1_target(torch.cat([next_state, next_action],1))
                        target_Q2 = self.critic_2_target(torch.cat([next_state, next_action],1))
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

                        # Optimize Critic 1:
                        current_Q1 = self.critic_1(torch.cat([state, action],1))
                        loss_Q1 = F.mse_loss(current_Q1, target_Q)
                        self.critic_1_optimizer.zero_grad()
                        loss_Q1.backward()
                        self.critic_1_optimizer.step()

                        # Optimize Critic 2:
                        current_Q2 = self.critic_2(torch.cat([state, action],1))
                        loss_Q2 = F.mse_loss(current_Q2, target_Q)
                        self.critic_2_optimizer.zero_grad()
                        loss_Q2.backward()
                        self.critic_2_optimizer.step()
                        # Delayed policy updates:
                        if i % policy_delay == 0:
                                # Compute actor loss:
                                # actor_loss = - self.critic_1(torch.cat([state, actor(state)],1)).mean()
                                actor.loss_fn = lambda x,y: - self.critic_1(torch.cat([y, x],1)).mean()
                                # print(f'actor loss: {actor_loss.item()}')

                                loss, l = actor.train(state, state)
                                print(min(l))
                                # Optimize the actor
                                # actor.opt.zero_grad()
                                # actor_loss.backward()
                                # actor.opt.step()

                                for param, target_param in zip(actor.parameters_i(best_index), self.actor_target.parameters()):
                                        target_param.data.copy_(((1- tau) * target_param.data) + tau * param.data)

                                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                                        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                                        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                                self.num_actor_update_iteration += 1
                self.num_critic_update_iteration += 1
                self.num_training += 1


@torch.no_grad()
def cross_fn(model, ind_one, ind_two, child_ind, cross_w):
        for i in range(len(model.layers[0])):
                model.layers[child_ind][i].weight[:,:] = (model.layers[ind_one][i].weight + model.layers[ind_two][i].weight)/2.0
                model.layers[child_ind][i].bias[:] = (model.layers[ind_one][i].bias + model.layers[ind_two][i].bias)/2.0

@torch.no_grad()
def mut_fn(model, ind):
        strength = (random.random()/50.0)
        mut_dim = round((len(model.layers[0])-1)*random.random())
        model.layers[ind][mut_dim].weight += torch.normal(0.0,strength,size=model.layers[ind][mut_dim].weight.shape).to(device)
        model.layers[ind][mut_dim].bias += torch.normal(0.0,strength,size=model.layers[ind][mut_dim].bias.shape).to(device)


@torch.no_grad()
def eval_traj(actor, traj, robots, pool):
        err = []
        for e in range(actor.nets):
                err.append([e, 0.0])
        links = robots[0].link_cnt
        for robot in robots:
                robot.set_joint_state(traj[0,:])
        x = np.empty((actor.nets,links*4))
        new_x = np.empty((actor.nets,links*4))
        for step in range(run_steps):
                curr_err = [0.0]*actor.nets
                for i, robot in enumerate(robots):
                        state = robot.joint_state()
                        x[i,0:links*2] = state
                        x[i,2*links:4*links] = traj[run_steps-1,:]
                taus = actor(torch.tensor(x).float().to(device))
                shape = taus.shape[1]//actor.nets
                for i, robot in enumerate(robots):
                        robot.apply_torques(taus[i,i*shape:(i+1)*shape].cpu().detach().numpy().flatten())
                robots[0].step()
                for i, robot in enumerate(robots):
                        new_state = robot.joint_state()
                        new_x[i,0:links*2] = new_state
                        new_x[i,2*links:4*links] = traj[run_steps-1,:]
                        curr_err[i] = np.linalg.norm(new_state-traj[run_steps-1,:])
                        if abs(curr_err[i]) < 1e-8:
                                curr_err[i] = 1000.0
                        else:
                                curr_err[i] = 1.0/curr_err[i]
                        err[i][1] += curr_err[i]
                        done = 0
                        if step == run_steps - 1:
                                done = 1
                        REPLAY_BUFFER.push((x[i,:], new_x[i,:], taus[i,i*shape:(i+1)*shape].cpu().detach().numpy().flatten(), curr_err[i], float(done)))
        return err




def gen_traj():
        traj = np.empty((run_steps,14))

        a = np.arange(2)
        b = np.random.uniform(low=-np.pi,high=np.pi,size=(2,7))
        cs = CubicSpline(a,b,bc_type=((1, tuple([0.0]*7)), (1, tuple([0.0]*7))))

        for i in range(run_steps):
                t = float(i)/float(run_steps-1)
                traj[i,0:7] = cs(t)
                traj[i,7:14] = cs(t,1)


        return traj



robots = []
for r in range(ROBOTS):
        ro = Robot(bc, loc=[(r//5)*2.5,(r%5)*2.5,0.0])
        ro.unlock()
        robots.append(ro)

pop = ROBOTS
gen = 10000
sub_gen = 20
cross = 0.3
mut = 0.2

elit_lim = 0

cross_keep = 0.55


td3_struct = TD3(robots[0].link_cnt*4, robots[0].link_cnt)
net_struct = [4,[robots[0].link_cnt*4,75,75,75,robots[0].link_cnt]]

pool = []
pool_place = []

m = MultiNetwork.MultiNetwork(pop,
                              net_struct[1],
                              [F.leaky_relu,F.leaky_relu,F.leaky_relu,f_act],
                              opt.AdamW,
                              F.mse_loss,
                              device=device)


for g in range(gen):

        for sg in range(sub_gen):
                traj = gen_traj()

                # eval
                tic = time.perf_counter()
                pool = eval_traj(m, traj, robots, pool)
                toc = time.perf_counter()

                print(f'elapsed time: {toc-tic}')

                pool.sort(reverse=True, key=lambda x: x[1])

                noncrossed = list(range(round(pop*(1-cross)),pop))
                for i in range(pop):
                        if len(noncrossed) > 0:
                                mut_i = random.sample([j for j in range(pop) if j < round(pop*(1-cross))],1)[0]
                                ch_index = noncrossed[-1]
                                if pool[mut_i][1] < pool[i][1]:
                                        cross_fn(m, pool[i][0], pool[mut_i][0], pool[ch_index][0], cross_keep)
                                else:
                                        cross_fn(m, pool[i][0], pool[mut_i][0], pool[ch_index][0], 1.0-cross_keep)
                                noncrossed.remove(ch_index)

                        if random.random() < mut and i >= elit_lim:
                                mut_fn(m, pool[i][0])

                print(f'gen: {g} - best: {pool[0][1]:.4f} ... {pool[-1][1]:.4f}')

        tic = time.perf_counter()
        td3_struct.update(m,num_iterations,pool[0][0])
        toc = time.perf_counter()
        print(f'elapsed time: {toc-tic}')

        # obj_f(pool[0][0],robot_demo,t)


for e in envs:
        bc.disconnect()
