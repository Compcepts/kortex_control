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

from scipy.interpolate import CubicSpline

import multiprocessing as mp

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


ENVIRONMENTS = 8


time_step = 1.0/500.0
runtime = 4.0
run_steps = round(runtime/time_step)
envs = []
for e in range(ENVIRONMENTS):
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0,0,-9.8)
        pb.setRealTimeSimulation(0)
        pb.setTimeStep(time_step)
        envs.append(pb)

bc_demo = bc.BulletClient(connection_mode=p.GUI)
bc_demo.setAdditionalSearchPath(pybullet_data.getDataPath())
bc_demo.setGravity(0,0,-9.8)
bc_demo.setRealTimeSimulation(0)
bc_demo.setTimeStep(time_step)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("using device:", device)
# print(torch.cuda.get_device_name(device))
# torch.multiprocessing.set_start_method('spawn')

manager = mp.Manager()
procs = manager.list()
for i in range(ENVIRONMENTS):
        procs.append(i)
results = manager.list()

replay_buffer = manager.list()
buff_ptr = manager.list()
buff_ptr.append(0)
max_size = capacity




def f(x):
        return x

def f_act(x):
        return torch.tanh(x)*50.0

class Robot:
        def __init__(self, pb):
                self.arm = pb.loadURDF("gen3.urdf", [0,0,0], useFixedBase=True)
                self.link_cnt = int(pb.getNumJoints(self.arm))-1
                self.pb = pb
                self.acc_err = 0.0
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

        @torch.no_grad()
        def cross(self, m2, cross_w):
                child = copy.deepcopy(self)
                for i in range(len(self.layers)):
                        if random.random() > cross_w:
                                child.layers[i] = copy.deepcopy(m2.layers[i])
                return child

        @torch.no_grad()
        def mut(self):
                strength = random.random()+1e-6
                mut_dim = round((len(self.layers)-1)*random.random())
                sw = self.layers[mut_dim].weight.shape
                sb = self.layers[mut_dim].bias.shape
                self.layers[mut_dim].weight[:,:] += torch.normal(0.0,strength,size=(sw[0],sw[1]))
                self.layers[mut_dim].bias[:] += torch.normal(0.0,strength,size=(sb[0],))

class ReplayBuffer:
        def __init__(self, buff, ptr, max_size):
                self.storage = buff
                self.max_size = max_size
                self.ptr = ptr

        def push(self, data):
                if len(self.storage) == self.max_size:
                        self.storage[int(self.ptr[0])] = data
                        self.ptr[0] = (self.ptr[0] + 1) % self.max_size
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



        # @torch.no_grad()
        # def mut(self):
                # self.actor.mut()

        # @torch.no_grad()
        # def cross(self, m2, cross_w):
                # child = copy.deepcopy(self)
                # child.actor = self.actor.cross(m2.actor, cross_w)
                # return child

        def update(self, actor, num_iteration, update_actor_target):

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
                                actor_loss = - self.critic_1(torch.cat([state, actor(state)],1)).mean()
                                print(f'actor loss: {actor_loss.item()}')

                                # Optimize the actor
                                actor.opt.zero_grad()
                                actor_loss.backward()
                                actor.opt.step()

                                if update_actor_target:
                                        for param, target_param in zip(actor.parameters(), self.actor_target.parameters()):
                                                target_param.data.copy_(((1- tau) * target_param.data) + tau * param.data)

                                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                                        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                                        target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

                                self.num_actor_update_iteration += 1
                self.num_critic_update_iteration += 1
                self.num_training += 1


@torch.no_grad()
def obj_f_mp(actor, robot, traj, iters, j):
        err = 0.0
        links = robot.link_cnt
        robot.set_joint_state(traj[0,:])
        x = np.empty((links*4,))
        new_x = np.empty((links*4,))
        for step in range(run_steps):
                curr_err = 0.0
                state = robot.joint_state()
                x[0:links*2] = state
                x[links*2:links*4] = traj[step,:]
                taus = actor(torch.tensor(x).unsqueeze(0).float().to(device))
                robot.apply_torques(taus.cpu().detach().numpy().flatten())
                robot.step()
                new_state = robot.joint_state()
                new_x[0:links*2] = new_state
                new_x[links*2:links*4] = traj[min(step+1,run_steps-1),:]
                curr_err = -np.linalg.norm(new_state-traj[step,:])
                err += curr_err
                done = 0
                if step == run_steps - 1:
                        done = 1
                REPLAY_BUFFER.push((x, new_x, taus.squeeze(0).cpu().detach().numpy(), curr_err, float(done)))
        procs.append(j)
        results[iters] = err




@torch.no_grad()
def obj_f(actor, robot, traj):
        links = robot.link_cnt
        robot.set_joint_state(traj[0,:])
        x = np.empty((links*4,))
        for step in range(run_steps):
                state = robot.joint_state()
                x[0:links*2] = state
                x[links*2:links*4] = traj[step,:]
                taus = actor(torch.tensor(x).unsqueeze(0).float().to(device))
                robot.apply_torques(taus.cpu().detach().numpy().flatten())
                robot.step()
                time.sleep(time_step)
        print(robot.joint_state()-traj[-1,:])



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
for e in envs:
        r = Robot(e)
        r.unlock()
        robots.append(r)

robot_demo = Robot(bc_demo)
robot_demo.unlock()

pop = 8
gen = 10000
sub_gen = 15
cross = 0.3
mut = 0.2

elit_lim = 3

cross_keep = 0.55


td3_struct = TD3(robots[0].link_cnt*4, robots[0].link_cnt)
net_struct = [4,[robots[0].link_cnt*4,75,75,75,robots[0].link_cnt]]

pool = []

for e in range(pop):
    m = Network(net_struct[0],net_struct[1],f_act,opt.AdamW).to(device)
    pool.append([m,0.0])
    results.append(0.0)


best_model = None
best_score = -np.inf

for g in range(gen):

        # device = torch.device("cpu")
        for sg in range(sub_gen):
                t = gen_traj()

                # eval
                tic = time.perf_counter()
                iters = 0
                while iters < pop:
                        if len(procs) > 0:
                                j = procs.pop()
                                mp.Process(target=obj_f_mp,
                                           args=(pool[iters][0],
                                                 robots[j],
                                                 t,
                                                 iters,
                                                 j,)).start()
                                iters += 1
                        time.sleep(0.001)
                while(len(procs)) != ENVIRONMENTS:
                        pass
                toc = time.perf_counter()
                print(f'elapsed time: {toc-tic}')

                for i in range(pop):
                        pool[i][1] = results[i]

                pool.sort(reverse=True, key=lambda x: x[1])
                if pool[0][1] > best_score:
                        best_score = pool[0][1]
                        best_model = copy.deepcopy(pool[0][0])

                noncrossed = list(range(round(pop*(1-cross)),pop))
                for i in range(elit_lim,pop):
                        # if random.random() < cross:
                        if len(noncrossed) > 0:
                                mut_i = random.sample([j for j in range(pop) if j < round(pop*(1-cross))],1)[0]
                                if pool[mut_i][1] < pool[i][1]:
                                        child = pool[i][0].cross(pool[mut_i][0], cross_keep)
                                else:
                                        child = pool[i][0].cross(pool[mut_i][0], 1.0-cross_keep)
                                ch_index = noncrossed[-1]
                                pool[ch_index][0] = child
                                noncrossed.remove(ch_index)

                        if random.random() < mut:
                                pool[i][0].mut()

                print(f'gen: {g} - best: {best_score}, {pool[0][1]:.4f} ... {pool[-1][1]:.4f}')

        # device = torch.device("cuda")
        tic = time.perf_counter()
        for i in range(pop):
                td3_struct.update(pool[i][0],num_iterations,i==0)


        obj_f(pool[0][0],robot_demo,t)


for e in envs:
        e.disconnect()



























'''

robot = Robot()


# not necessary but helps conceptualize whats happening
state_types = ["position", "velocity"]


EPOCHS = 20000
DATA_SIZE = 20000000
TRAIN_SIZE = 50000

joints = robot.link_cnt
states = int(joints*len(state_types))


model = Network(4, [states,250,300,350,int(states**2)+joints])



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

                p.stepSimulation()

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


p.disconnect()

'''
