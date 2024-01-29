import gym
import cv2

import numpy as np
import os, glob

from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from PIL import Image
from gym_minigrid.wrappers import *
from matplotlib import pyplot as plt

max_step_per_episode = 4500


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class GymVectorEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=0,
            h=0,
            w=0,
            life_done=False,
            sticky_action=False,
            p=0, save_img=None):
        super(GymVectorEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id)
        if 'minigrid' in env_id.lower():
            self.env = FlatObsWrapper(self.env)


        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn


        self.last_action = 0
        self.save_image_dir = save_img
        if self.save_image_dir:
            if not os.path.isdir(self.save_image_dir):
                os.mkdir(self.save_image_dir)



        self.clear_imgs()
        self.reset()

    def clear_imgs(self):
        if self.save_image_dir:
            if os.path.isdir(self.save_image_dir):
                files = glob.glob(f'{self.save_image_dir}/*')
                for f in files:
                    os.remove(f)
            print(f"done prepare img dir {self.save_image_dir}")

    def run(self):
        super(GymVectorEnvironment, self).run()
        done = False
        while True:
            if done:
                self.clear_imgs()

            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()
            if self.save_image_dir and self.steps%1==0:
                img = self.env.render(mode='rgb_array')
                iimg_id =  f'{self.steps:05d}.png'
                plt.imsave(self.save_image_dir+'/'+iimg_id, img)
                with open(self.save_image_dir+'/agent_loc.txt', 'a') as file:
                    file.write(str(self.env.agent_pos[0]))
                    file.write(" ")
                    file.write(str(self.env.agent_pos[1]))
                    file.write("\n")
                with open(self.save_image_dir+'/agent_dir.txt', 'a') as file:
                    file.write(str(self.env.agent_dir))
                    file.write("\n")


            s, reward, done, info = self.env.step(int(action))
            # print(max_step_per_episode, self.steps)
            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                # print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {} ".format(
                #     self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))
                self.reset()

            self.child_conn.send(
                [s, reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        return s

class GymVectorContEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=0,
            h=0,
            w=0,
            life_done=False,
            sticky_action=False,
            p=0, save_img=None):
        super(GymVectorContEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id)


        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn


        self.last_action = 0
        self.save_image_dir = save_img
        if self.save_image_dir:
            if not os.path.isdir(self.save_image_dir):
                os.mkdir(self.save_image_dir)



        self.clear_imgs()
        self.reset()

    def clear_imgs(self):
        if self.save_image_dir:
            if os.path.isdir(self.save_image_dir):
                files = glob.glob(f'{self.save_image_dir}/*')
                for f in files:
                    os.remove(f)
            print(f"done prepare img dir {self.save_image_dir}")

    def run(self):
        super(GymVectorContEnvironment, self).run()
        done = False
        mstep = 0
        while True:
            if done:
                self.clear_imgs()

            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()
            if self.save_image_dir and self.steps%1==0:
                img = self.env.render(mode='rgb_array')
                iimg_id =  f'{self.steps:05d}.png'
                plt.imsave(self.save_image_dir+'/'+iimg_id, img)


            s, reward, done, info = self.env.step(action)
            if isinstance(s,dict):
                s = s['observation']
                s = np.append(s, [mstep], axis=0)  # add time step feature

            # print(max_step_per_episode, self.steps)
            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.rall += reward
            self.steps += 1
            mstep += 1e-3

            if done:
                self.recent_rlist.append(self.rall)
                # print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {} ".format(
                #     self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist)))
                self.reset()

            self.child_conn.send(
                [s, reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        if isinstance(s,dict):
            s = s['observation']
            s = np.append(s, [0], axis=0)  # add time step feature
        return s

class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25, save_img=None):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.save_image_dir = save_img
        if self.save_image_dir:
            if not os.path.isdir(self.save_image_dir):
                os.mkdir(self.save_image_dir)



        self.clear_imgs()

        self.reset()
    def clear_imgs(self):
        if self.save_image_dir:
            if os.path.isdir(self.save_image_dir):
                files = glob.glob(f'{self.save_image_dir}/*')
                for f in files:
                    os.remove(f)
            print(f"done prepare img dir {self.save_image_dir}")

    def run(self):
        super(AtariEnvironment, self).run()
        done = False
        while True:
            action = self.child_conn.recv()
            if done:
                self.clear_imgs()
            if self.is_render:
                self.env.render()
            if self.save_image_dir and self.steps%1==0:
                img = self.env.render(mode='rgb_array')
                iimg_id =  f'{self.steps:05d}.png'
                plt.imsave(self.save_image_dir+'/'+iimg_id, img)

            if 'Breakout' in self.env_id:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action
            # print(action)
            s, reward, done, info = self.env.step(action)
            # print("ssss")
            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:

                self.recent_rlist.append(self.rall)
                # print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                #     self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                #     info.get('episode', {}).get('visited_rooms', {})))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))/255.0
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


