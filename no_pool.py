import numpy as np
from env.move_to_pose import MoveToPose
import time
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from env.move_to_pose import MoveToPose
from model import *

class parse_args:

    def __init__(self):
    # fmt: off
        self.exp_name = "ddpg"
        self.seed = 1
        self.torch_deterministic = True
        self.cuda = True
        self.track = False
        self.env_id = "car"
        self.total_timesteps = 5000000
        self.learning_rate = 3e-4
        self.buffer_size = 1e6
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 512
        self.exploration_noise = 0.1
        self.learning_starts = 25e3
        self.policy_frequency = 2
        self.noise_clip = 0.5
        # fmt: on


class ReplayBuffer:
    def __init__(self, dims=(14, 3, 1, 14, 1, 1), max_size=1e6):
        """
        observations
        actions
        rewards
        next_observations
        dones
        results
        """
        self.buffer = [np.zeros((int(max_size), dim)) for dim in dims]
        self.max_size = int(max_size)
        self.size = 0
        self.cursor = 0
    
    def add(self, transition):
        for buffer, value in zip(self.buffer, transition):
            buffer[self.cursor] = value
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def extend(self, tajectory):
        for t in tajectory:
            self.add(t)
    
    def repeat(self, idx):
        for buffer in self.buffer:
            buffer[self.cursor] = buffer[idx]
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return [buffer[idx] for buffer in self.buffer], idx
    

class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self, dims=(14, 3, 1, 14, 1, 1), max_size=1000000):
        super().__init__(dims, max_size)
        self.priority = np.zeros((int(max_size), ))
    
    def set_priority(self, idx, priority):
        self.priority[idx] = priority
    
    def add(self, transition):
        if self.size < self.max_size:
            ReplayBuffer.add(self, transition)
            return
        idx = np.random.randint(0, self.size, size=128)
        priority = self.priority[idx]
        cursor = idx[priority.argmin()]
        for buffer, value in zip(self.buffer, transition):
            buffer[cursor] = value
        self.priority[cursor] = priority.mean()


def random_goal():
    r1, r2, r3 = np.random.random((3,))
    x = -(20*r2) * np.sin(np.pi*(2*r1 - 1))
    y = (20*r2) * np.cos(np.pi*(2*r1 - 1))
    a = np.pi*(2*r1 - 1) + np.pi*(2*r3 - 1)
    return x, y, a


def random_obstacle():
    r1, r2 = np.random.random((2,))
    x = -(15*r2) * np.sin(np.pi*(2*r1 - 1))
    y = (15*r2) * np.cos(np.pi*(2*r1 - 1))
    r = np.random.uniform(1.5, 6)
    return x, y, r

def tajectory_result(tj, gamma):
    results = [-500.0] * (len(tj) + 1)
    for i, t in enumerate(reversed(tj)):
        results[len(tj) - i - 1] = t[2] + (1 - t[4]) * gamma * results[len(tj) - i]
    tj2 = [tuple(list(t) + [r]) for t, r in zip(tj, results)]
    return tj2


class NoObstacleAgent(object):

    def __init__(self, args: parse_args, env: MoveToPose, filename):
        self.args = args
        self.filename = filename
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        self.writer = writer

        self.env = env
        self.test_env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.low = low = -1.0
        self.high = high = 1.0
        hidden_dim = 512
        reward_dim = env.reward_dim
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        self.device = device

        actor = Actor(state_dim, action_dim, low, high, hidden_dim).to(device)
        qf1 = QNetwork(state_dim, action_dim, hidden_dim, reward_dim).to(device)
        qf1_target = QNetwork(state_dim, action_dim, hidden_dim, reward_dim).to(device)
        qf2 = QNetwork(state_dim, action_dim, hidden_dim, reward_dim).to(device)
        qf2_target = QNetwork(state_dim, action_dim, hidden_dim, reward_dim).to(device)
        target_actor = Actor(state_dim, action_dim, low, high, hidden_dim).to(device)
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        # rb = ReplayBuffer(dims=(14, 3, 1, 14, 1, 1), max_size=1e6)
        rb = PriorityReplayBuffer(dims=(14, 3, 1, 14, 1, 1), max_size=1e6)

        self.actor = actor
        self.qf1 = qf1
        self.qf1_target = qf1_target
        self.qf2 = qf2
        self.qf2_target = qf2_target
        self.target_actor = target_actor
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.rb = rb

        self.global_step = 0
    
    def new_goal(self):
        return random_goal()
    
    def new_obstacle(self):
        return None
        
    def env_reset(self, goal_=None, obstacle_=None, sub_goal=None, env=None):
        if env is None:
            env = self.env
        if goal_ is None:
            goal_ = random_goal()
        # if obstacle_ is None:
        #     obstacle_ = random_obstacle()
        while True:
            obs, info = env.reset(goal_, obstacle_, sub_goal)
            if obs is None:
                obstacle_ = None
                goal_ = random_goal()
                obs, info = env.reset(goal_, obstacle_, sub_goal)
            if obs is not None:
                obstacle_ = info["obstacle"]
                return obs, info, goal_, obstacle_
            else:
                obstacle_ = random_obstacle()
                goal_ = random_goal()
    
    def save(self, name=None, check=False):
        args = self.args
        if name is None:
            name = self.filename
        torch.save({
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }, f"{name}.pth")
    
    def load(self, name, map_location=None):
        if map_location is None:
            map_location = self.device
        dct = torch.load(f"{name}.pth", map_location=map_location)
        self.actor.load_state_dict(dct["actor"])
        self.qf1.load_state_dict(dct["qf1"])
        self.qf1_target.load_state_dict(dct["qf1_target"])
        self.qf2.load_state_dict(dct["qf2"])
        self.qf2_target.load_state_dict(dct["qf2_target"])
        self.target_actor.load_state_dict(dct["target_actor"])
        self.q_optimizer.load_state_dict(dct["q_optimizer"])
        self.actor_optimizer.load_state_dict(dct["actor_optimizer"])
    
    def replay_record(self, goal_, obstacle_, record, check=False):
        # record [(action, attitude), ....]
        x, y = record[-1][1][:2]
        if check and np.hypot(x, y) < 10.0 and len(record) > 300:
            return -500.0
        obs, info, goal_, obstacle_ = self.env_reset(goal_, obstacle_)
        tj = []
        for action, _ in record:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            if not info.get("error", False):
                tj.append((obs, action, reward, next_obs, terminated))
            obs = next_obs
            if terminated or truncated:
                if info['episode']['r'] > 0.0:
                    self.add_tajectory(tj)
                return info['episode']['r']
        return info['episode']['r']
    
    def add_tajectory(self, tj):
        tj = tajectory_result(tj, self.args.gamma)
        self.rb.extend(tj)
    
    def actor_action(self, obs, noise=None):
        if noise is None:
            noise = self.args.exploration_noise
        with torch.no_grad():
            action = self.target_actor(torch.Tensor(obs).to(self.device))
            action += torch.normal(0, self.actor.action_scale * self.args.exploration_noise)
            action = action.cpu().numpy().clip(self.low, self.high)
        return action
    
    def train(self, total_step=None):
        if total_step is None:
            total_step = self.args.total_timesteps
        try:
            start_time = time.time()
            acc = [False]
            start_global_step = global_step = self.global_step

            while global_step < total_step:
                goal_ = self.new_goal()
                obstacle_ = self.new_obstacle()
                obs, info, goal_, obstacle_ = self.env_reset(goal_, obstacle_)
                record = []
                tj = []

                for _ in range(400):
                    # ALGO LOGIC: put action logic here
                    if global_step < self.args.learning_starts:
                        action = self.env.action_space.sample()
                    else:
                        action = self.actor_action(obs)

                    # TRY NOT TO MODIFY: execute the game and log data.
                    next_obs, reward, terminated, truncated, info = self.env.step(action)

                    # TRY NOT TO MODIFY: record rewards for plotting purposes

                    if not info.get("error", False):
                        tj.append((obs, action, reward, next_obs, terminated))
                        record.append((action, info["attitude"]))

                    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                    obs = next_obs

                    if terminated or truncated:
                        break

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                total_reward = info["episode"]["r"]
                self.writer.add_scalar("charts/episodic_return", total_reward, global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                acc.append(total_reward >= 0.0)
                if len(acc) > 100:
                    acc.pop(0)
                
                acc1 = sum(acc) / len(acc)
                
                print(f"global_step={global_step}, acc={acc1}")
                self.writer.add_scalar("charts/acc1", acc1, global_step)

                self.add_tajectory(tj)
                # TODO:添加额外训练数据
                self.on_terminated(goal_, obstacle_, record, total_reward, tj)

                    
                for _ in range(100):
                    global_step = self.global_step
                    self.global_step += 1

                    self.on_step(global_step)

                    # ALGO LOGIC: training.
                    if global_step > self.args.learning_starts:
                        # batch_size = 4096
                        batch_size = self.args.batch_size
                        device = self.device
                        rb = self.rb
                        sample, index = rb.sample(batch_size)
                        observations      = torch.FloatTensor(sample[0]).to(device)
                        actions           = torch.FloatTensor(sample[1]).to(device)
                        rewards           = torch.FloatTensor(sample[2]).to(device)
                        next_observations = torch.FloatTensor(sample[3]).to(device)
                        dones             = torch.FloatTensor(sample[4]).to(device)
                        results           = torch.FloatTensor(sample[5]).to(device)

                        reward_dim = self.env.reward_dim
                        B = dones.shape[0]
                        with torch.no_grad():
                            next_state_actions = self.target_actor(next_observations)
                            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
                            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
                            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                            next_q_value = rewards + (1 - dones) * self.args.gamma * (min_qf_next_target)
                            next_q_value = torch.maximum(next_q_value, results)

                        bound = 10.0
                        qf1_a_values = self.qf1(observations, actions)
                        qf2_a_values = self.qf2(observations, actions)

                        if isinstance(rb, PriorityReplayBuffer):
                            with torch.no_grad():
                                priority = (next_q_value - qf1_a_values).abs() + (next_q_value - qf2_a_values).abs()
                                priority = priority.flatten().cpu().numpy()
                                rb.set_priority(index, priority)

                        qf1_loss = F.mse_loss(qf1_a_values, (torch.clamp(next_q_value - qf1_a_values, -bound, bound) + qf1_a_values).detach())
                        qf2_loss = F.mse_loss(qf2_a_values, (torch.clamp(next_q_value - qf2_a_values, -bound, bound) + qf2_a_values).detach())
                        qf_loss = qf1_loss + qf2_loss

                        # optimize the model
                        self.q_optimizer.zero_grad()
                        qf_loss.backward()
                        self.q_optimizer.step()

                        if global_step % self.args.policy_frequency == 0:
                            actor_loss = self.on_actor_update(observations)

                            # update the target network
                            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

                        if global_step % 100 == 0:
                            self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                            self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                            self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                            self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                            self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                            print("SPS:", int((global_step - start_global_step) / (time.time() - start_time)))
                            self.writer.add_scalar("charts/SPS", int((global_step - start_global_step) / (time.time() - start_time)), global_step)

        finally:
            self.writer.close()
            self.save()
    
    def on_step(self, global_step):
        pass
    
    def on_actor_update(self, observations):
        pi = self.actor(observations)
        qf1_pi = self.qf1(observations, pi)
        qf2_pi = self.qf2(observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = - min_qf_pi.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss
        
    def on_terminated(self, goal, obstacle, record, total_reward, tj):
        self.her(goal, obstacle, record, total_reward)
    
    def her(self, goal, obstacle, record, total_reward):
        if len(record) > 5 and total_reward < 0.0:
            _, goal_2 = record[np.random.randint(len(record), size=4).max()]
            return self.replay_record(goal_2, obstacle, record) > 0
        return False
    
    def test(self, times):
        res = []
        for i in range(times):
            obs, info, goal_, obstacle_ = self.env_reset(self.new_goal(), self.new_obstacle(), env=self.test_env)
            while True:
                action = self.actor_action(obs, noise=0.0)
                next_obs, reward, terminated, truncated, info = self.test_env.step(action)
                obs = next_obs
                if terminated or truncated:
                    print(f"no. {i}, score{info['episode']['r']}")
                    res.append(info['episode']['r'])
                    break
        res = np.array(res)
        return res, sum(res > 0) / times, sum(res <= 0) / times

    def view(self):
        while True:
            obs, info, goal_, obstacle_ = self.env_reset(self.new_goal(), self.new_obstacle())
            while True:
                action = self.actor_action(obs, noise=0.0)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = next_obs
                if terminated or truncated:
                    break
    
    def on_step(self, global_step):
        if global_step % 30000 == 0:
            _, acc_test, _ = self.test(100)
            self.save(f"./results/{self.filename}_{global_step}_{acc_test}")
            self.writer.add_scalar("charts/acc_test", acc_test, global_step)


class ObstalceAgent(NoObstacleAgent):

    def __init__(self, args: parse_args, env: MoveToPose, filename, no_obstacle: NoObstacleAgent):
        super().__init__(args, env, filename)
        self.no_obstacle = no_obstacle

    def new_obstacle(self):
        return random_obstacle()
    
    def on_terminated(self, goal_, obstacle_, record, total_reward, tj_):
        self.her(goal_, obstacle_, record, total_reward)
        self.error_explore(goal_, obstacle_, record, total_reward)
    
    def error_explore(self, goal_, obstacle_, record, total_reward):
        if obstacle_ is not None and total_reward < 0:
            obstacle_re = obstacle_
            obs, info, _, _ = self.no_obstacle.env_reset(goal_, None)
            record = []
            while True:
                action = self.no_obstacle.actor_action(obs)
                next_obs, reward, terminated, truncated, info = self.no_obstacle.env.step(action)
                if not info.get("error", False):
                    record.append((action, info["attitude"]))
                obs = next_obs
                if terminated or truncated:
                    break
            result = self.replay_record(goal_, obstacle_re, record)
            if result > 0:
                return True
            for _ in range(16):
                obstacle_ = obstacle_re
                obs, info, _, _ = self.no_obstacle.env_reset(random_goal(), None)
                record = []
                sub_limit = np.random.randint(30, 80)
                for _ in range(sub_limit):
                    action = self.no_obstacle.actor_action(obs)
                    next_obs, reward, terminated, truncated, info = self.no_obstacle.env.step(action)
                    if not info.get("error", False):
                        record.append((action, info["attitude"]))
                    obs = next_obs
                    if terminated or truncated:
                        break
                
                obs, info, goal_, obstacle_ = self.env_reset(goal_, obstacle_)
                tj = []
                terminated = truncated = False
                for action, _ in record:
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    if not info.get("error", False):
                        tj.append((obs, action, reward, next_obs, terminated))
                    obs = next_obs
                    if terminated or truncated:
                        if info['episode']['r'] > 0.0:
                            self.add_tajectory(tj)
                            return True
                        break
                while True:
                    action = self.actor_action(obs)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    if not info.get("error", False):
                        tj.append((obs, action, reward, next_obs, terminated))
                    obs = next_obs
                    if terminated or truncated:
                        sub_reward = info['episode']['r']
                        print(f"sub_mode = {sub_reward}")
                        if sub_reward > 0.0:
                            self.add_tajectory(tj)
                            return True
                        break
        return False
