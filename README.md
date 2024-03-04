# RL-Obstacle-Avoidance-Car

这是我做的第一个强化学习的项目，由我自己设计任务场景和算法。作者：ZJEast

## What is it?

任务中，要求四轮小车到达指定目标并保证自己不会与障碍物发生碰撞。目标用黄色三角形代表，障碍物用红色圆圈来控制。
读者可以下载视频 [2024-03-03 2016-01-32.mkv](./2024-03-03%2016-01-32.mkv) 来参看具体的任务场景，还有模型完成训练后的效果。

![car](./car.png)

在制作这一任务的仿真环境时，主要参考了以下这些资料：

- OpenAI gym CarDynamics [https://github.com/openai/gym/blob/master/gym/envs/box2d/car_dynamics.py](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_dynamics.py)

- OpenAI gym CarRacing [https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)

- Box2D 教程 [http://www.iforce2d.net/b2dtut/](http://www.iforce2d.net/b2dtut/)

- 四轮小车动力学分析 [http://www.iforce2d.net/b2dtut/top-down-car](http://www.iforce2d.net/b2dtut/top-down-car)

- Planar Geometric Library "shapely" [https://github.com/shapely/shapely](https://github.com/shapely/shapely)

在OpenAI gym CarRacing任务的基础上进行修改，删除了原有的赛道，修改了状态观察，添加了目标和障碍物的代码实现。
对于修改后的状态观察，读者可以跳到 [./env/move_to_pose.py#L241](./env/move_to_pose.py#L241) 了解更多，它们包括但不限于

``` python

    def _base_state(self):
        v = np.array(self.car.hull.linearVelocity)
        true_speed = np.linalg.norm(v)
        state = np.array([
            true_speed,
            self.car.wheels[0].omega,
            self.car.wheels[1].omega,
            self.car.wheels[2].omega,
            self.car.wheels[3].omega,
            self.car.wheels[0].joint.angle,
            self.car.hull.angularVelocity,
        ])
        return state

```

分别代表车身的线速度、四轮各自的速度，前轮角度，车身的角速度等。除此以外，还有小车对于障碍物和目标的观察，在此不列举了。

当小车达到目标时（包括方向正确），会获得500分的奖励；当小车与障碍物发生碰撞时，获得-500分。这是一个稀疏奖励的任务。

## How it works?

对于这一个基础的经典的控制问题，人们提出过非常多方法和思路。然而对于这样的问题结合具体运动学模型来求解精确解可能是困难的，结合最优化理论来对机器人进行运动规划恐怕也是不切实际的，因为这样的问题往往不具备凸性，以至于不得不做一些简化的处理，否则无法保证计算的高效性。

近年来，强化学习在新一轮AI浪潮中也得到了发展。在这篇工作中，我们提议可以尝试深度强化学习来解决这一问题。我们参考的工作有：

- CleanRL [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)

- Stable Baselines 3 [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

- DDPG [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)

- TD3 [https://arxiv.org/abs/1802.09477](https://arxiv.org/abs/1802.09477)

- Hindsight Experience Replay [https://arxiv.org/abs/1707.01495](https://arxiv.org/abs/1707.01495)

- Prioritized Experience Replay [https://arxiv.org/abs/1511.05952v4](https://arxiv.org/abs/1511.05952v4)

然而，直接把前人的模型拿过来恐怕还不能直接解决这个问题，我们必须再做一些额外的工作。
