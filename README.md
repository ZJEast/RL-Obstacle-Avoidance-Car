# RL-Obstacle-Avoidance-Car

这是我做的第一个强化学习的项目，由我自己设计任务场景和算法。作者：ZJEast

## What is it?

任务中，要求四轮小车到达指定目标并保证自己不会与障碍物发生碰撞。目标用黄色三角形代表，障碍物用红色圆圈来控制。
读者可以下载视频 [2024-03-03 2016-01-32.mkv](./2024-03-03%2016-01-32.mkv) 来参看具体的任务场景，还有模型完成训练后的效果。

![car](./car.png)

在制作这一任务的仿真环境时，主要参考了以下这些资料：

- OpenAI gym CarDynamics[https://github.com/openai/gym/blob/master/gym/envs/box2d/car_dynamics.py](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_dynamics.py)

- OpenAI gym CarRacing[https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)

- Box2D 教程 [http://www.iforce2d.net/b2dtut/](http://www.iforce2d.net/b2dtut/)

- 四轮小车动力学分析 [http://www.iforce2d.net/b2dtut/top-down-car](http://www.iforce2d.net/b2dtut/top-down-car)

- Planar Geometric Library "shapely" [https://github.com/shapely/shapely](https://github.com/shapely/shapely)
