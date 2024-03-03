# -*- coding: utf-8 -*-

from typing import Optional, Union
import numpy as np
from . import car_racing as cr
from gym import spaces
from Box2D.b2 import fixtureDef, polygonShape, circleShape
from shapely.geometry import Polygon, Point
from gym import spaces
from shapely.ops import nearest_points

# 检查要生成的障碍物和其他物体是否相交，如果相交则对障碍物做调整
def check_overlap(obstacle, shapely_obj_list, gap=0.0):
    x, y, r = obstacle
    r += gap
    ok = False
    cnt = 0
    while not ok:
        ok = True
        if cnt > 100:
            return None
        cnt += 1
        for poly in shapely_obj_list:
            obstacle = Point(x, y)
            distance = poly.distance(obstacle)
            if distance >= r + 0.1:
                continue
            ok = False
            p1, p2 = nearest_points(poly, obstacle)
            v = np.array((p1.x - p2.x, p1.y - p2.y))
            nv = np.linalg.norm(v)
            if nv <= 0:
                a = np.pi * 2 * (np.random.random() * 2 -1)
                v = np.array((np.cos(a), np.sin(a)))
            else:
                v = v/nv
            if distance <= 0.0:
                go = nv + r + 0.1
            else:
                go = - (r + 0.1 - nv)
            dx, dy = go * v + 0.1 * (np.random.random(2) * 2 -1)
            x, y = x + dx, y + dy
    return (x, y, r - gap)


class MoveToPose(cr.CarRacing):
    """
        继承的参数
        render_mode: Optional[str] = None, 
        verbose: bool = False, 
        lap_complete_percent: float = 0.95, 
        domain_randomize: bool = False, 
        continuous: bool = True,

        生成的障碍物和其他物品的最少距离
        gap = 0.0,
        返回的状态是否包含之前几帧的状态
        precious_obs = 0,
        返回的reward是标量还是向量
        vector_reward = False
    """
    def __init__(self, 
        render_mode: Optional[str] = None, 
        verbose: bool = False, 
        lap_complete_percent: float = 0.95, 
        domain_randomize: bool = False, 
        continuous: bool = True,
        gap = 1.5,
        precious_obs = 0,
        vector_reward = False,
        r_inflate = 0.0
        ):
        super().__init__(render_mode, verbose, lap_complete_percent, domain_randomize, continuous)
        obs_dim = 14
        if precious_obs > 0:
            obs_dim = (obs_dim) * (precious_obs + 1)
        self.observation_space = spaces.Box(-100, 100, (obs_dim,))
        self.action_space = spaces.Box(-1, 1, (3,))
        self.gap = gap
        self.precious_obs = precious_obs
        self.vector_reward = vector_reward
        self.reward_dim = 1
        if vector_reward:
            self.reward_dim = 3
        self.r_inflate = r_inflate
    
    """
        目标
        goal, 
        障碍物
        obstacle=None, 
        子目标
        sub_goal=None, 
        是否限时
        time_limit=True, 
        继承的参数
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    """
    def reset(self, goal, obstacle=None, sub_goal=None, time_limit=True, seed: Optional[int] = None, options: Optional[dict] = None):
        self._reset(seed=seed, options=options)

        self._create_goal(goal)
        self._create_sub_goal(sub_goal)
        self.obstacle = obstacle
        self.error = False
        self.hit = False
        self._create_obstacle(obstacle)

        self.state = None
        self.time_limit = time_limit
        self.time_cnt = 0
        self.episodic_return = 0.0
        self.episodic_length = 0
        self.state_queue = []

        if not self.error:
            state, step_reward, terminated, truncated, info = self.step(None)
            info["obstacle"] = self.obstacle
            return state, info
        else:
            return None, {"error": True}
    
    # 创建目标
    def _create_goal(self, goal):
        self.goal = goal
        x, y, a = self.goal
        vertices = [
            (   0,  3),
            ( 1.5, -3),
            (-1.5, -3),
        ]
        area = self.world.CreateStaticBody(
            position = (x, y),
            angle = a,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=vertices)
            )
        )
        # 添加碰撞检测
        self._setup_detect(area)
        # 允许其他对象穿过这个对象，同时触发碰撞事件
        area.fixtures[0].sensor = True
        # 目标在Box2d里面的对象
        self.area = area
    
    # 创建子目标，子目标在探索时发挥作用
    def _create_sub_goal(self, sub_goal):
        if sub_goal is None:
            sub_goal = self.goal
        x, y, a = sub_goal
        # 子目标没有设置碰撞检测
        self.sub_goal = (x, y, a)
        self.sub_goal_state = None
        self.sub_goal_vertices = [
            (   0,  3),
            ( 1.5, -3),
            (-1.5, -3),
        ]
    
    # 创建障碍物，保证障碍物不与其他物体重叠
    def _create_obstacle(self, obstacle):
        self.obstacle = obstacle
        if obstacle is None:
            return
        # create subgoal and obstacle
        x, y, a = self.goal
        self.obstacle = obstacle # x, y, r
        assert hasattr(self, "obstacle")
        w = 3 + 0.2
        h = 6 + 0.2
        vertices = [
            (-w/2, -h/2),
            ( w/2, -h/2),
            ( w/2,  h/2),
            (-w/2,  h/2),
        ]
        goal = np.array([self.area.GetWorldVector(p) for p in vertices])
        goal[:, 0] += self.area.position[0]
        goal[:, 1] += self.area.position[1]
        # 目标的shapely对象
        goal = Polygon(goal)
        # 小车的shapely对象
        car = Polygon(vertices)
        self.error = False
        # 使用shapely看看他们有没有交叉，防止出BUG
        obstacle = check_overlap(obstacle, [goal, car], gap=self.gap)
        self.obstacle = obstacle
        if obstacle is None:
            return

        # 在box2d里创建障碍物的对象
        x, y, r = self.obstacle
        area = circleShape()
        area.radius = r
        area = self.world.CreateStaticBody(
            position = (x, y),
            fixtures = fixtureDef(
                shape=area
            )
        )
        # 设置碰撞事件
        self._setup_detect(area)
        self.obstacle_area = area
        # 还没有发生碰撞
        self.hit = False
    
    # 删除所有对象
    def _destroy(self):
        if hasattr(self, "area") and self.area is not None:
            self.world.DestroyBody(self.area)
            self.area = None
        if hasattr(self, "obstacle_area") and self.obstacle_area is not None:
            self.world.DestroyBody(self.obstacle_area)
            self.obstacle_area = None
        return super()._destroy()
    
    # 把场景用pygame画出来
    def _render_scene(self, zoom, translation, angle):
        super()._render_scene(zoom, translation, angle)
        # goal
        self._draw_area_poly(self.area, np.array([255]*3 + [100]), 
            zoom, translation, angle)
        # obstacle
        if self.obstacle is not None:
            self._draw_area_circle(self.obstacle_area, np.array([200, 100, 100, 100]), 
                zoom, translation, angle)
        # subgoal
        self._draw_area_poly_(self.sub_goal, self.sub_goal_vertices,
            np.array([255, 255, 100, 100]), zoom, translation, angle)
    
    # 碰撞响应
    def begin_contact(self, contact):
        super().begin_contact(contact)
        body_id, fixtureA, fixtureB = self.check_contact(contact)
        if self.obstacle is not None:
            if body_id == self.obstacle_area.body_id:
                self.hit = True
    
    # 小车的运动状态
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
    
    # 目标相对于小车的位置
    def _relative(self, goal):
        a = self.car.hull.angle
        xg, yg, ag = goal
        beta = (ag - a + np.pi) % (2*np.pi) - np.pi
        rho, alpha = self._rho_alpha(xg, yg)
        return rho, alpha, beta
    
    # 目标的直角坐标转为相对的极坐标
    def _rho_alpha(self, xg, yg):
        x, y = self.car.hull.position
        a = self.car.hull.angle
        
        xl = (xg - x) * np.cos(a) + (yg - y) * np.sin(a)
        yl = -(xg - x) * np.sin(a) + (yg - y) * np.cos(a)
        alpha = -np.arctan2(xl, yl)

        rho = np.hypot((xg - x), (yg - y))
        if np.abs(alpha) > np.pi/2:
            rho = -rho
            alpha = np.sign(alpha) * (np.pi - np.abs(alpha))
        
        return rho, alpha
    
    # 无用,遗留的
    def old_state(self, state):
        rho, alpha, beta = state[7: 10]
        obs, rho_2, alpha_2, theta = state[10:]
        if rho < 0:
            alpha = np.sign(alpha) * (np.pi - np.abs(alpha))
        if rho_2 < 0:
            alpha_2 = np.sign(alpha_2) * (np.pi - np.abs(alpha_2))
            rho_2 = - rho_2
        if obs > 0.0:
            return np.hstack((state[:7], [rho, alpha, beta, rho_2, alpha_2, theta]))
        else:
            return np.hstack((state[:7], [rho, alpha, beta]))

    def step(self, action: Union[np.ndarray, int]):
        """
        reward 由三部分组成
        0 时间 刹车 倒车
        1 完成目标
        2 碰撞 走得太远
        """
        state, step_reward, terminated, truncated, info = super().step(action)

        x, y = self.car.hull.position
        a = self.car.hull.angle
        # 小车在世界坐标下的姿态
        info["attitude"] = (x, y, a)

        v = np.array(self.car.hull.linearVelocity)
        v = v / (np.linalg.norm(v) + 1e-5)

        # 目标相对位置
        rho, alpha, beta = self._relative(self.goal)
        state = np.hstack((self._base_state(), [rho, alpha, beta]))

        brake = 0.0
        if action is not None:
            brake = action[2]

        # 时间流逝,倒车,刹车惩罚
        step_reward = [-1.0]
        step_reward[-1] -= 0.1 * float(-v[0]*np.sin(a) + v[1]*np.cos(a) < 0)
        step_reward[-1] -= 0.1 * float(brake > 0.4)
    
        # 达到目标的奖励
        step_reward.append(0.0)
        if np.abs(rho) < 0.5 and np.abs(beta) < np.pi/180*5 :
            if action is not None:
                step_reward[-1] += 500.0
            terminated = True
        
        # 时间限制
        if self.time_limit:
            self.time_cnt += 1
            if self.time_cnt >= 400:
                truncated = True
        
        # 子目标状态,放在info里面
        rho, alpha, beta = self._relative(self.sub_goal)
        self.sub_goal_state = np.hstack((state[:7], [rho, alpha, beta], [0.0]*4))
        info["sub_goal_state"] = self.sub_goal_state

        # 障碍物的相对位置
        x, y = self.car.hull.position
        a = self.car.hull.angle
        if self.obstacle is not None:
            xo, yo, r = self.obstacle
            r += self.r_inflate
            rho, alpha = self._rho_alpha(xo, yo)
            theta = np.arcsin(np.clip(r / np.abs(rho), 0.0, 1.0))
            state = np.hstack((state, [1.0, rho * (1 - np.sin(theta)), alpha, theta]))
        else:
            state = np.hstack((state, [0.0] * 4))

        # 碰撞的惩罚
        info["hit"] = self.hit
        step_reward.append(0.0)
        if self.hit:
            step_reward[-1] -= 500.0
            terminated = True
        
        # 走太远走出场景也有惩罚
        x, y = self.car.hull.position
        if np.hypot(x, y) > 100.0:
            step_reward[-1] -= 500.0
            terminated = True
        
        info["error"] = self.error
        if self.error:
            self.episodic_return = 0.0
            self.episodic_return = 0
            truncated = True
            terminated = False
        
        self.episodic_return += sum(step_reward)
        self.episodic_length += 1
        self.state = state
        info["episode"] = {
            "r": self.episodic_return,
            "l": self.episodic_length
        }
        
        # 状态队列
        que = self.state_queue
        if len(que) == 0:
            que = [state]
        if len(que) < self.precious_obs + 1:
            que = [que[0]] * (self.precious_obs + 1 - len(que)) + que
        que.pop(0)
        que.append(state)
        self.state_queue = que
        que = np.array(que).flatten()

        # 向量reward还是标量reward
        if self.vector_reward:
            step_reward = np.array(step_reward)
        else:
            step_reward = sum(step_reward)
        return que, step_reward, terminated, truncated, info