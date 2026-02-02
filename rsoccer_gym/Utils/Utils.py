import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


# Base on baselines implementation
class OrnsteinUhlenbeckAction(object):
    def __init__(self, action_space, theta=0.17, dt=0.025, x0=None):
        self.theta = theta
        self.mu = (action_space.high + action_space.low) / 2
        self.sigma = (action_space.high - self.mu) / 2
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )
    

class Geometry2D():
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax    

    def _get_3dots_angle_between(self, obj1, obj2, obj3):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])
        p3 = np.array([obj3['x'], obj3['y']])

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1['x'], obj1['y']])
        p2 = np.array([obj2['x'], obj2['y']])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.xmax - self.xmin, self.ymax - self.ymin])
        dist = np.linalg.norm(diff_vec)

        return np.clip(dist / max_dist, 0, 1)
    
    def _invert_coordinates(self, obj, on_x = False, on_y = False):
        if on_x:
            obj['x'] = -obj['x']
            obj['theta'] = 180 - obj['theta'] if obj['theta'] < 180 else 540 - obj['theta']
        
        if on_y:
            obj['y'] = -obj['y']
            obj['theta'] = -obj['theta']

        return obj


def show_reward(reward_func, robot='blue_0'):
    def wrapper(*args, **kwargs):
        reward = reward_func(*args, **kwargs)
        print(f"{reward_func.__name__} - {robot} {reward[robot]}")
        return reward
    return wrapper


def decorator_observations(obs_func):
    def wrapper(n_blue, n_yellow, raw_observations, field_info, kwargs):
        results = {}
        ball = raw_observations["ball"]
        robot_colors = ['blue'] * n_blue + ['yellow'] * n_yellow
        geometry = Geometry2D(
            -field_info["length"]/2, 
            field_info["length"]/2, 
            -field_info["width"]/2, 
            field_info["width"]/2
        )
        mapper_inverter = {
            'blue': lambda x: x,
            'yellow': lambda x: geometry._invert_coordinates(x, on_x=True)
        } # AI will see yellow robots as if it were blue. Invertion trick

        for i, (color_main, color_adv) in enumerate(zip(robot_colors, robot_colors[::-1])):
            idx = i % n_blue
            inverter = mapper_inverter[color_main] 

            n_main, n_adv = (n_blue, n_yellow) if color_main == 'blue' else (n_yellow, n_blue)
            main_robots = [inverter(robot) for name, robot in raw_observations.items() if "blue" in name]
            adv_robots = [inverter(robot) for name, robot in raw_observations.items() if "yellow" in name]

            main = main_robots[idx] 
            allys = [main_robots[j] for j in range(n_main) if j != idx]
            advs = [adv_robots[j] for j in range(n_adv)]
            results[f"{color_main}_{idx}"] = obs_func(f"{color_main}_{idx}", main, allys, advs, ball, **kwargs)
        
        return results
        
    return wrapper
    
class StackWrapper(MultiAgentEnv):
    def __init__(self, base_env, stack_size, observation_funcs, *args, **kwargs):
        #super().__init__()
        self.base_env = base_env
        self.stack_size = stack_size
        self.observation_funcs = observation_funcs

        n_blue = self.base_env.n_robots_blue
        n_yellow = self.base_env.n_robots_yellow
        self.reset()
        self.observation_space = Dict(
            **{f'blue_{i}': Box(low=-1, high=1, shape=(self.stack_size * self.obs_size,), dtype=np.float64) for i in range(n_blue)},
            **{f'yellow_{i}':Box(low=-1, high=1, shape=(self.stack_size * self.obs_size,), dtype=np.float64) for i in range(n_yellow)}
        )
    
    def __getattr__(self, attr):
        return getattr(self.base_env, attr)

    
    def _reset_stack(self, obs_size):
        stack_obs = {
            **{f'blue_{i}': np.zeros(self.stack_size * obs_size, dtype=np.float64) for i in range(self.base_env.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(self.stack_size * obs_size, dtype=np.float64) for i in range(self.base_env.n_robots_yellow)}
        }

        return stack_obs


    def _update_stack(self, stack_obs, observations):
        for agent, obs in observations.items():       
            stack_obs[agent] = np.concatenate([
                np.delete(
                    stack_obs[agent], 
                    range(len(obs))
                ), # remove oldest observation
                obs
            ], axis=0, dtype=np.float64)
        
        return stack_obs
    

    def _calculate_observations(self, raw_observations):
        observations = {
            **{f'blue_{i}': np.zeros(0, dtype=np.float64) for i in range(self.base_env.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(0, dtype=np.float64) for i in range(self.base_env.n_robots_yellow)}
        }
        obs_size = 0
        for observation_func, class_attrs in self.observation_funcs:
            kwargs = {
                attr: (
                    getattr(self.base_env, attr, None) or
                    getattr(self, attr, None)
                )
                for attr in class_attrs
            }
            obs_result = observation_func(
                self.base_env.n_robots_blue, 
                self.base_env.n_robots_yellow, 
                raw_observations,
                self.base_env.field_info, 
                kwargs=kwargs
            )

            for agent, obs in obs_result.items():
                observations[agent] = np.hstack([observations[agent], obs])
            obs_size += len(obs)

        return observations, obs_size


    def reset(self, *args, **kwargs):
        self.last_actions = {
            **{f'blue_{i}': np.zeros(4) for i in range(self.base_env.n_robots_blue)}, 
            **{f'yellow_{i}': np.zeros(4) for i in range(self.base_env.n_robots_yellow)}
        }

        raw_observations, info = self.base_env.reset(*args, **kwargs)
        observations, obs_size = self._calculate_observations(raw_observations)
        stack_obs = self._reset_stack(obs_size)
        self.stack_obs = self._update_stack(stack_obs, observations)

        self.obs_size = obs_size
        return self.stack_obs, info
    

    def step(self, action):

        raw_observations, reward, done, truncated, info = self.base_env.step(action)
        observations, _ = self._calculate_observations(raw_observations)
        self.stack_obs = self._update_stack(self.stack_obs, observations)

        self.last_actions = action.copy()
        return self.stack_obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)

        
