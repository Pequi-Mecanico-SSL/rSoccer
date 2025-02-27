import numpy as np


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

        f = lambda x: np.isnan(x).any() or np.isinf(x).any()
        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])
        p3 = np.array([obj3.x, obj3.y])

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos_theta = np.dot(vec1, vec2)/ (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        return np.sin(theta), np.cos(theta), theta/np.pi

    def _get_2dots_angle_between(self, obj1, obj2):
        """Retorna o angulo formado pelas retas que ligam o obj1 com obj2 e obj3 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        theta = np.arctan2(diff_vec[1], diff_vec[0])

        return np.sin(theta), np.cos(theta), theta/np.pi
    
    def _get_dist_between(self, obj1, obj2):
        """Retorna a distância formada pela reta que liga o obj1 com obj2"""

        p1 = np.array([obj1.x, obj1.y])
        p2 = np.array([obj2.x, obj2.y])

        diff_vec = p1 - p2
        
        max_dist = np.linalg.norm([self.xmax - self.xmin, self.ymax - self.ymin])
        dist = np.linalg.norm(diff_vec)

        return np.clip(dist / max_dist, 0, 1)
    

def show_reward(reward_func, robot='blue_0'):
    def wrapper(*args, **kwargs):
        reward = reward_func(*args, **kwargs)
        print(f"{reward_func.__name__} - {robot} {reward[robot]}")
        return reward
    return wrapper
