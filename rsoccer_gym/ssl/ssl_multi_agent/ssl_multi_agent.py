import numpy as np
from gymnasium.spaces import Box, Dict
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
import random
from collections import namedtuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import OrderedDict
from rsoccer_gym.Entities.Robot import Robot
import copy
from gymnasium.wrappers import RecordVideo
import random
from rsoccer_gym.Utils.Utils import Geometry2D
from rsoccer_gym.judges.ssl_judge import Judge

class SSLMultiAgentEnv(SSLBaseEnv, MultiAgentEnv):
    default_players = 3
    def __init__(self,
        judge: Judge,
        init_pos,
        field_type=2, 
        fps=40,
        match_time=40,
        stack_observation=8,
        render_mode='human',
        dense_rewards = {},
        sparse_rewards = {},
        possession_radius_scale=3,
        direction_change_threshold=1
    ):

        self.n_robots_blue = min(len(init_pos["blue"]), 3)
        self.n_robots_yellow = min(len(init_pos["yellow"]), 3)
        self.score = {'blue': 0, 'yellow': 0}
        self.render_mode = render_mode
        super().__init__(
            field_type=field_type, 
            n_robots_blue=self.n_robots_blue,
            n_robots_yellow=self.n_robots_yellow, 
            time_step=1/fps,
            render_mode=render_mode
        )
        self.dense_rewards = dense_rewards
        self.sparse_rewards = sparse_rewards
        self.geometry = Geometry2D(-self.field.length/2, self.field.length/2, -self.field.width/2, self.field.width/2)
        self.goal_template = namedtuple('goal', ['x', 'y'])
        self.ball_template = namedtuple('ball', ['x', 'y', 'v_x', 'v_y'])
        agent_ids_blue = [f'blue_{i}'for i in range(self.n_robots_blue)]
        agent_ids_yellow = [f'yellow_{i}'for i in range(self.n_robots_yellow)]
        self._agent_ids = [*agent_ids_blue, *agent_ids_yellow]
        self.max_ep_length = int(match_time*fps)
        self.fps = fps
        self.last_actions = {
            **{f'blue_{i}': np.zeros(4) for i in range(self.n_robots_blue)}, 
            **{f'yellow_{i}': np.zeros(4) for i in range(self.n_robots_yellow)}
        }

        self.stack_observation = stack_observation
        # Limit robot speeds
        self.max_v = 1.5
        self.max_w = 10
        self.kick_speed_x = 3.0

        self.init_pos = init_pos

        self.obs_size = 77 #obs[f'blue_0'].shape[0]
        #self.obs_size = 33
        self.act_size = 4

        self.actions_bound = {"low": -1, "high": 1}

        blue = {f'blue_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=self.actions_bound["low"], high=self.actions_bound["high"], shape=(self.act_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.action_space =  Dict(**blue, **yellow)

        blue = {f'blue_{i}': Box(low=-self.NORM_BOUNDS - 0.001, high=self.NORM_BOUNDS + 0.001, shape=(self.stack_observation * self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}': Box(low=-self.NORM_BOUNDS - 0.001, high=self.NORM_BOUNDS + 0.001, shape=(self.stack_observation * self.obs_size, ), dtype=np.float64) for i in range(self.n_robots_yellow)}
        self.observation_space = Dict(**blue, **yellow)

        self.observations = {
            **{f'blue_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': np.zeros(self.stack_observation * self.obs_size, dtype=np.float64) for i in range(self.n_robots_yellow)}
        }

        init_frame = self._get_initial_positions_frame(42)
        self.judge = judge(
            field=self.field, 
            initial_frame=init_frame, 
            possession_radius_scale=possession_radius_scale, 
            direction_change_threshold=direction_change_threshold
        )
        self.judge_last_status, self.judge_last_info = None, None
        self.judge_status, self.judge_info = self.judge.judge(init_frame)

    def _get_commands(self, actions):
        commands = []
        for i in range(self.n_robots_blue):
            robot_actions = actions[f'blue_{i}'].copy()
            angle = self.frame.robots_blue[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))
            cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)
        
        for i in range(self.n_robots_yellow):
            robot_actions = actions[f'yellow_{i}'].copy()
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(robot_actions, np.deg2rad(angle))

            cmd = Robot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta, kick_v_x=self.kick_speed_x if robot_actions[3] > 0 else 0.)
            commands.append(cmd)

        return commands
    
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""

        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle),\
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x,v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta


    def _calculate_reward_done(self):
        self.judge_last_status = self.judge_status
        self.judge_last_info = self.judge_info
        self.judge_status, self.judge_info = self.judge.judge(self.frame)

        done = {'__all__': False}
        truncated = {'__all__': False}

        reward_agents = {
            **{f"blue_{idx}":  0 for idx in range(self.n_robots_blue)},
            **{f"yellow_{idx}": 0 for idx in range(self.n_robots_yellow)},
        }
        for weight, reward_func, list_attr in self.dense_rewards:
            kwargs = {attr: getattr(self, attr) for attr in list_attr}
            reward_result = reward_func(
                self.field, self.frame, self.last_frame, 
                left="blue", right="yellow", 
                **kwargs
            )

            for agent, reward in reward_result.items():
                reward_agents[agent] += weight * reward

        if self.judge_status == "RIGHT_GOAL":
            done = {'__all__': True}
            self.score['blue'] += 1

            reward_agents.update({f'blue_{i}': self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f'yellow_{i}': -self.sparse_rewards.get("GOAL_REWARD", 0)for i in range(self.n_robots_yellow)})
        
        elif self.judge_status == "LEFT_GOAL":
            done = {'__all__': True}
            self.score['yellow'] += 1

            reward_agents.update({f'blue_{i}': -self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({f'yellow_{i}': self.sparse_rewards.get("GOAL_REWARD", 0) for i in range(self.n_robots_yellow)})
        
        elif self.judge_status in ["RIGHT_BOTTOM_LINE", "LEFT_BOTTOM_LINE", "LOWER_SIDELINE", "UPPER_SIDELINE"]:
            last_touch = self.judge_info["last_touch"]
            #reward_agents.update({last_touch: self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({"blue": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})
            reward_agents.update({"yellow": self.sparse_rewards.get("OUTSIDE_REWARD", 0) for i in range(self.n_robots_blue)})

            initial_pos_frame: Frame = self._get_initial_positions_frame(42)
            self.rsim.reset(initial_pos_frame)
            self.frame = self.rsim.get_frame()
        
        return reward_agents, done, truncated

    def reset(self, seed=42, options={}):
        self.steps = 0
        self.last_frame = None
        self.sent_commands = None

        # Close render window
        # del(self.view)
        # self.view = None

        initial_pos_frame: Frame = self._get_initial_positions_frame(seed)
        self.rsim.reset(initial_pos_frame)

        # Get frame from simulator
        self.frame = self.rsim.get_frame()

        blue = {f'blue_{i}': {} for i in range(self.n_robots_blue)}
        yellow = {f'yellow_{i}':{} for i in range(self.n_robots_yellow)}
        self.score = {'blue': 0, 'yellow': 0}

        self._frame_to_observations()

        return self.observations.copy(), {**blue, **yellow}
  
    def _get_initial_positions_frame(self, seed):
        '''Returns the position of each robot and ball for the initial frame'''
        #np.random.seed(seed)

        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        places = KDTree()

        pos_frame: Frame = Frame()

        if isinstance(self.init_pos["ball"], list): 
            pos_frame.ball = Ball(x=self.init_pos["ball"][0], y=self.init_pos["ball"][1])
        else:
            pos_frame.ball = Ball(x=random.uniform(-2, 2), y=random.uniform(-1.2, 1.2))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        min_dist = 0.2
        for i in range(self.n_robots_blue):
            pos = self.init_pos['blue'][i+1] 
            while places.get_nearest(pos[:2])[1] < min_dist:
                pos = (x(), y(), theta()) 
            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])


        for i in range(self.n_robots_yellow):
            pos = self.init_pos['yellow'][i+1] 
            while places.get_nearest(pos[:2])[1] < min_dist:
                pos = (x(), y(), theta()) 

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=pos[2])

        return pos_frame

    def _get_pos(self, obj):

        x = self.norm_pos(obj.x)
        y = self.norm_pos(obj.y)
        v_x = self.norm_v(obj.v_x)
        v_y = self.norm_v(obj.v_y)
        
        theta = np.deg2rad(obj.theta) if hasattr(obj, 'theta') else None
        sin = np.sin(theta) if theta is not None else None
        cos = np.cos(theta) if theta is not None else None
        theta = np.arctan2(sin, cos)/np.pi if theta is not None else None
        v_theta = self.norm_w(obj.v_theta) if theta is not None else None
        #tan = np.tan(theta) if theta else None

        return x, y, v_x, v_y, sin, cos, theta, v_theta

    def inverted_robot(self, robot):
        return Robot(
            x=-robot.x, 
            y=robot.y, 
            theta= 180 - robot.theta if robot.theta < 180 else 540 - robot.theta, 
            v_x=-robot.v_x, 
            v_y=-robot.v_y, 
            v_theta=-robot.v_theta
        )

    def _frame_to_observations(self):

        # print("=====================================OBSERVATION===================================================")
        f = lambda x: " ".join([f"{i:.2f}" for i in x])
        for i in range(self.n_robots_blue):
            robot = self.frame.robots_blue[i] 
            robot_action = self.last_actions[f'blue_{i}']
            allys = [self.frame.robots_blue[j] for j in range(self.n_robots_blue) if j != i]
            allys_actions = [self.last_actions[f'blue_{j}'] for j in range(self.n_robots_blue) if j != i]
            advs = [self.frame.robots_yellow[j] for j in range(self.n_robots_yellow)]

            ball = self.ball_template(x=self.frame.ball.x, y=self.frame.ball.y, v_x=self.frame.ball.v_x, v_y=self.frame.ball.v_y)

            goal_adv = self.goal_template(x=   0.2 + self.field.length/2, y=0)
            goal_ally = self.goal_template(x= -0.2 - self.field.length/2, y=0)

            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)
            self.observations[f'blue_{i}'] = np.delete(self.observations[f'blue_{i}'], range(len(robot_obs)))
            self.observations[f'blue_{i}'] = np.concatenate([self.observations[f'blue_{i}'], robot_obs], axis=0, dtype=np.float64)

            # if i == 1:
            #     print(f"blue_{i}")
            #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
            #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")

        for i in range(self.n_robots_yellow):
            robot = self.inverted_robot(self.frame.robots_yellow[i])
            robot_action = self.last_actions[f'yellow_{i}']
            allys = [self.inverted_robot(self.frame.robots_yellow[j]) for j in range(self.n_robots_yellow) if j != i]
            allys_actions = [self.last_actions[f'yellow_{j}'] for j in range(self.n_robots_yellow) if j != i]
            advs = [self.inverted_robot(self.frame.robots_blue[j]) for j in range(self.n_robots_blue)]

            ball = self.ball_template(x=-self.frame.ball.x, y=self.frame.ball.y, v_x=-self.frame.ball.v_x, v_y=self.frame.ball.v_y)

            goal_adv = self.goal_template(x=  -(-0.2 - self.field.length/2), y=0)
            goal_ally = self.goal_template(x= -( 0.2 + self.field.length/2), y=0)
            
            robot_obs = self.robot_observation(robot, allys, advs, robot_action, allys_actions, ball, goal_adv, goal_ally)

            self.observations[f'yellow_{i}'] = np.delete(self.observations[f'yellow_{i}'], range(len(robot_obs)))
            self.observations[f'yellow_{i}'] = np.concatenate([self.observations[f'yellow_{i}'], robot_obs], axis=0, dtype=np.float64)

            # if i == 1:
            #     print(f"\nyellow_{i}")
            #     print(f"\tX={robot.x}\tY={robot.y}\tTheta={robot.theta}\tVx={robot.v_x}\tVy={robot.v_y}\tVtheta={robot.v_theta}")
            #     print(f"\t pos: {f(robot_obs[:14])} \n\t ori: {f(robot_obs[14:32])} \n\t dist: {f(robot_obs[32:40])} \n\t ang: {f(robot_obs[40:64])} \n\t last_act: {f(robot_obs[64:76])} \n\t time_left: {robot_obs[76]}")

    def robot_observation(self, robot, allys, adversaries, robot_action, allys_actions, ball, goal_adv, goal_ally):

        positions = []
        orientations = []
        dists = []
        angles = []
        last_actions = np.array([robot_action] + allys_actions).flatten()

        x_b, y_b, *_ = self._get_pos(ball)
        sin_BG_al, cos_BG_al, theta_BG_al = self.geometry._get_2dots_angle_between(goal_ally, ball)
        sin_BG_ad, cos_BG_ad, theta_BG_ad = self.geometry._get_2dots_angle_between(goal_adv, ball)
        dist_BG_al = self.geometry._get_dist_between(ball, goal_ally)
        dist_BG_ad = self.geometry._get_dist_between(ball, goal_adv)

        x_r, y_r, *_, sin_r, cos_r, theta_r, _  = self._get_pos(robot)
        sin_BR, cos_BR, theta_BR = self.geometry._get_2dots_angle_between(ball, robot)
        dist_BR = self.geometry._get_dist_between(ball, robot)

        positions.append([x_r, y_r])
        orientations.append([sin_r, cos_r, theta_r])
        #orientations.append([theta_r])
        dists.append([dist_BR, dist_BG_al, dist_BG_ad])
        angles.append([
            sin_BR, cos_BR, theta_BR, 
            sin_BG_al, cos_BG_al, theta_BG_al, 
            sin_BG_ad, cos_BG_ad, theta_BG_ad
        ])

        for ally in allys:
            x_al, y_al, *_, sin_al, cos_al, theta_al, _ = self._get_pos(ally)
            sin_AlR, cos_AlR, theta_AlR = self.geometry._get_2dots_angle_between(ally, robot)
            ally_dist = self.geometry._get_dist_between(ally, robot)
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            #orientations.append([theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])
        
        for i in range(self.default_players - len(allys) - 1):
            print("não é pra entrar aqui")
            x_al, y_al, sin_al, cos_al, theta_al = 0, 0, 0, 0, 0
            sin_AlR, cos_AlR, theta_AlR = 0, 0, 0
            ally_dist = 0
            positions.append([x_al, y_al])
            orientations.append([sin_al, cos_al, theta_al])
            # orientations.append([theta_al])
            dists.append([ally_dist])
            angles.append([sin_AlR, cos_AlR, theta_AlR])

        
        for adv in adversaries:
            x_adv, y_adv, *_,  sin_adv, cos_adv, theta_adv, _ = self._get_pos(adv)
            sin_AdR, cos_AdR, theta_AdR = self.geometry._get_2dots_angle_between(adv, robot)
            adv_dist = self.geometry._get_dist_between(adv, robot)
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            #orientations.append([theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        for i in range(self.default_players - len(adversaries)):
            x_adv, y_adv, sin_adv, cos_adv, theta_adv = 0, 0, 0, 0, 0
            sin_AdR, cos_AdR, theta_AdR = 0, 0, 0
            adv_dist = 0
            positions.append([x_adv, y_adv])
            orientations.append([sin_adv, cos_adv, theta_adv])
            #orientations.append([theta_adv])
            dists.append([adv_dist])
            angles.append([sin_AdR, cos_AdR, theta_AdR])

        positions.append([x_b, y_b])

        positions = np.concatenate(positions)
        orientations = np.concatenate(orientations)
        dists = np.concatenate(dists)
        angles = np.concatenate(angles)
        time_left = [(self.max_ep_length - self.steps)/self.max_ep_length]

        #print(f"len_pos: {len(positions)} \t len_ori: {len(orientations)} \t len_dist: {len(dists)} \t len_ang: {len(angles)} \t len_last_act: {len(last_actions)} \t len_time_left: {len(time_left)}")
        robot_obs = np.concatenate([positions, orientations, dists, angles, last_actions, time_left], dtype=np.float64)
        # robot_obs = np.concatenate([positions, last_actions, time_left], dtype=np.float64)
        return robot_obs
    
    def _friendly_observation(self, obs):
        friendly_obs = {'pos': {}, 'ori': {}, 'dist': {}, 'ang': {}, 'last_act': {}, 'time_left': {}}

        friendly_obs["pos"]["robot"]   = obs[ 0: 2]
        friendly_obs["pos"][f"ally_0"] = obs[ 2: 4]
        friendly_obs["pos"][f"ally_1"] = obs[ 4: 6]
        friendly_obs["pos"][f"adv_0"]  = obs[ 6: 8]
        friendly_obs["pos"][f"adv_1"]  = obs[ 8:10]
        friendly_obs["pos"][f"adv_2"]  = obs[10:12]
        friendly_obs["pos"]["ball"]    = obs[12:14]

        friendly_obs["ori"]["robot"]   = obs[14:17]
        friendly_obs["ori"][f"ally_0"] = obs[17:20]
        friendly_obs["ori"][f"ally_1"] = obs[20:23]
        friendly_obs["ori"][f"adv_0"]  = obs[23:26]
        friendly_obs["ori"][f"adv_1"]  = obs[26:29]
        friendly_obs["ori"][f"adv_2"]  = obs[29:32]

        friendly_obs["dist"]["dist_BR"]    = obs[32:33]
        friendly_obs["dist"]["dist_BG_al"] = obs[33:34]
        friendly_obs["dist"]["dist_BG_ad"] = obs[34:35]
        friendly_obs["dist"]["ally_0"]     = obs[35:36]
        friendly_obs["dist"]["ally_1"]     = obs[36:37]
        friendly_obs["dist"]["adv_0"]      = obs[37:38]
        friendly_obs["dist"]["adv_1"]      = obs[38:39]
        friendly_obs["dist"]["adv_2"]      = obs[39:40]

        friendly_obs["ang"]["ball_robot"]     = obs[40:43]
        friendly_obs["ang"]["ball_goal_ally"] = obs[43:46]
        friendly_obs["ang"]["ball_goal_adv"]  = obs[46:49]
        friendly_obs["ang"]["ally_0"]         = obs[49:52]
        friendly_obs["ang"]["ally_1"]         = obs[52:55]
        friendly_obs["ang"]["adv_0"]          = obs[55:58]
        friendly_obs["ang"]["adv_1"]          = obs[58:61]
        friendly_obs["ang"]["adv_2"]          = obs[61:64]

        friendly_obs["last_act"]["robot"]   = obs[64:68]
        friendly_obs["last_act"][f"ally_0"] = obs[68:72]
        friendly_obs["last_act"][f"ally_1"] = obs[72:76]

        friendly_obs["time_left"] = obs[76:77]

        return friendly_obs
    
    def step(self, action):
        self.steps += 1
        # Join agent action with environment actions
        commands = self._get_commands(action)
        # Send command to simulator
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_actions = action.copy()

        # Get Frame from simulator
        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        # Calculate environment observation, reward and done condition
        self._frame_to_observations()
        reward, done, truncated = self._calculate_reward_done()

        if self.steps >= self.max_ep_length:
            done = {'__all__': False}
            truncated = {'__all__': True}

        infos = {
            **{f'blue_{i}': {} for i in range(self.n_robots_blue)},
            **{f'yellow_{i}': {} for i in range(self.n_robots_yellow)}
        }  

        if done.get("__all__", False) or truncated.get("__all__", False):
            for i in range(self.n_robots_blue):
                infos[f'blue_{i}']["score"] = self.score.copy()    

            for i in range(self.n_robots_yellow):
                infos[f'yellow_{i}']["score"] = self.score.copy()
        
        return self.observations.copy(), reward, done, truncated, infos

class SSLMultiAgentEnv_record(RecordVideo, MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_ids = self.env._agent_ids