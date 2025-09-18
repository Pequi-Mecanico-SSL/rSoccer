from rSoccer.rsoccer_gym.Entities import Frame, Field, Robot
import numpy as np
import math

class Judge():
    def __init__(self, field: Frame, initial_frame: Field=None, 
                 possession_radius_scale: float=3, direction_change_threshold: float=1,
                 left="blue", kickoff=True, initial_ball_position=[0,0]
        ):
        self.field = field
        self.last_frame = None
        self.frame = initial_frame

        self.ball_possession: str = None

        self.last_touch: str = None
        self.possession_radius_scale = possession_radius_scale
        self.direction_change_threshold = direction_change_threshold #in degrees
        self.left = left

        self.initial_ball_position = initial_ball_position
        self.historical_ball_positions = set()
        self.is_kickoff = kickoff

    def judge(self, frame: Frame) -> tuple[str, dict]:
        """
        Método que executa o julgamento do juiz, verificando a posse de bola, último toque e se houve gol, lateral ou linha de fundo.
        :return: tuple - (status, infos)
            status: str - Indica o status do jogo, podendo ser "left_goal", "right_bottom_line", "sideline" ou None
            infos: dict - Informações adicionais sobre a posse de bola e último toque
        """

        self.last_frame = self.frame
        self.frame = frame

        self._update_ball_possession()
        self._update_last_touch()
        self._update_offenses()
        infos = {
            "ball_possession": self.ball_possession,
            "last_touch": self.last_touch,
            "offenses": self.offenses
        }

        goal = self._check_goal()
        bottom_line = self._check_bottom_line()
        sideline_line = self._check_sideline()

        # status = None indicando que não houve nenhum evento relevante
        status = goal or bottom_line or sideline_line or None

        return status, infos   
    

    def _check_goal(self) -> str|None:
        # Medidas já escaladas
        ball = self.frame.ball
        half_len = self.field.length/2 
        goal_top  = self.field.goal_width / 2
        goal_bottom = -self.field.goal_width / 2

        if ball.x > half_len and goal_bottom <= ball.y <= goal_top:
            return "RIGHT_GOAL"

        if ball.x < -half_len and goal_bottom <= ball.y <= goal_top:
            return "LEFT_GOAL"
        
        return None
    
    def _check_bottom_line(self) -> str|None:
        """
        Verifica se a bola saiu pela linha de fundo
        :return: str - "right_bottom_line" ou "left_bottom_line" se a bola saiu pela linha de fundo, None caso contrário
        """
        ball = self.frame.ball
        half_len = self.field.length / 2
        half_wid = self.field.width / 2

        if ball.x > half_len and abs(ball.y) < half_wid:
            return "RIGHT_BOTTOM_LINE"
        
        if ball.x < -half_len and abs(ball.y) < half_wid:
            return "LEFT_BOTTOM_LINE"
        
        return None
    
    def _check_sideline(self) -> str|None:
        """
        Verifica se a bola saiu pela lateral
        :return: str - "right_sideline" ou "left_sideline" se a bola saiu pela lateral, None caso contrário
        """
        ball = self.frame.ball
        half_wid = self.field.width/2

        if ball.y > half_wid:
            return "UPPER_SIDELINE"
        
        if ball.y < -half_wid:
            return "LOWER_SIDELINE"
        return None
    
    def _check_opponent_defense_area(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        robot_name = f"{color}_{robot.id}"
        half_len = self.field.length/2 
        penalty_x = half_len - self.field.penalty_length
        penalty_y = self.field.penalty_width/2

        if side == "right":
            in_last_third = robot.x < -penalty_x
        else:
            in_last_third = robot.x > penalty_x

        if (
            in_last_third and 
            abs(robot.y) <= penalty_y and
            self.ball_possession == robot_name
        ):
            return "OPPONENT_DEFENSE_AREA"
        return None
    
    def _check_ally_defense_area(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        half_len = self.field.length/2 
        penalty_x = half_len - self.field.penalty_length
        penalty_y = self.field.penalty_width/2
        if side == "right":
            inside_area = robot.x > penalty_x and abs(robot.y) <= penalty_y
        else:
            inside_area = robot.x < -penalty_x and abs(robot.y) <= penalty_y

        n_robots_in_area = getattr(self, f"n_{side}_robots_in_defense")
        if inside_area:
            n_robots_in_area += 1
            setattr(self, f"n_{side}_robots_in_defense", n_robots_in_area)

        if inside_area and n_robots_in_area > 1:
            return "TEAM_DEFENSE_AREA"
        return None
    
    def _check_collision(self, robot, side, color) -> str|None:
        # Medidas já escaladas
        all_robots = {
            **self.frame.robots_blue, 
            **self.frame.robots_yellow
        }

        for other_idx, other_robot in all_robots.items():
            dist = ((robot.x - other_robot.x)**2 + (robot.y - other_robot.y)**2)**(1/2)
            if 0 < dist < 0.25:
                return "COLLISION"
        return None
    
    def _check_double_touch(self):
        if self.is_kickoff == False: return None

        dist = math.hypot(
            self.frame.ball.x - self.initial_ball_position[0], 
            self.frame.ball.x - self.initial_ball_position[0]
        )

        if (
            self.ball_possession is not None and
            dist > 0.1 and 
            len(self.historical_ball_positions) == 1 and
            self.ball_possession in self.historical_ball_positions
        ):
            self.is_kickoff = False
            return "DOUBLE_TOUCH"
        
        elif len(self.historical_ball_positions) == 2:
            self.is_kickoff = False
            return None

        return None


    def _update_offenses(self) -> None:
        self.offenses = {}
        robots_left, robots_right = self.frame.robots_blue, self.frame.robots_yellow
        robot_left_color, robot_right_color = "blue", "yellow"
        if self.left == "yellow":
            robots_left, robots_right = robots_right, robots_left
            robot_left_color, robot_right_color = "yellow", "blue"

        self.n_left_robots_in_defense = 0
        self.n_right_robots_in_defense = 0
        for idx, robot_left in robots_left.items():
            self.offenses[f"{robot_left_color}_{idx}"] = []
            for func in [self._check_opponent_defense_area, self._check_ally_defense_area, self._check_collision]:
                result = func(robot_left, side="left", color=robot_left_color)
                if result:
                    self.offenses[f"{robot_left_color}_{idx}"].append(result)
        
        for idx, robot_right in robots_right.items():
            self.offenses[f"{robot_right_color}_{idx}"] = []
            for func in [self._check_opponent_defense_area, self._check_ally_defense_area,  self._check_collision]:
                result = func(robot_right, side="right", color=robot_right_color)
                if result:
                    self.offenses[f"{robot_right_color}_{idx}"].append(result)
        
        result = self._check_double_touch()
        if result:
            self.offenses[self.ball_possession].append(result)
                  
    def _update_ball_possession(self) -> str|None:
        """
        Determina qual robô tem a posse da bola ou se a bola está livre.

        Args:
            ball (Ball): O objeto da bola.
            robots (list): Uma lista de objetos Robot.
            possession_radius_scale (float): Fator de escala para o raio de posse da bola
                                            em relação ao tamanho do robô.

        Returns:
            tuple: (robot_id, team_color) do robô com a posse, ou (None, None) se a bola estiver livre.
        """
        
        ball = self.frame.ball
        n_blue = len(self.frame.robots_blue)
        robots = {
            **self.frame.robots_blue,
            **{idx+n_blue: robot for idx, robot in self.frame.robots_yellow.items()}
        }

        closest_robot: Robot = None
        min_distance = float('inf')

        for idx, robot in robots.items():
            distance = math.hypot(ball.x - robot.x, ball.y - robot.y)
            if distance < min_distance:
                min_distance = distance
                closest_robot = robot
                closest_robot.id = idx % n_blue
                closest_robot.yellow = idx // n_blue == 1  # Verifica se é amarelo ou azul

        # Define a zona de domínio do robô como um pouco maior que seu próprio tamanho
        # Isso pode ser ajustado para simular o "controle" da bola
        self.ball_possession = None  
        if not closest_robot: return self.ball_possession
        
        #possession_threshold = closest_robot.rbt_radius * self.possession_radius_scale
        possession_threshold = 0.22 # 0.21 era problematico
        robot_name = f"yellow_{closest_robot.id}" if closest_robot.yellow else f"blue_{closest_robot.id}"
        if min_distance <= possession_threshold:
            self.ball_possession = robot_name
            self.historical_ball_positions.add(self.ball_possession)
        
        return self.ball_possession
              
    def _update_last_touch(self) -> str|None:
        last_ball = self.last_frame.ball
        last_velocity = np.array([last_ball.v_x, last_ball.v_y])
        norm_last_velocity = np.linalg.norm(last_velocity)
        
        ball = self.frame.ball
        current_velocity = np.array([ball.v_x, ball.v_y])
        norm_current_velocity = np.linalg.norm(current_velocity)

        if norm_last_velocity == 0 and norm_current_velocity == 0:
           return self.last_touch

        if norm_last_velocity == 0 and norm_current_velocity > 0:
            last_velocity = -current_velocity 
            norm_last_velocity = norm_current_velocity
        
        if norm_last_velocity > 0 and norm_current_velocity == 0:
            current_velocity = -last_velocity
            norm_current_velocity = norm_last_velocity

        cos_theta = np.dot(last_velocity, current_velocity)
        cos_theta /= (norm_last_velocity * norm_current_velocity)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        direction_changed = angle > self.direction_change_threshold
        if direction_changed and self.ball_possession:
            self.last_touch = self.ball_possession
        
        return self.last_touch
        

            

            





