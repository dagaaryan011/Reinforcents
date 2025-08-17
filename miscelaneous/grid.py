import numpy as np

class GridRunnerEnv:
    #init function apna constructor hai , ye apna env bana daalega , filhaal 6*6 grid ; 0,0 to 5,5
    def __init__(self):
        self.grid = np.zeros((6, 6), dtype=int)  # all 0s for now


        goalx = np.random.randint(0, 6)
        goaly = np.random.randint(0, 6)
        self.goal = (goalx, goaly)
        

        while True:
         startx = np.random.randint(0, 6)
         starty = np.random.randint(0, 6)
         self.startpos = (startx, starty)
         if self.startpos != self.goal:
          break

        self.playerpos = [self.startpos for _ in range(3)]# 3 players,sab same start point se
        self.generWalls()
        self.actions = [
            (-1, 0),  # 0: UP
            (1, 0),   # 1: DOWN
            (0, -1),  # 2: LEFT
            (0, 1)    # 3: RIGHT
            ]

        self.rewardEarned = 0

    def generWalls(self):
       #walls after every turn random generate hogi , so walls ko randomly generate karege
       self.walls = []#walls
       for i in range(0,np.random.randint(4,15)):
          wallx = np.random.randint(0, 6)
          wally = np.random.randint(0, 6)
          wall = (wallx,wally)
          while(True):
             if wall != self.goal and wall not in self.playerpos:
              self.walls.append(wall)
              break
    def restart(self):
        self.playerpos = [self.startpos for _ in range(3)]
        self.rewardEarned = 0
        
        


    def move(self, agent_id, action):
        x, y = self.playerpos[agent_id]
        stepx, stepy = self.actions[action]
        nx, ny = x + stepx, y + stepy

        # 
        collision = (
            not (0 <= nx < 6 and 0 <= ny < 6) or
            (nx, ny) in self.walls or
            (nx, ny) in [pos for i, pos in enumerate(self.playerpos) if i != agent_id]
        )

        if collision:
            reward = -7
            done = False
            return self.playerpos[agent_id], reward, self.walls, done

        #
        self.playerpos[agent_id] = (nx, ny)

        if self.playerpos[agent_id] == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        # 
        self.generWalls()

        return self.playerpos[agent_id], reward, self.walls, done



class AgentSarsa:
   # State Action Reward State Action
   
   
    def __init__(self, env, agent_id, episodes, ep=0.1, df=0.9, lr=0.1):
        self.env = env
        self.agent_id = agent_id  # each agent gets an ID (0, 1, 2)
        self.episodes = episodes
        self.epsilon = ep
        self.discount_factor = df
        self.learning_rate = lr
        self.qvalues = np.zeros((6, 6, 4))
        




    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)  # explore
        else:
            x, y = state
            return np.argmax(self.qvalues[x, y])  # exploit


    def train(self):
        for episode in range(self.episodes):

            if episode % 100 == 0:
                print(".", end="", flush=True)

            self.env.restart()
            self.paths = [[] for _ in range(3)]
            state = self.env.playerpos[self.agent_id]
            action = self.choose_action(state)
            done = False

            while not done:
                next_state, reward, _, done = self.env.move(self.agent_id, action)
                next_action = self.choose_action(next_state)

                # SARSA update
                x, y = state
                nx, ny = next_state
                self.paths[self.agent_id].append((nx,ny))
                td_target = reward + self.discount_factor * self.qvalues[nx, ny, next_action]
                td_error = td_target - self.qvalues[x, y, action]
                self.qvalues[x, y, action] += self.learning_rate * td_error

                state = next_state
                action = next_action
                if (len(self.paths[self.agent_id])>30):
                    break



if __name__ == "__main__":
    env = GridRunnerEnv()
    agents = [
        AgentSarsa(env, agent_id=0, episodes=1000, ep=0.3, df=0.8, lr=0.1),
        AgentSarsa(env, agent_id=1, episodes=1000, ep=0.3, df=0.8, lr=0.1),
        AgentSarsa(env, agent_id=2, episodes=1000, ep=0.3, df=0.8, lr=0.1),
    ]

    for agent in agents:
        print(f"goal:{env.goal}")

        agent.train()

    
    for i, path in enumerate(paths):
        print(f"Agent {i} path: {path}")

