import cv2
import numpy as np
import time
from tqdm import tqdm

#-------------------Env----------------------------------
class GridRunnerEnv:
    #init function apna constructor hai , ye apna env bana daalega , filhaal 6*6 grid ; 0,0 to 5,5
    def __init__(self,number):
        self.grid = np.zeros((10, 10), dtype=int)  # all 0s for now


        goalx = np.random.randint(0, 6)
        goaly = np.random.randint(6, 10)
        self.goal = (goalx, goaly)
        startx = np.random.randint(6, 10)
        starty = np.random.randint(0, 6)
        self.startpos = (startx, starty) 
        self.numberofwalls = number
        while self.startpos == self.goal:
         startx = np.random.randint(0, 10)
         starty = np.random.randint(0, 10)
         self.startpos = (startx, starty)
         
        self.playerpos = self.startpos#player start yaha se karega
        
        
        
        self.walls = []#walls
        while self.numberofwalls > len(self.walls):
          wallx = np.random.randint(0, 10)
          wally = np.random.randint(0, 10)
          wall = (wallx,wally)
          if wall == self.goal or wall == self.startpos:
            wallx = np.random.randint(0, 10)
            wally = np.random.randint(0, 10)
            wall = (wallx,wally)
            self.walls.append(wall)
          else: self.walls.append(wall)
          # while wall == self.goal or wall == self.startpos:
          #   wallx = np.random.randint(0, 6)
          #   wally = np.random.randint(0, 6)
          #   wall = (wallx,wally)
          #   self.walls.append(wall)
        self.actions = [
            (-1, 0),  # 0: UP
            (1, 0),   # 1: DOWN
            (0, -1),  # 2: LEFT
            (0, 1)    # 3: RIGHT
            ]

        self.rewardEarned = 0
    def restart(self):
        self.playerpos = self.startpos
        self.rewardEarned = 0
        


    def move(self, action):
        
        x, y = self.playerpos  # current position

        # kidhar jayga?
        stepx, stepy = self.actions[action]
        nx = x + stepx
        ny = y + stepy

        #takraya ki nai
        if (0 <= nx < 10) and (0 <= ny < 10) and ((nx, ny) not in self.walls):
            self.playerpos = (nx, ny)  # badgaya aage
        else:
        # wall ya boundary ko takraya toh jyada punishment
            reward = -200
            return self.playerpos, reward
        
        #pahucha toh good , nai toh -1 reward
        if self.playerpos == self.goal:
            reward = 1000
            
        else:
            reward = -75

        return self.playerpos, reward



#----------------------Agent-------------------------------
class AgentExpecsarsa: 
    def __init__(self,env,episodes,epsilon,df,lr):
      self.env = env # enviourment

      # self.qvalues = np.random.rand(10,10,4)#qtable
      self.qvalues = np.random.rand(10,10,4)#qtable

      self.epsilon = epsilon#not very lalchi  
      self.discount_factor = df #discount factor for future rewards
      self.learning_rate = lr #the rate at which the AI agent should learn
      
      self.episodes = episodes
      self.posx, self.posy = self.env.playerpos
      self.agent_name = "Expected Sarsa"
      # reward = 0 #starting mei reward 0 rakhege , fir aage jaake zero kardenge
    
    
    def get_action(self,ep):
        # if np.random.random()  ep:
        #   return np.argmax(self.qvalues[self.posx, self.posy])
        # else: #choose a random action
        #   return np.random.randint(4)
        if np.random.random() < ep:
          return np.random.randint(4) 
        else:
          return np.argmax(self.qvalues[self.posx,self.posy])
    def Qvalues(self):
      for i in range(10):
        for j in range(10):
            print(f"Q[{i},{j}] = {self.qvalues[i,j]}")
    def Qvaluesanalysis(self):
      actions = {
        0: '^',  # Up
        1: 'v',  # Down
        2: '<',  # Left
        3: '>'   # Right
                  }
      print("Q Map:")
      for i in range(10):
        row = ""
        for j in range(10):
            if (i, j) in self.env.walls:
                row += "X "       # Wall cell
            elif (i, j) == self.env.goal:
                row += "G "       # Goal
            elif (i, j) == self.env.startpos:
                row += "S "       # Goal
            else:
                action = np.argmax( self.qvalues[i][j])
                row += f"{actions[action]}  "
        print(row)
    def path(self):
      self.path = []
      self.env.restart()  # start from beginning
      pos = self.env.playerpos


      while True:
        x, y = pos
        action = np.argmax(self.qvalues[x, y])
        self.path.append(pos)

        #
        pos, _ = self.env.move(action)

        if pos == self.env.goal:
            self.path.append(pos)  
            break

        if len(self.path) > 50:
            print("Not trained nicely")
            break

      return self.path 
    
    def train(self):
      startTime = time.time()
    #   print("Q-table before training:")
    #   print(self.qvalues)

      for episode in tqdm(range(self.episodes)):
          visited = set()        
          self.env.restart()
          meow = episode
          # if episode % 1000 == 999:
          
          step = 0
          self.posx, self.posy = self.env.playerpos

          
          action = self.get_action(self.epsilon)

          while step < 40 and self.env.playerpos != self.env.goal:
              step += 1
              # Inside your training loop:
              

              # Old position
              oldx, oldy = self.env.playerpos

              # karm karo , fal pao
              newpos, reward = self.env.move(action)
              newx, newy = newpos
              if newpos in visited:
                  reward -= 2  # discourage revisits
              visited.add(newpos)
              # reward -= step
              #ab kya karajaye
              next_action = self.get_action(self.epsilon)

              # sarsa wala q tablee
              old_qvalue = self.qvalues[oldx, oldy, action]
              qvals = self.qvalues[newx, newy, :]
              greedy_action = np.argmax(qvals)
              expected_q = sum(
             ((1 - self.epsilon) + self.epsilon / 4 if a == greedy_action else self.epsilon / 4) * qvals[a]for a in range(4)
)

              temporal_difference = reward + (self.discount_factor * expected_q) - old_qvalue
              new_qvalue = old_qvalue + (self.learning_rate * temporal_difference)

              self.qvalues[oldx, oldy, action] = new_qvalue

              
              action = next_action
              self.posx, self.posy = newx, newy

               
          self.epsilon *= 0.999999999999999999999999999999999999999995
      endTime = time.time()
      
      print("Q-table after training:")
      self.Qvaluesanalysis()
      self.pathdikha(self.episodes)
      
      print(f"Agent trained in: {endTime - startTime:.2f} seconds")
       

#-----------------------visulaize--------------
    def genimage(self,pos,episode):
        cell_size = 50
        size = 10
        goal = self.env.goal
        walls = self.env.walls
        start = self.env.startpos
        playerpos = pos
        img = np.zeros((size * cell_size, size * cell_size, 3), dtype=np.uint8)


        for i in range(1, size):
          cv2.line(img, (0, i * cell_size), (size * cell_size, i * cell_size), (255, 255, 255), 1)
          cv2.line(img, (i * cell_size, 0), (i * cell_size, size * cell_size), (255, 255, 255), 1)
      
        for i in range(1, size):
          # cv2.line(img, (0, i * cell_size), (size * cell_size, i * cell_size), (255, 255, 255), 1)
          # cv2.line(img, (i * cell_size, 0), (i * cell_size, size * cell_size), (255, 255, 255), 1)

          # Draw goal
          gx, gy = goal
          cv2.rectangle(img, (gy * cell_size, gx * cell_size),
                          ((gy + 1) * cell_size, (gx + 1) * cell_size),
                          (0, 255, 0), -1)
          sx,sy = start
          cv2.rectangle(img, (sy * cell_size, sx * cell_size),
                          ((sy + 1) * cell_size, (sx + 1) * cell_size),
                          (0, 0, 255), -1)
          # diwaar
          for wx, wy in walls:
              cv2.rectangle(img, (wy * cell_size, wx * cell_size),
                               ((wy + 1) * cell_size, (wx + 1) * cell_size),
                               (128, 128, 128), -1)

          
          px, py = playerpos
          cv2.rectangle(img, (py * cell_size, px * cell_size),
                            ((py + 1) * cell_size, (px + 1) * cell_size),
                            (255, 0, 0), -1)
          

          
          cv2.imshow(f"GridRunner - Test {self.agent_name}", img)
          cv2.waitKey(100) 


    def pathdikha(self, episode):
        self.latestpath = []
        self.env.restart()
        pos = self.env.playerpos
        visited = set()  # store all visited positions

        step = 0
        while step < 50:
            x, y = pos

            
            if pos in visited:
                action = np.random.randint(4)
            else:
                action = np.argmax(self.qvalues[x, y])

            visited.add(pos)
            self.latestpath.append(pos)

            newpos, _ = self.env.move(action)
            
            tries = 0
            while newpos in visited and tries < 4:
                newpos, _ = self.env.move(np.random.randint(4))
                tries += 1

            pos = newpos

            if pos == self.env.goal:
                self.latestpath.append(pos)
                self.genimage(pos, episode)
                break

            self.genimage(pos, episode)
            step += 1

        cv2.destroyAllWindows()

        
class Agentsarsa: 
    def __init__(self,env,episodes,epsilon,df,lr):
      self.env = env # enviourment

      # self.qvalues = np.random.rand(10,10,4)#qtable
      self.qvalues = np.zeros((10,10,4))#qtable

      self.epsilon = epsilon#not very lalchi  
      self.discount_factor = df #discount factor for future rewards
      self.learning_rate = lr #the rate at which the AI agent should learn
      
      self.episodes = episodes
      self.posx, self.posy = self.env.playerpos
      self.agent_name = "Sarsa"

      # reward = 0 #starting mei reward 0 rakhege , fir aage jaake zero kardenge
    
    
    def get_action(self,ep):
        # if np.random.random()  ep:
        #   return np.argmax(self.qvalues[self.posx, self.posy])
        # else: #choose a random action
        #   return np.random.randint(4)
        if np.random.random() < ep:
          return np.random.randint(4) 
        else:
          return np.argmax(self.qvalues[self.posx,self.posy])
    def Qvalues(self):
      for i in range(10):
        for j in range(10):
            print(f"Q[{i},{j}] = {self.qvalues[i,j]}")
    def Qvaluesanalysis(self):
      actions = {
        0: '^',  # Up
        1: '\/',  # Down
        2: '<',  # Left
        3: '>'   # Right
                  }
      print("Q Map:")
      for i in range(10):
        row = ""
        for j in range(10):
            if (i, j) in self.env.walls:
                row += "X "       # Wall cell
            elif (i, j) == self.env.goal:
                row += "G "       # Goal
            elif (i, j) == self.env.startpos:
                row += "S "       # Goal
            else:
                action = np.argmax( self.qvalues[i][j])
                row += f"{actions[action]}  "
        print(row)
    def path(self):
      self.path = []
      self.env.restart()  # start from beginning
      pos = self.env.playerpos


      while True:
        x, y = pos
        action = np.argmax(self.qvalues[x, y])
        self.path.append(pos)

        #
        pos, _ = self.env.move(action)

        if pos == self.env.goal:
            self.path.append(pos)  
            break

        if len(self.path) > 50:
            print("Not trained nicely")
            break

      return self.path 
    
    def train(self):
      startTime = time.time()
    #   print("Q-table before training:")
    #   print(self.qvalues)
      self.Qvaluesanalysis()

      for episode in tqdm(range(self.episodes)):
          visited = set()        
          self.env.restart()
          meow = episode
          # if episode % 1000 == 999:
          
          step = 0
          self.posx, self.posy = self.env.playerpos

          
          action = self.get_action(self.epsilon)

          while step < 40 and self.env.playerpos != self.env.goal:
              step += 1
              # Inside your training loop:
              

              # Old position
              oldx, oldy = self.env.playerpos

              # karm karo , fal pao
              newpos, reward = self.env.move(action)
              newx, newy = newpos
              if newpos in visited:
                  reward -= 0.5  # discourage revisits
              visited.add(newpos)
              # reward -= step
              #ab kya karajaye
              next_action = self.get_action(self.epsilon)

              # sarsa wala q tablee
              old_qvalue = self.qvalues[oldx, oldy, action]
              next_qvalue = max(old_qvalue , self.qvalues[newx, newy, next_action] )
              temporal_difference = reward + (self.discount_factor * next_qvalue) - old_qvalue
              new_qvalue = old_qvalue + (self.learning_rate * temporal_difference)

              self.qvalues[oldx, oldy, action] = new_qvalue

              
              action = next_action
              self.posx, self.posy = newx, newy

               
          self.epsilon *= 0.999999999999999999999999999999999999999995
      endTime = time.time()
      self.pathdikha(self.episodes)
      print("Q-table after training:")
      self.Qvaluesanalysis()
      
      print(f"Agent trained in: {endTime - startTime:.2f} seconds")
       

#-----------------------visulaize--------------
    def genimage(self,pos,episode):
        cell_size = 50
        size = 10
        goal = self.env.goal
        walls = self.env.walls
        start = self.env.startpos
        playerpos = pos
        img = np.zeros((size * cell_size, size * cell_size, 3), dtype=np.uint8)


        for i in range(1, size):
          cv2.line(img, (0, i * cell_size), (size * cell_size, i * cell_size), (255, 255, 255), 1)
          cv2.line(img, (i * cell_size, 0), (i * cell_size, size * cell_size), (255, 255, 255), 1)
      
        for i in range(1, size):
          # cv2.line(img, (0, i * cell_size), (size * cell_size, i * cell_size), (255, 255, 255), 1)
          # cv2.line(img, (i * cell_size, 0), (i * cell_size, size * cell_size), (255, 255, 255), 1)

          # Draw goal
          gx, gy = goal
          cv2.rectangle(img, (gy * cell_size, gx * cell_size),
                          ((gy + 1) * cell_size, (gx + 1) * cell_size),
                          (0, 255, 0), -1)
          sx,sy = start
          cv2.rectangle(img, (sy * cell_size, sx * cell_size),
                          ((sy + 1) * cell_size, (sx + 1) * cell_size),
                          (0, 0, 255), -1)
          # diwaar
          for wx, wy in walls:
              cv2.rectangle(img, (wy * cell_size, wx * cell_size),
                               ((wy + 1) * cell_size, (wx + 1) * cell_size),
                               (128, 128, 128), -1)

          
          px, py = playerpos
          cv2.rectangle(img, (py * cell_size, px * cell_size),
                            ((py + 1) * cell_size, (px + 1) * cell_size),
                            (255, 0, 0), -1)
          

          
          cv2.imshow(f"GridRunner - Test {self.agent_name}", img)
          cv2.waitKey(100) 


    def pathdikha(self,episode):
      self.latestpath = []
      self.env.restart()
      pos = self.env.playerpos

      step = 0
      while step < 50:
        x, y = pos
        action = np.argmax(self.qvalues[x, y])
        self.latestpath.append(pos)

        newpos, _ = self.env.move(action)

        
        if newpos == pos:
            break

        pos = newpos  
        if pos == self.env.goal:
            self.latestpath.append(pos)
            self.genimage(pos,episode)
            break

        self.genimage(pos,episode)
        step += 1
      print (self.latestpath)
      cv2.destroyAllWindows()  
    # Close the window after path visualization
    

        

      

      
#---------------------------------------------------------------------------------------------------       
    def showPath(self):
      grid_image = np.ones((80*10, 80*10, 3), dtype=np.uint8) * 255
      
      for i in range(0,10):
         row1 = (80*i,0)
         row2 = (80*i,480)
         cv2.line(grid_image, row1, row2, color=(0, 0, 0), thickness=5)
      for i in range(0,10):
         col1 = (0,80*i)
         col2 = (480,80*i)
         cv2.line(grid_image, col1, col2, color=(0, 0, 0), thickness=5)
      
      #WALLS   pe red cross banana hai
      cell_size = 80
      for (grid_y, grid_x) in self.env.walls:
        x = grid_x * cell_size + cell_size // 2
        y = grid_y * cell_size + cell_size // 2

        pt1 = (x - 40, y - 40)
        pt2 = (x + 40, y + 40)
        pt3 = (x + 40, y - 40)
        pt4 = (x - 40, y + 40)

        cv2.line(grid_image, pt1, pt2, color=(0, 0, 0), thickness=5)
        cv2.line(grid_image, pt3, pt4, color=(0, 0, 0), thickness=5)

        cell_size = 80
#       x, y = self.env.goal
#       print(f"goal is {self.env.goal}")
# # Convert to center pixel of the goal cell
#       cx = (2*x-1)*40
#       cy = (2*y-1)*40
      x,y= self.env.goal
      for i in range((2*x)*40,(2*x + 2)*40):
        for j in range((2*y)*40,(2*y + 2)*40):
           grid_image[i,j]=(0,255,0)
      x,y= self.env.startpos
      for i in range((2*x)*40,(2*x + 2)*40):
        for j in range((2*y)*40,(2*y + 2)*40):
           grid_image[i,j]=(0,0,255)
         



      
      grad = 1 / len(self.path)

      for i in range(len(self.path) - 1):
        p1 = self.path[i]
        p2 = self.path[i + 1]

        # Convert (row, col) to (x, y) pixel coordinates
        pt1 = (p1[1] * 80 + 40, p1[0] * 80 + 40)  # center of cell
        pt2 = (p2[1] * 80 + 40, p2[0] * 80 + 40)

        cv2.line(grid_image, pt1, pt2, color=(0*(1-grad), 255, 255), thickness=5)

      cv2.imshow(f"Path{self}", grid_image)


    
      
#---------------------Playground----------------
if __name__ == "__main__":
    env = GridRunnerEnv(32)
    agent1 = Agentsarsa(env, episodes=10000,epsilon =0.9 , df = 0.6, lr= 0.5)  
    agent1.train()
    
    agent2 = AgentExpecsarsa(env, episodes=10000,epsilon = 1, df = 1, lr= 0.99999999999999999999999999999999999)  
    agent2.train()
    print(f"Start : {agent1.env.startpos}")
    print(f"Goal: {agent1.env.goal}")
    print(f"the walls are at: {agent1.env.walls}")
    print("the final path for sarsa:")
    print (agent1.path())
    print("the final path for Expected sarsa:")
    print (agent2.path())
    # agent1.showPath()
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

          
           


    