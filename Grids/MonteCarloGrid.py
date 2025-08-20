import numpy as np

class grid:
    def __init__(self):

        self.columns=15
        self.rows=15

        #random starting point
        #self.start=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))  
        self.start=(11,12)
        #random ending point which is not atrting point
        #self.end=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        self.end=(4,3)
        while self.end==self.start:
            self.end=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))

        #actions
        self.actions=[np.array([-1,0]), np.array([1,0]), np.array([0,-1]), np.array([0,1])]

        self.gen_walls()


    def gen_walls(self):
        #generate random walls
        # self.walls_num=np.random.randint(10,20)  #random no of walls more than 4, less than 15
        # self.walls=[]
        # for i in range(0, self.walls_num):
        #     x=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        #     while x==self.start or x==self.end:
        #         x=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        #     self.walls.append(x)

        self.walls = [
            (6, 9), (0, 13), (12, 6), (8, 12), (10, 2), (14, 8), (1, 5), (11, 11), (4, 14),
            (2, 3), (7, 10), (5, 1), (13, 1), (9, 6), (12, 12), (6, 0), (10, 13), (8, 1),
            (3, 3), (0, 4), (7, 6), (11, 7), (14, 2), (1, 13), (5, 2), (2, 11), (13, 12),
            (9, 0), (12, 3), (6, 5), (0, 1), (5, 8), (10, 10), (8, 6), (3, 8), (1, 2),
            (11, 0), (14, 11), (2, 8), (4, 6), (13, 9), (7, 13), (9, 3), (6, 13), (0, 7)
        ]


        #assigning rewards
        self.rewards=np.full((self.rows, self.columns), -1)
        for (x,y) in self.walls:
            self.rewards[x,y] = -10
        self.rewards[self.end[0], self.end[1]]= 100


    def is_wall(self, state):
        if state in self.walls:
            return True
        else:
            return False
        
    def is_end(self, state):
        if state==self.end:
            return True
        else:
            return False
        
    def is_bound(self, state):
        if state[0]>=0 and state[0]<self.rows and state[1]>=0 and state[1]<self.columns:
            return True
        else:
            return False
    
    def reset(self):
        return self.start
    
    #func to access just next states
    def next_state(self, coord, act):
        state=np.array(coord)
        #print (state,act)
        if not self.is_bound(tuple(state + self.actions[act])):
            newstate=state
        elif self.is_wall(tuple(state + self.actions[act])):
            newstate=state
        else:
            newstate=state + self.actions[act]
        #print(newstate, reward, done)

        return (int(newstate[0]), int(newstate[1]))


    def move(self, coord, act):
        state=np.array(coord)
        #print (state,act)
        if not self.is_bound(tuple(state + self.actions[act])):
            newstate=state
            reward=-2     #penalty for going out of bound
        elif self.is_wall(tuple(state + self.actions[act])):
            newstate=state
            reward=self.rewards[newstate[0], newstate[1]]
        else:
            newstate=state + self.actions[act]
            reward=self.rewards[newstate[0], newstate[1]]

        done=self.is_end(tuple(newstate))
        #print(newstate, reward, done)

        #regerates walls
        #self.gen_walls()

        return (int(newstate[0]), int(newstate[1])), reward, done


env=grid()

class MON:
    def __init__(self):
        self.columns=env.columns
        self.rows=env.rows
        self.wall=env.walls
        self.gamma=1
        self.alpha=0.5
        self.epsilon=1
        self.q_table=np.zeros((env.rows, env.columns, 4))
        self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        self.end=env.end
        self.goal=0
        
        

    def start(self):
        self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        while self.sta in self.wall:
            self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        self.pos=self.sta
        done=False
        max_steps=100
        steps=0
        self.states=[]
        self.actions=[]
        self.rewards=[]
        while ( steps<max_steps ):
            self.act=self.get_action(self.pos)
            self.states.append(self.pos)
            self.newpos, self.rew, done = env.move(self.pos, self.act)
            self.actions.append(self.act)
            self.rewards.append(self.rew)
            #print(self.pos, self.act)
            #print(self.newpos, self.rew, done )
            if done:
                self.states.append(self.end)
                self.alpha=1
                self.goal+=1
                break
            self.pos=self.newpos
            steps+=1

        #return  self.states, self.actions, self.rewards
        
    def get_action(self, p):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)  #explore
        else:
            return np.argmax(self.q_table[p[0], p[1]]) 

    def Greturn(self):
        self.returns=[]
        for i in range(0,len(self.states)):
            if i==len(self.states)-1:
                break
            ret=0
            for j in range(i, len(self.states)):
                if j==len(self.states)-1:
                    break
                ret+=(self.gamma**(j-i))*self.rewards[j]
            self.returns.append(ret)

        #return self.returns

    def get_advantages(self):
        self.advantages=[]
        for i in range(0, len(self.states)):
            if i==len(self.states)-1:
                break
            advant=self.returns[i] - self.q_table[self.states[i][0], self.states[i][1], self.actions[i]]
            self.advantages.append(advant)

        #return self.advantages

    def update_q(self):
        for i in range(0, len(self.states)):
            if i==len(self.states)-1:
                break
            self.q_table[self.states[i][0],self.states[i][1],self.actions[i]] += self.alpha * self.advantages[i]
            #print(self.advantages)
        #self.alpha*=0.99999999999
        self.alpha=0.5
        #self.epsilon*=1.05
        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_path(self):
        #self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        #while self.sta in self.wall:
            #self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        self.pos=self.sta
        done=False
        max_steps=100
        steps=0
        self.states=[]
        while ( steps<max_steps ):
            self.act=np.argmax(self.q_table[self.pos[0], self.pos[1]])
            self.states.append(self.pos)
            self.newpos, self.rew, done = env.move(self.pos, self.act)
            #print(self.pos, self.act)
            #print(self.newpos, self.rew, done )
            if done:
                self.states.append(self.end)
                self.alpha=1
                self.goal+=1
                break
            self.pos=self.newpos
            steps+=1
        return  self.states
    
    def rand_sta(self):
        self.sta=env.start
        # self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))
        # while self.sta in self.wall:
        #     self.sta=(np.random.randint(0,self.rows), np.random.randint(0,self.columns))

    # def get_path(self):
    #     self.pos=self.sta
    #     done=False
    #     max_steps=100
    #     steps=0
    #     self.states=[]
    #     self.actions=[]
    #     self.rewards=[]
    #     while ( steps < max_steps ):
    #         self.act=np.argmax(self.q_table[self.pos[0], self.pos[1]])
    #         self.states.append(self.pos)
    #         self.newpos, self.rew, done = env.move(self.pos, self.act)
    #         #print(self.pos, self.act)
    #         #print(self.newpos, self.rew, done )
    #         self.actions.append(self.act)
    #         self.rewards.append(self.rew)
    #         if done:
    #             break
    #         self.pos=self.newpos
    #         steps+=1
        
    #     self.states.append(self.end)
    #     return self.states



import cv2
import time

a=MON()

start_time = time.time()

for i in range(0, 10000):
    a.start()
    a.Greturn()
    a.get_advantages()
    a.update_q()
    #if i%1000==0:
        #print("1")

mid_time = time.time()
print("Training time:", mid_time - start_time, "seconds")
print("no of times goal reached", a.goal)


#a.start()
#path=a.states
a.rand_sta()
path=a.get_path()
start=a.sta
end=a.end
walls=a.wall

print(start)
print(end)
print(walls)
print(path)

end_time = time.time()
print("Total time:", end_time - start_time, "seconds")
np.set_printoptions(precision=3, suppress=True)

# for i in range(0,4):
#     print(a.q_table[:, :, i])

s=30 #scale for pixel to box
s2=s/2
cols=a.columns
rows=a.rows

grid_image = np.ones((s*rows, s*cols, 3), dtype=np.uint8) * 255
for (i1, j1) in walls:
    for i in range(s*i1,s*(i1+1)):
        for j in range(s*j1, s*(j1+1)):
            grid_image[i,j]=(0,0,0)

for i in range(s*start[0],s*(start[0]+1)):
    for j in range(s*start[1], s*(start[1]+1)):
        grid_image[i,j]=(0,0,255)

for i in range(s*end[0],s*(end[0]+1)):
    for j in range(s*end[1], s*(end[1]+1)):
        grid_image[i,j]=(0,255,0)

base_grid=grid_image.copy()
    #cv2.imshow("Grid ", grid_image)

for i in range(0,len(path)):
    p = (path[i][1] * s + 15, path[i][0] * s + 15) 
    cv2.circle(grid_image, center=p, radius=10, color=(255, 255, 0), thickness=-1)
    cv2.imshow("Path", grid_image)
    if i==0:
        cv2.waitKey(5000)
    else:
        cv2.waitKey(500)
    grid_image=base_grid.copy()

cv2.destroyAllWindows()

