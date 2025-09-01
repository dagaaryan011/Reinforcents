import numpy as np

class replayBuffer:
    def __init__(self,maxSize , inputShape, nActions):
        self.memorysize = maxSize#after certain amt of actions , ppast ke action overwrite kardenge
        self.memcntr = 0 # memory traversal counter
        self.stateMemory = np.zeros((self.memorysize,*inputShape))
        self.newStateMemory = np.zeros((self.memorysize,*inputShape))
        self.actionMemory = np.zeros((self.memorysize,nActions))
        self.rewardmemory = np.zeros(self.memorysize)
        self.terminalMemory = np.zeros(self.memorysize , dtype=np.bool)
    
    def storeTransition(self , state , action , reward,newstate,done):
        index = self.memcntr%self.memorysize
        self.stateMemory[index] = state
        self.newStateMemory[index]=newstate
        self.actionMemory[index] = action
        self.rewardmemory[index] = reward
        self.terminalMemory[index] = done

        self.memcntr += 1
    
    def sampleBuffer(self , batchsize):
        maxMem = min(self.memcntr, self.memorysize)
        batch = np.random.choice(maxMem,batchsize,replace=False)
        states = self.stateMemory[batch]
        states_ = self.newStateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardmemory[batch]
        dones = self.terminalMemory[batch]

        return states, actions, rewards, states_, dones