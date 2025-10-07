import sys
from model import Network_Utils
import numpy as np

MAIN_DATA = np.random.rand(50, 6)

class Trainer:

    def __init__(self):
        self.all_models:list[Network_Utils] = []
        self.personality = {
            "Momentum Trader":[1, 1, 1, 0, 0],
            "Trend Follower":[0, 1, 0, 1, 1],
            "Hybrid Trader":[1, 0, 1, 0, 1],
            "All-Rounder":[1, 1, 1, 1, 1]
        }
        self.indicator_label = ['macd', 'rsi', 'stoch', 'adx', 'status']
        self.model_dir = "models"

    def save_models(self):
        for i in range(len(self.all_models)):
            self.all_models[i].build_model(f"models/{self.personality.keys()[i]}.pth")

    def trainer(self, raw):
        # expected in put will be 
        # what is the type of the x like numpy or what
        
        for i in range(len(self.all_models)):
            x_, y_ = self.input_preprocessing(raw)
            x_ = np.array(x_)
            y_ = np.array(y_)


    def outputs(self, x):
        # excepted shape of the input (1, 30, 5)
        output = []
        for i in self.all_models:
            output.append(i.output(x))
        return output

    def load_model(self):
        for i in self.personality.keys():
            obj = Network_Utils()
            obj.load_model(f"models/{i}.pth")
            self.all_models.append(obj)

    def build_model(self):
        for i in self.personality.keys():
            obj = Network_Utils()
            obj.build_model(f"models/{i}.pth")

    def input_preprocessing(self, raw, label=0):
        x = []
        y = []
        if len(raw) < 30:
            return
        
        for i in range(len(raw) - 30):
            dummpy_x = raw[i:30 + i][:5]
            dummpy_y = raw[30+i][-1]

            x.append(dummpy_x)
            y.append(dummpy_y)

        return x, y
        

        


if __name__ == "__main__":
    obj_1 = Trainer()
    obj_1.trainer(MAIN_DATA.tolist())