import numpy as np
import pandas as pd
from utils.model import Perceptron
from utils.all_utils import preparedata

AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}

df = pd.DataFrame(AND)
x,y=preparedata(df)
ETA = 1 # 0 and 1
EPOCHS = 100

model = Perceptron(n=ETA, e=EPOCHS)
model.fit(x, y)