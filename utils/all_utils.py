import numpy as np
import pandas as pd


def preparedata(df):
    y=df["y"]
    x=df.drop("y",axis=1)
    return x,y