import pandas as pd
import mediapipe as mp

mp_pose = mp.solutions.pose

data = pd.read_csv('./data/left/dataset.csv')
