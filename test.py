import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/output0.csv").iloc[:,1:]
plt.imshow(df,vmax=2)
plt.show()
