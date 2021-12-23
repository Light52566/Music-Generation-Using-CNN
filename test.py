import pandas as pd
img = [[0,1,0],[2,1,1],[0,1,1],[0,2,1]]
df = pd.DataFrame(img)
df.columns = range(3)
df.index = range(4)
df.to_csv("outputs/output1.csv",)