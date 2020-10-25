import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcdefaults()

df = pd.read_csv('data.csv',
                 usecols=['country', 'designation', 'description'])

df.drop_duplicates(subset=['designation', 'country',
                           'description'], inplace=True)

rows = df.groupby('designation').size().sort_values(ascending=True)

data = {}
for row, size in rows.items():
    if size >= 20:
        if size in data:
            data[size] += 1
        else:
            data[size] = 1

plt.bar(range(len(data)), list(data.values()), align='center')
plt.xticks(range(len(data)), list(data.keys()), rotation=90)
plt.xlabel('Number of reviews')
plt.ylabel('Different wine sorts')
plt.show()
