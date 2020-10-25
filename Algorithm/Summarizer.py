from SummarizerClass import Summarizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
plt.rcdefaults()

csv = csv.writer(open('reviews.csv', mode='w'), delimiter=';')
df = pd.read_csv('data.csv',
                 usecols=['country', 'designation', 'description'])

df.drop_duplicates(subset=['designation', 'country',
                           'description'], inplace=True)

rows = df.groupby(['designation'])[
    'designation', 'description', 'country'].filter(lambda x: len(x) >= 20)

data = {}
for i, row in rows.iterrows():
    if row[0] not in data:
        data[row[0]] = []
    data[row[0]].append(row[1])

# Sort the list by number of reviews
data = sorted(data.items(), key=lambda x: len(x[1]))

csv.writerow(['Wine', 'Number of reviews', 'Summary'])

for wine, reviews in data:
    sumarizer = Summarizer(reviews)
    summary = sumarizer.generate_summaries()
    csv.writerow([wine, len(reviews), summary])
