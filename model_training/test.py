import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

user_df = pd.read_csv("./data/users.csv")
games_df = pd.read_csv("./data/games.csv")
reviews_df = pd.read_csv("./data/recommendations.csv")

# show only the first 5 rows
""" print(user_df.head())
print(games_df.head())
print(reviews_df.head()) """

#exploratory data analysis
""" print(user_df.describe())
print(games_df.describe())
print(reviews_df.describe()) """

#convert date columns to datetime
games_df['date_release'] = pd.to_datetime(games_df['date_release'])
reviews_df['date'] = pd.to_datetime(reviews_df['date'])


""" plt.figure(figsize=(15, 6))
games_df['date_release'].dt.year.value_counts().sort_index().plot(kind='line')
plt.title('Number of Games Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.show() """

tags = set()
with open("./data/games_metadata.json", "r", encoding="utf-8") as f:
    for line in f:
        game = json.loads(line.strip())
        for tag in game.get('tags', []):
            tags.add(tag)

# Print the total amount of unique tags for the entire dataset
print(f"Total unique tags: {len(tags)}")
for tag in list(tags)[:10]:
    print(tag)


