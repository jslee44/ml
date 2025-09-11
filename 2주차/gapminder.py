import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

data = pd.read_csv('https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/gapminderData.csv')
data['continent'] = pd.Categorical(data['continent'])

fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

def update(frame):
    ax.clear()
    yearly_data = data.loc[data.year == frame, :]
    ax.scatter(
        x=yearly_data['lifeExp'], 
        y=yearly_data['gdpPercap'], 
        s=yearly_data['pop']/100000,
        c=yearly_data['continent'].cat.codes, 
        cmap="Accent", 
        alpha=0.6, 
        edgecolors="white", 
        linewidths=2
    )
    ax.set_title(f"Global Development in {frame}")
    ax.set_xlabel("Life Expectancy")
    ax.set_ylabel("GDP per Capita")
    ax.set_yscale('log')
    ax.set_ylim(100, 100000)
    ax.set_xlim(20, 90)
    return ax
ani = FuncAnimation(fig, update, frames=data['year'].unique())
ani.save('/content/gapminder-1.gif', fps=1)
plt.show()