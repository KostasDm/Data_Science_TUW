import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


tables_all = pd.read_csv('pl-tables-1993-2024.csv')
xgoals_2022 = pd.read_csv('Premier21-22.csv')
xgoals_2023 = pd.read_csv('Premier22_23.csv')
stats_2022 = pd.read_csv('2021_season_club_stats.csv')
stats_2023 = pd.read_csv('2022_season_club_stats.csv')
stats_2024 = pd.read_csv('2023_season_club_stats.csv')

####################################
# Data preprocessing

xgoals_2022['XG_per_game'] = (xgoals_2022['Home xG'].div(19) + xgoals_2022['Away xG'].div(19))/2
xgoals_2023['XG_per_game'] = (xgoals_2023['Home xG'].div(19) + xgoals_2023['Away xG'].div(19))/2

xgoals_2022['season'] = ['2021/2022']*xgoals_2022.shape[0]
xgoals_2023['season'] = ['2022/2023']*xgoals_2023.shape[0]

xgoals_2022 = xgoals_2022.rename(columns= {'Squad': 'club_name'})
xgoals_2023 = xgoals_2023.rename(columns= {'Squad': 'club_name'})

xgoals_2023['club_name'] = xgoals_2023['club_name'].str.replace("Nott'ham Forest", 'Nottingham Forest', regex=False)



stats_2022['club_name'] = stats_2022['club_name'].str.replace('Tottenham Hotspur', 'Tottenham', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('Tottenham Hotspur', 'Tottenham', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('Tottenham Hotspur', 'Tottenham', regex=False)

stats_2022['club_name'] = stats_2022['club_name'].str.replace('Wolverhampton Wanderers', 'Wolves', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('Wolverhampton Wanderers', 'Wolves', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('Wolverhampton Wanderers', 'Wolves', regex=False)

stats_2022['club_name'] = stats_2022['club_name'].str.replace('Brighton and Hove Albion', 'Brighton', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('Brighton and Hove Albion', 'Brighton', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('Brighton and Hove Albion', 'Brighton', regex=False)

stats_2022['club_name'] = stats_2022['club_name'].str.replace('West Ham United', 'West Ham', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('West Ham United', 'West Ham', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('West Ham United', 'West Ham', regex=False)


stats_2022['club_name'] = stats_2022['club_name'].str.replace('Manchester United', 'Manchester Utd', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('Manchester United', 'Manchester Utd', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('Manchester United', 'Manchester Utd', regex=False)

stats_2022['club_name'] = stats_2022['club_name'].str.replace('Newcastle United', 'Newcastle Utd', regex=False)
stats_2023['club_name'] = stats_2023['club_name'].str.replace('Newcastle United', 'Newcastle Utd', regex=False)
stats_2024['club_name'] = stats_2024['club_name'].str.replace('Newcastle United', 'Newcastle Utd', regex=False)

stats_2024['club_name'] = stats_2024['club_name'].str.replace('Sheffield United', 'Sheffield Utd', regex=False)

stats_2022 = pd.merge(stats_2022[['season', 'club_name', 'Games Played', 'Goals', 'Goals Conceded']], xgoals_2022[['club_name', 'season', 'XG_per_game']],
                       on = ['season', 'club_name'], how = 'left')

stats_2023 = pd.merge(stats_2023[['season', 'club_name', 'Games Played', 'Goals', 'Goals Conceded']], xgoals_2023[['club_name', 'season', 'XG_per_game']],
                       on = ['season', 'club_name'], how = 'left')

stats_2022.loc[stats_2022['club_name']=='Newcastle Utd', 'Games Played'] = 38
stats_2022.loc[stats_2022['club_name']=='Newcastle Utd', 'Goals'] = 44
stats_2022.loc[stats_2022['club_name']=='Newcastle Utd', 'Goals Conceded'] = 62

stats_2023.loc[stats_2023['club_name']=='Newcastle Utd', 'Games Played'] = 38
stats_2023.loc[stats_2023['club_name']=='Newcastle Utd', 'Goals'] = 68
stats_2023.loc[stats_2023['club_name']=='Newcastle Utd', 'Goals Conceded'] = 33

stats_2022['season'] = stats_2022['season'].str.replace('2021/2022', '2022', regex=False)
stats_2023['season'] = stats_2023['season'].str.replace('2022/2023', '2023', regex=False)
stats_2024['season'] = stats_2024['season'].str.replace('2023/2024', '2024', regex=False)

cols_of_interest = ['club_name', 'season', 'Games Played', 'Goals', 'Goals Conceded', 'XG_per_game']
stats_2024['XG_per_game'] = stats_2024['XG'].div(stats_2024['Games Played'])

stats_2024 = stats_2024[cols_of_interest]

df_stats_all = pd.concat([stats_2022, stats_2023, stats_2024])

df_stats_all['Goals_per_game'] = df_stats_all['Goals'].div(df_stats_all['Games Played'])
df_stats_all['Goals_Conceded_per_game'] = df_stats_all['Goals Conceded'].div(df_stats_all['Games Played'])
df_stats_all['GD_per_game'] = df_stats_all['Goals_per_game'] - df_stats_all['Goals_Conceded_per_game']
df_stats_all['Goal_efficiency_per_game'] = df_stats_all['Goals_per_game'].div(df_stats_all['XG_per_game'])


############
### Data merging 
tables_all = tables_all.rename(columns = {'season_end_year': 'season'})
tables_all = tables_all.rename(columns = {'team': 'club_name'})

df_stats_all['season'] = df_stats_all['season'].astype(int)

df = pd.merge(df_stats_all, tables_all[['season', 'club_name', 'position']], on=['season', 'club_name'], how ='left')



plt.figure(figsize=(9,7))
ax = sns.scatterplot(
    data=df,
    x="GD_per_game",
    y="position",
    hue="season",
    palette="tab10",
    s=100
)

ax = sns.lineplot(
    data=df,
    x="GD_per_game",
    y="position",
    hue="season",
    palette="tab10",
    linestyle='dotted',
    legend=False
)

ax.invert_yaxis()
ax.set_title("Premier League stats (2022 - 2024) -\n Does average goal difference reflect the final position?")
ax.set_xlabel("Goal difference per game", fontsize=13)
ax.set_ylabel("Final position", fontsize=13)
ax.grid(True, linestyle="--", alpha=0.3)

positions = sorted(df["position"].unique())
ax.set_yticks(positions)
ax.set_yticklabels([str(pos) for pos in positions])
# --- Create inset axis ---
inset_ax = inset_axes(
    ax,
    width="35%", 
    height="35%", 
    loc="lower right",
    bbox_to_anchor=(0, 0.15, 1, 1),  # (x0, y0, width, height) relative to parent axes
    bbox_transform=ax.transAxes
)

# Plot regression lines and compute correlations per season
colors = {"2022": "#1f77b4", "2023": "#ff7f0e", "2024": "#2ca02c"}
y_offset = 0.9  # starting point for text annotation inside inset

for season, color in colors.items():
    subset = df[df["season"] == int(season)]
    sns.regplot(
        data=subset, x="GD_per_game", y="position",
        scatter=False, color=color, ci=None, ax=inset_ax,
        line_kws={'linewidth': 2}
    )
    r, _ = pearsonr(subset["GD_per_game"], subset["position"])
    inset_ax.text(
        0.05, y_offset, f"{season}: r = {r:.2f}",
        transform=inset_ax.transAxes, color=color, fontsize=9, weight="bold"
    )
    y_offset -= 0.12

# Invert y for consistency and style inset
inset_ax.invert_yaxis()
inset_ax.set_title("Pearson correlations", fontsize=10)
inset_ax.set_xlabel("Goal difference per game")
inset_ax.set_ylabel("Final position")
inset_ax.tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
plt.show()

