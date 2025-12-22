import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


df = pd.read_csv('emissions.csv')


####################################
# Data preprocessing




# 3. Percentage calculation


df = df.loc[df['Country'].isin(['Africa', 'Europe', 'Asia', 'Australia and New Zealand', 'Americas', 'World'])]

df = df.replace('Water supply; sewerage, waste management and remediation activities', 'Water supply')

quarter_cols = df.columns[(df.columns.str.contains(r'^\d{4}Q[1-4]$')) |(df.columns.isin(['Country', 'ObjectId', 'Industry', 'Gas Type', 'Seasonal Adjustment']))]

df_q = df[quarter_cols]

# 2. Melt into long form
id_cols = ['Country', 'ObjectId', 'Industry', 'Gas Type', 'Seasonal Adjustment']   # replace with your actual identifiers

df_long = df_q.melt(id_vars=id_cols, var_name='period', value_name='value')


# Split into year and quarter
df_long['year'] = df_long['period'].str[:4].astype(int)
df_long['quarter'] = df_long['period'].str[-2:]   # Q1, Q2, Q3, Q4

#df_long = df_long.loc[df_long['year'].astype(int)<2025]
# 4. Pivot wide
#df_final = df_long.pivot(index=id_cols + ['year'], columns='quarter', values='value')

# Optional: clean index
df_final = df_long.reset_index()

df_year = df_final[(df_final['Gas Type'] == 'Greenhouse gas') & (df_final['Seasonal Adjustment'] =='Seasonally Adjusted')]#.groupby('Industry')['value'].sum().reset_index()


df_world = df_year[df_year['Country'] == 'World'].copy() # World
df_cont  = df_year[df_year['Country'] != 'World'].copy()

quarter_to_month = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}

df_world['month'] = df_world['quarter'].map(quarter_to_month)

# Build datetime
df_world['date'] = pd.to_datetime(df_world['year'].astype(str) + "-" + df_world['month'] + "-01")

df_world['value_z'] = df_world.groupby('Industry')['value'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

print(df_world['date'] )


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define color palette
palette = sns.color_palette('tab10', n_colors=df_world['Industry'].nunique())
industry_colors = dict(zip(df_world['Industry'].unique(), palette))

threshold_year = '2020-01-01'
stop_year = '2021-06-06'

plt.figure(figsize=(16,6))
ax_main = plt.gca()  # main axes

# ---- Main plot: full time series ----
for industry in df_world['Industry'].unique():
    subset = df_world[df_world['Industry'] == industry]
    color = industry_colors[industry]

    # Before 2020 → faded
    before = subset[subset['date'] < threshold_year]
    if not before.empty:
        sns.lineplot(data=subset, x='date', y='value', color=color, alpha=0.3, linewidth=2, ax=ax_main)
        sns.scatterplot(data=before, x='date', y='value', color=color,  alpha=0.3, legend=False, ax=ax_main)
    
    # Year 2020 → dashed
    year2020 = subset[(subset['date'] >= threshold_year) & (subset['date'] <= stop_year)]
    if not year2020.empty:
        sns.lineplot(data=year2020, x='date', y='value', color=color, alpha=0.5, linewidth=2, ax=ax_main)
        sns.scatterplot(data=year2020, x='date', y='value', color=color, alpha=0.5, legend=False, ax=ax_main)
    
    # After 2020 → solid strong
    after = subset[subset['date'] > stop_year]
    if not after.empty:
        sns.lineplot(data=after, x='date', y='value', label=industry, color=color, linewidth=2.5, ax=ax_main)
        sns.scatterplot(data=after, x='date', y='value', color=color, legend=False, ax=ax_main)

ax_main.set_ylabel("Greenhouse Gas Emissions (log-scale)", fontsize=13)
ax_main.set_xlabel('Year')
ax_main.set_title("World Greenhouse Gas Emissions per activity sector (2010-2025)", fontsize=13)
ax_main.set_yscale('symlog', linthresh=1)

ax_main.grid(True, which='both', linestyle='-', linewidth=0.7, alpha=0.5)
# ---- Expand main y-axis by 1 order of magnitude ----
ymin, ymax = ax_main.get_ylim()
ax_main.set_ylim(ymin, ymax * 1000)  # multiply upper limit by 10

# ---- Inset plot: normalized 2022-2024 ----
df_inset = df_world[(df_world['date'] >= stop_year) & (df_world['year'] <= 2024)].copy()
df_inset['value_z'] = df_inset.groupby('Industry')['value'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
covid_start = pd.to_datetime("2020-03-01")
covid_end   = pd.to_datetime("2021-06-01")

# Add shaded rectangle to main axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_main.axvspan(covid_start, covid_end, color='red', alpha=0.1, label='COVID period')
ax_main.legend(title=None, bbox_to_anchor=(0.75, 1), loc='upper left')
# Create inset axes inside main plot
ax_inset = inset_axes(ax_main, loc='upper center', width="65%", height="60%", bbox_to_anchor=(-0.05, 0.25, 0.75, 0.7), bbox_transform=ax_main.transAxes, borderpad=0)

for industry in df_inset['Industry'].unique():
    subset = df_inset[df_inset['Industry'] == industry]
    sns.lineplot(data=subset, x='date', y='value_z', label=industry, color=industry_colors[industry], linewidth=2, ax=ax_inset)

# Expand y-axis so it does not overlap with main plot

ax_inset.set_title("Post-Covid Greenhouse Emissions per activity sector", fontsize=10)
ax_inset.set_ylabel("Normalized emissions (min–max)")
ax_inset.set_xlabel("")
ax_inset.tick_params(axis='x', rotation=45)
ax_inset.set_ylim(-0.05, 1.05)
ax_inset.legend().remove()  # remove inset legend to avoid clutter
plt.tight_layout()
plt.show()

plt.clf()


plt.figure(figsize=(16,6))
sns.lineplot(data=df_world, x='date', y='value_z', hue='Industry')
plt.yticks([i/20 for i in range(0, 21)]) 
#plt.yscale('log')
plt.legend(title='Industry',
           bbox_to_anchor=(1.02, 1),   # move legend right
           loc='upper left')           # align to top-left of legend box

plt.tight_layout()
plt.show()

plt.clf()

# 2. Merge keys (everything except country)


df_cont['month'] = df_cont['quarter'].map(quarter_to_month)

# Build datetime
df_cont['date'] = pd.to_datetime(df_cont['year'].astype(str) + "-" + df_cont['month'] + "-01")


merge_keys = ['year', 'date', 'quarter', 'Industry', 'Gas Type', 'Seasonal Adjustment']

df_merged = df_cont.merge(
    df_world[merge_keys + ['value']].rename(columns={'value': 'world_value'}),
    on=merge_keys,
    how='left'
)

# 3. Percentage calculation


df_merged = df_merged.loc[df_merged['Industry']=='Total Industry and Households']

df_yearly = (
    df_merged.groupby(['year', 'Country'])[['value', 'world_value']]
    .mean()
    .reset_index())




df_merged['date'] = pd.to_datetime(df_merged['date'])
print(df_merged)
df_yearly['percent_of_world'] = (df_yearly['value'] / df_yearly['world_value']) * 100
sns.scatterplot(data=df_yearly, x='date', y='percent_of_world', hue='Country')
plt.ylabel('% of total greenhouse gas emissions')
plt.show()


df_pivot = df_yearly.pivot_table(
    index='year',
    columns='Country',
    values='percent_of_world'
).fillna(0)

df_pivot = df_pivot.apply(lambda row: row.sort_values(ascending=False), axis=1)
sns.set(style="whitegrid")  # seaborn style
print(df_pivot)
# Prepare colors from seaborn palette
countries = df_pivot.columns.tolist()
colors = sns.color_palette("tab10", n_colors=len(countries))

# Plot stacked bars
fig, ax = plt.subplots(figsize=(12,6))
bottom = pd.Series([0]*len(df_pivot), index=df_pivot.index)

for i, country in enumerate(countries):
    print(df_pivot[country])
    ax.bar(df_pivot.index, df_pivot[country], bottom=bottom, label=country, color=colors[i])
    bottom += df_pivot[country]
    
ax.set_ylabel('% of total greenhouse gas emissions', fontsize = 13)
ax.set_xlabel('Year')
ax.set_xticks(df_pivot.index)        # put a tick at every year in the index
ax.set_xticklabels(df_pivot.index)
ax.set_title("Continents' contribution to greenhouse gas emissions between 2010-2025 (households and industry)", fontsize = 13)
ax.legend(title='Continent', bbox_to_anchor=(1., 1), loc='upper left')
plt.tight_layout()

plt.show()
plt.clf()


