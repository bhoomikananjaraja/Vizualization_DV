#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import re
import plotly.graph_objects as go
import plotly.subplots as ps
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import kstest,shapiro,normaltest
import statsmodels.api as sm
import warnings


# In[24]:


##Reading the data
df = pd.read_csv('/Users/bhoomikan/Documents/Reza_Dataviz/Project/IncidentFile_APP.csv')


# In[25]:


###Label encoding incident_state variable and filling the missing values in sys_created_at column and convert the temporal datetime columns to a proper format and extracting month,year and day
map_incidentstate = {'New':1,'Active':2,'Awaiting Problem':3,'Awaiting User Info':4,'Awaiting Vendor':5,'Awaiting Evidence':6,'Resolved':7,'Closed':8}
df['incident_state_mapped'] = df['incident_state'].map(map_incidentstate)
df['sys_created_at'] = df['sys_created_at'].replace('?', np.nan)
grouped = df.groupby('number')
min_incident_state_mapped = grouped['incident_state_mapped'].min()

corresponding_sys_updated_at = grouped.apply(lambda group: group.loc[group['incident_state_mapped'].idxmin(), 'sys_updated_at'])
for incident_number, sys_updated_at in corresponding_sys_updated_at.items():
    df.loc[df['number'] == incident_number, 'sys_created_at'].fillna(sys_updated_at, inplace=True)
    
temporal_cols = ['opened_at','sys_created_at','sys_updated_at','resolved_at','closed_at']
for col in temporal_cols:
    fname = f"{col}_custom"
    df[fname] = pd.to_datetime(df[col], infer_datetime_format=True).dt.strftime('%Y-%m-%d %H:%M')
    
df['Month_Opened'] = pd.to_datetime(df['opened_at_custom']).dt.month
df['Day_Opened'] = pd.to_datetime(df['opened_at_custom']).dt.day_name()
df['Year_Opened'] = pd.to_datetime(df['opened_at_custom']).dt.year

df['Month_Closed'] = pd.to_datetime(df['closed_at_custom']).dt.month
df['Day_Closed'] = pd.to_datetime(df['closed_at_custom']).dt.day_name()
df['Year_Closed'] = pd.to_datetime(df['closed_at_custom']).dt.year

map_dayname = {'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Friday':6,'Saturday':7}
df['dayopened_mapped'] = df['Day_Opened'].map(map_dayname)
df['dayclosed_mapped'] = df['Day_Closed'].map(map_dayname)


# In[ ]:


df.head(10)


# In[26]:


##Calculation of completion_time_days
df['completion_time_days'] = abs(round((pd.to_datetime(df['closed_at_custom']) - pd.to_datetime(df['opened_at_custom'])).dt.total_seconds()/3600/24))
df = df[df['incident_state']!='-100']

df.to_csv('IncidentFile_APP.csv',header=True)
# In[7]:


###Function To Show the Distribution Plots With and Without Outliers

total_len = len(df)

def dist_plots(dfn,f,flag,title):
    if flag == 0:
        dfn = dfn
    else:
        Q1 = np.percentile(df[f], 25)
        Q3 = np.percentile(df[f], 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        num_len = len(dfn[(dfn[col] < lower_bound) | (dfn[col] > upper_bound)])
        print(f"% of rows to be dropped because of outliers for {col}\n")
        print((num_len/total_len)*100)
        
        dfn = dfn[(dfn[f] >= lower_bound) & (dfn[f] <= upper_bound)]
    
    sns.set(style="whitegrid", palette="muted", color_codes=True)

    # Create a 2x2 subplot matrix
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Dist Plot
    sns.distplot(dfn[f], kde=False, ax=axes[0, 0], color='m')
    axes[0, 0].set_title(f'Distribution Plot for: {f} [{title}]')

    # Histogram with KDE
    sns.histplot(dfn[f], kde=True, ax=axes[0, 1], color='b')
    axes[0, 1].set_title(f'Histogram with KDE for: {f} [{title}]')

    # QQ-Plot
    stat.probplot(dfn[f], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'QQ Plot for: {f} [{title}]')

    # KDE Plot with Fill
    sns.kdeplot(dfn[f], ax=axes[1, 1], color='g', fill=True, alpha=0.6, linewidth=2)
    axes[1, 1].set_title(f'KDE Plot with Fill for: {f} [{title}]')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# In[8]:


num_cols = ['completion_time_days','reassignment_count','reopen_count','sys_mod_count']
for col in num_cols:
    dist_plots(df,col,0,'With Outliers')
    dist_plots(df,col,1,'Without Outliers')


# In[27]:


def outlier(df,f):
    Q1 = np.percentile(df[f], 25)
    Q3 = np.percentile(df[f], 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    cleaned_df_cd = df[(df[f] >= lower_bound) & (df[f] <= upper_bound)]  
    
    return cleaned_df_cd


# In[28]:


odf = df
dfn = df
num_cols = ['reassignment_count','reopen_count','sys_mod_count','completion_time_days']
for col in num_cols:
    dfn = outlier(dfn,col)


# In[21]:


###
dfn.head(10)
dfn.describe()


# In[29]:


def plotly_plot(f,merged_df):
    fig = go.Figure()

                # Add bar plot for unique_incident_count
    fig.add_trace(go.Bar(
                    x=merged_df[f],
                    y=merged_df['unique_incident_count'],
                    name='Unique Incident Count'
                ))

                # Add line plot for completion_time_days (avg)
    fig.add_trace(go.Scatter(
                    x=merged_df[f],
                    y=merged_df['completion_time_days (avg)'],
                    mode='lines',
                    name='Completion Time Days (Avg)',
                    yaxis='y2'
                ))

                # Update layout
    fig.update_layout(
                    title=f'Unique Incident Count and Completion Time Days (avg) by {f}',
                    xaxis=dict(title=f'Feature {f}'),
                    yaxis=dict(title='Unique Incident Count'),
                    yaxis2=dict(title='Completion Time Days (Avg)', overlaying='y', side='right'),
                    legend=dict(x=0, y=1.1, traceorder='normal')
                )

                # Show plot
    fig.show()


def tb_plot(f,merged_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    # Bar plot with line plot
    sns.barplot(data=merged_df, x=f, y='unique_incident_count', ax=axes[0], color='skyblue')
    axes[0].set_title(f'Unique Incident Count and Completion Time Days (avg) by {f}')
    axes[0].set_ylabel('Unique Incident Count')
    axes[0].set_xlabel(f)
    # Adding a secondary axis for the line plot
    ax2 = axes[0].twinx()
    sns.lineplot(data=merged_df, x=f, y='completion_time_days (avg)', ax=ax2, color='red', marker='o')
    ax2.set_ylabel('Completion Time Days (avg)')

    # Box plot
    sns.boxplot(data=dfn, x=f, y='completion_time_days', ax=axes[1])
    axes[1].set_title(f'Box Plot of Completion Time Days (avg) by {f}')
    axes[1].set_ylabel('Completion Time Days (avg)')
    axes[1].set_xlabel(f)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

cols = ['priority','knowledge','contact_type','impact','urgency','notify','made_sla','u_priority_confirmation','u_symptom_num','category','subcategory','closed_code','assignment_group_num','assigned_to_num','location','resolved_by_num']
for f in cols:
    if f in ['made_sla','u_priority_confirmation']:
        closeddf = dfn[dfn['incident_state'] == 'Closed']
        closeddf = closeddf.drop_duplicates(subset='number')
        proportions_df = closeddf.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
        median_response_time_df = closeddf.groupby([f])['completion_time_days'].median().reset_index(name='completion_time_days (avg)')
        merged_df = pd.merge(proportions_df, median_response_time_df, on=f, how='inner')
        tb_plot(f,merged_df)
    elif f in ['u_symptom_num','category','subcategory','closed_code','assignment_group_num','assigned_to_num','location','resolved_by_num']:
        proportions_df = dfn.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
        median_response_time_df = dfn.groupby([f])['completion_time_days'].median().reset_index(name='completion_time_days (avg)')
        merged_df = pd.merge(proportions_df, median_response_time_df, on=f, how='inner')
        plotly_plot(f,merged_df)
    else:
        proportions_df = dfn.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
        median_response_time_df = dfn.groupby([f])['completion_time_days'].median().reset_index(name='completion_time_days (avg)')
        merged_df = pd.merge(proportions_df, median_response_time_df, on=f, how='inner')
        tb_plot(f,merged_df)


# In[30]:


dfn.info()


# In[101]:


##Grouped Bar Plot For Priority By Knowledge Base And Stacked Bar Plot For Priority By SLA
def proportions_d(f):
    prop_df = dfn.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
    return prop_df

prop_df = proportions_d(['priority', 'knowledge'])
pivot_df = prop_df.pivot_table(index='priority', columns='knowledge', values='unique_incident_count', fill_value=0)

# Setup subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 2 rows, 1 column, with custom figure size

# Plotting the first subplot (Grouped Bar Plot)
ind = np.arange(len(pivot_df))  # the x locations for the groups
width = 0.35  # the width of the bars

# Generate bars for each knowledge base in the first subplot
for i, col in enumerate(pivot_df.columns):
    ax1.bar(ind - width/2 + i*width, pivot_df[col], width, label=col)

ax1.set_xlabel('Priority')
ax1.set_ylabel('Unique Incident Count')
ax1.set_title('Grouped Unique Incident Counts by Priority and Knowledge Base')
ax1.set_xticks(ind)
ax1.set_xticklabels(pivot_df.index)
ax1.legend(title='Knowledge Base')

# Second subplot data
prop_df = proportions_d(['priority', 'made_sla'])
pivot_df = prop_df.pivot_table(index='priority', columns='made_sla', values='unique_incident_count', fill_value=0)

# Base for stacking the bars in the second subplot
bottom = np.zeros(len(pivot_df))

# Generate bars for each 'made_sla' category in the second subplot
for col in pivot_df.columns:
    ax2.bar(pivot_df.index, pivot_df[col], bottom=bottom, label=col)
    bottom += pivot_df[col]

ax2.set_xlabel('Priority')
ax2.set_ylabel('Unique Incident Count')
ax2.set_title('Stacked Unique Incident Counts by Priority and Made_SLA')
ax2.legend(title='Made_SLA')

plt.grid(True)
# Adjust layout to prevent overlap and make everything fit
fig.tight_layout()

# Show the combined plot
plt.show()
#prop_df = proportions_d(['priority','knowledge'])


# In[104]:


fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# Boxenplot for priority
sns.boxenplot(x='priority', y='completion_time_days', data=dfn, ax=axs[0])
axs[0].set_title('Distribution Of Completion Time Days by Priority')
axs[0].set_xlabel('Priority')
axs[0].set_ylabel('Completion Time Days')
axs[0].grid(True) 
# Boxenplot for made_sla
sns.boxenplot(x='made_sla', y='completion_time_days', data=dfn, ax=axs[1])
axs[1].set_title('Distribution of Completion Time Days by SLA')
axs[1].set_xlabel('SLA')
axs[1].set_ylabel('Completion Time Days')
axs[1].grid(True) 
# Boxenplot for knowledge
sns.boxenplot(x='knowledge', y='completion_time_days', data=dfn, ax=axs[2])
axs[2].set_title('Distribution of Completion Time Days by Knowledge')
axs[2].set_xlabel('Knowledge')
axs[2].set_ylabel('Completion Time Days')
axs[2].grid(True) 

plt.grid(True)
# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# In[114]:


states = ['Awaiting User Info', 'Awaiting Problem', 'Awaiting Vendor', 'Awaiting Evidence']
data_df = dfn[dfn['incident_state'].isin(states)]

df_inc = proportions_d(data_df['incident_state'])

min_state_cindex = df_inc['unique_incident_count'].idxmin()
explode_state = [0.4 if i == min_state_cindex else 0 for i in range(len(df_inc))]

percent1 = 100.*df_inc['unique_incident_count']/df_inc['unique_incident_count'].sum()
x1 = df_inc['incident_state']

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='incident_state', data=data_df)
plt.title('Countplot of Incident_State')
plt.xlabel('Incident_State')
plt.ylabel('Frequency')
plt.xticks(rotation=45) 
plt.grid(True)

plt.subplot(1,2,2)
patches, texts = plt.pie(df_inc['unique_incident_count'],explode=explode_state)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x1, percent1)]
plt.title('Distribution Of Incident State By Percentage')
plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),fontsize=8)

plt.tight_layout()
plt.show()


# In[142]:


sns.catplot(x='priority',y='completion_time_days' ,data=dfn,kind='violin')
plt.title('Completion_Time_Days of Priority')
plt.xlabel('Priority')
plt.ylabel('Completion_Time_Days')
plt.grid(True)
plt.tight_layout()
plt.show()

sns.catplot(x='knowledge',y='completion_time_days' ,data=dfn,kind='strip')
plt.title('Completion_Time_Days of Knowledge')
plt.xlabel('Knowledge')
plt.ylabel('Completion_Time_Days')
plt.grid(True)
plt.tight_layout()
plt.show()

'''sns.catplot(x='knowledge',y='completion_time_days' ,data=dfn,kind='swarm')
plt.title('Completion_Time_Days of Knowledge')
plt.xlabel('Knowledge')
plt.ylabel('Completion_Time_Days')
plt.grid(True)
plt.tight_layout()
plt.show()'''


# In[155]:


##Line-Plot
##Area Plot
df_mprio = dfn[dfn['priority_num']==3].loc[:,['number','incident_state','opened_at_custom','sys_updated_at_custom']].reset_index(drop=True)
df_mprio['state_completion_time_hr'] = abs(round((pd.to_datetime(df_mprio['sys_updated_at_custom']) - pd.to_datetime(df_mprio['opened_at_custom'])).dt.total_seconds()/3600))

df_lprio = dfn[dfn['priority_num']==4].loc[:,['number','incident_state','opened_at_custom','sys_updated_at_custom']].reset_index(drop=True)
df_lprio['state_completion_time_hr'] = abs(round((pd.to_datetime(df_mprio['sys_updated_at_custom']) - pd.to_datetime(df_mprio['opened_at_custom'])).dt.total_seconds()/3600))
df_lprio_plot = df_lprio[df_lprio['incident_state']!='New']
median_completion_time_lprio = df_lprio_plot.groupby('incident_state')['state_completion_time_hr'].median().reset_index()

df_mprio_plot = df_mprio[(df_mprio['incident_state']!='New')]
median_completion_time_mprio = df_mprio_plot.groupby('incident_state')['state_completion_time_hr'].median().reset_index()

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)  
plt.plot(median_completion_time_lprio['incident_state'], 
         median_completion_time_lprio['state_completion_time_hr'], 
         marker='o', linestyle='-')
plt.title('Median Completion Time Over Time (Lower Priority)')
plt.xlabel('Incident State')
plt.ylabel('Median Completion Time (Hours)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()

plt.subplot(1, 2, 2)  # 2 rows, 1 column, subplot 2
plt.plot(median_completion_time_mprio['incident_state'], 
         median_completion_time_mprio['state_completion_time_hr'], 
         marker='o', linestyle='-', label='Median Completion Time')
plt.title('Median Completion Time Over Time (Moderate Priority)')
plt.xlabel('Incident State')
plt.ylabel('Median Completion Time (Hours)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# In[156]:


dfn.head(10)


# In[161]:


trenddf = dfn[['number','Month_Opened','Day_Opened','dayopened_mapped','Year_Opened','Month_Closed','dayclosed_mapped','Year_Closed', 'completion_time_days']]
x_tick_values = trenddf['dayopened_mapped'].unique()
x_tick_values2 = trenddf['Month_Opened'].unique()

# Calculate the counts of incidents opened and closed for each year, month, and day
opened_counts = trenddf.groupby(['Year_Opened', 'Month_Opened', 'dayopened_mapped']).size().reset_index(name='Total_Incidents_Opened')
closed_counts = trenddf.groupby(['Year_Closed', 'Month_Closed', 'dayclosed_mapped']).size().reset_index(name='Total_Incidents_Closed')

# Merge the opened and closed counts DataFrames
merged_counts = pd.merge(opened_counts, closed_counts, how='outer', left_on=['Year_Opened', 'Month_Opened', 'dayopened_mapped'], right_on=['Year_Closed', 'Month_Closed', 'dayclosed_mapped'])
merged_counts.fillna(0, inplace=True)

# Simplifying to show the plots by Month_Opened (you should adjust for real datetime management)
merged_counts['Month'] = merged_counts['Month_Opened'].fillna(merged_counts['Month_Closed'])
merged_counts['Day'] = merged_counts['dayopened_mapped'].fillna(merged_counts['dayclosed_mapped'])

merged_counts = merged_counts[(merged_counts['Year_Opened'] != 0.0) & (merged_counts['Year_Closed'] != 0.0)]
# Plot the area chart
plt.figure(figsize=(12, 12))

plt.subplot(2,1,1)
plt.fill_between(merged_counts['Month'].sort_values().unique(), merged_counts.groupby('Month')['Total_Incidents_Opened'].sum(), color="skyblue", alpha=0.4, label='Total Incidents Opened')
plt.fill_between(merged_counts['Month'].sort_values().unique(), merged_counts.groupby('Month')['Total_Incidents_Closed'].sum(), color="olive", alpha=0.5, label='Total Incidents Closed')
plt.title('Opened vs Closed by Month')
plt.xlabel('Month')
plt.ylabel('Total Incidents')
plt.legend()
plt.xticks(range(1, len(x_tick_values2) + 1),rotation=45)


plt.subplot(2,1,2)
plt.fill_between(merged_counts['Day'].sort_values().unique(), merged_counts.groupby('Day')['Total_Incidents_Opened'].sum(), color="skyblue", alpha=0.4, label='Total Incidents Opened')
plt.fill_between(merged_counts['Day'].sort_values().unique(), merged_counts.groupby('Day')['Total_Incidents_Closed'].sum(), color="olive", alpha=0.5, label='Total Incidents Closed')
plt.title('Opened vs Closed by Day')
plt.xlabel('Day')
plt.ylabel('Total Incidents')
plt.legend()
plt.xticks(range(1, len(x_tick_values) + 1), x_tick_values,rotation=45)

plt.tight_layout()
plt.show()


# In[163]:


average_completion_time = trenddf.groupby(['Year_Opened', 'Month_Opened', 'Day_Opened'])['completion_time_days'].median().reset_index(name='Average_Completion_Time')

plt.subplot(2,1,1)
sns.lineplot(data=average_completion_time, x='Day_Opened', y='Average_Completion_Time', marker='o', palette='viridis', ci=None,label='Average_Completion_Time')
plt.title('Average_Completion_Time (Day)')
plt.xlabel('Day')
plt.ylabel('Total_Incidents')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2,1,2)
sns.lineplot(data=average_completion_time, x='Month_Opened', y='Average_Completion_Time', marker='o', palette='viridis', ci=None,label='Average_Completion_Time')
plt.title('Average_Completion_Time (Day)')
plt.xlabel('Month')
plt.ylabel('Total_Incidents')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[32]:


def top_c(f):
    topcontri = dfn
    topcontri[f] = topcontri[f].astype(str)
    proportions_df = dfn.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
    median_response_time_df = dfn.groupby(f)['completion_time_days'].median().reset_index(name='completion_time_days (avg)')
    merged_df = pd.merge(proportions_df, median_response_time_df, on=f, how='inner')
    merged_df_sorted = merged_df.sort_values(by='completion_time_days (avg)', ascending=True)
    merged_df_sorted = merged_df_sorted[0:40]

    plt.figure(figsize=(20, 10))
    plt.bar(merged_df_sorted[f],merged_df_sorted['completion_time_days (avg)'])
    plt.xlabel(f'{f}')
    plt.ylabel('Average Completion Time(Days)')
    plt.title(f'Top Contributors: {f}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[33]:


top_c('assigned_to_num')


# In[12]:


continuous_features = ['reassignment_count', 'reopen_count', 'sys_mod_count', 'completion_time_days']
col_h = ['incident_state_encoded','knowledge_encoded', 'contact_type_encoded', 'impact_num', 'urgency_num', 'priority_num', 'notify_encoded', 'closed_code_num', 'category_num', 'subcategory_num', 'assignment_group_num', 'resolved_by_num','assigned_to_num','location_num', 'made_sla_encoded', 'u_priority_confirmation_encoded','dayopened_mapped','dayclosed_mapped','Month_Opened','Month_Closed']
hmap = continuous_features + col_h
cmap = ['completion_time_days','reassignment_count', 'sys_mod_count']
#print(cols)

def heatmap(df,flag):
    corr_matrix = df.corr()
    if flag==0:
    # Create heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap for All The Features',fontsize=20)
        plt.show()
    else:
        plt.figure(figsize=(15, 15))
        sns.clustermap(corr_matrix, cmap='coolwarm',annot=True, fmt=".2f", linewidths=1.0, cbar_kws={"shrink": 0.2})
        plt.title('Cluster Map of Incident Analysis Dataset',fontsize=5)
        plt.show()
 
hdf = dfn[hmap]
heatmap(hdf,0)
cdf = dfn[cmap]
heatmap(cdf,1)

'''sns.set(rc={'figure.figsize':(15,10)})

# Create a clustermap with adjusted features
g = sns.clustermap(hdf, figsize=(15, 10), cmap='coolwarm', annot=False)

# Rotate x-axis labels
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')

# Adjust the size of the dendrogram
g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_position([0.05, 0.8, 0.2, 0.12])

# Show the clustermap
plt.show()'''

#heatmap(hdf)


# In[19]:


#%%
from prettytable import PrettyTable

# Create a PrettyTable object
table = PrettyTable()
table.align = "l"
# Define column names and alignment
table.field_names = ["Variable Pair", "Correlation Coefficient", "Relationship Strength", "Observations from Scatter Plot"]
table.align["Variable Pair"] = "l"
table.align["Correlation Coefficient"] = "r"
table.align["Relationship Strength"] = "l"
table.align["Observations from Scatter Plot"] = "l"

# Add data to the table
table.add_row(["reassignment_count and sys_mod_count", 0.5, "Moderate Positive", "Incidents that are reassigned more often have more changes."])
table.add_row(["priority and impact_num", 0.89, "Very Strong Positive", "Higher impact incidents tend to have higher priority."])
table.add_row(["opened_by_num and sys_created_by_num", 0.91, "Very Strong Positive", "Incidents are often opened and created by the same person."])
table.add_row(["resolved_by_num and closed_code_num", 0.94, "Very Strong Positive", "The resolver of an incident often uses consistent closure codes."])
table.add_row(["made_sla_encoded and u_priority_confirmation_encoded", -0.061, "Weak Negative", "Incidents meeting the SLA are less likely to need priority confirmation."])
table.add_row(["Month_Opened and Month_Closed", "-0.098", "Weak Negative", "There might be a seasonal pattern or trend in incident management."])
table.max_width["Observations from Scatter Plot"] = 50

# Print the table
print(table)
# Print the table
# %%


# In[190]:


def feature_imp(rf_classifier,X_train,Target):
    feature_importances = rf_classifier.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = feature_importances.argsort()[::-1]

    # Plot feature importances
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
    plt.xticks(range(len(feature_importances)), X_train.columns[sorted_indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Random Forest Feature Importance for: {Target}')
    plt.show()


#X = df_train.drop(['assigned_to_num'], axis=1)
#y = df_train['assigned_to_num']
def rmodel_acc(df_train,target,flag):
    X = df_train.drop(target,axis=1)
    y = df_train[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #X_missing = df_test
    # Initialize and train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    if flag ==1:
        feature_imp(rf_model, X_train,target)

rmodel_acc(hdf,'completion_time_days',1)


# In[26]:


othercolum = ['reassignment_count','reopen_count','made_sla_encoded','category_num','impact_num','urgency_num','priority_num','closed_code_num']

plt.figure(figsize=(10, 8))
sns.pairplot(data=dfn[othercolum])
plt.title('Pair Plot')
plt.tight_layout()
plt.show()


# In[27]:


idencolum = ['opened_by_num','sys_created_by_num','sys_updated_by_num','assignment_group_num','assigned_to_num','resolved_by_num']  
plt.figure(figsize=(10, 8))
sns.pairplot(data=dfn[idencolum])
plt.title('Pair Plot')
plt.tight_layout()
plt.show()


# In[39]:


plt.figure(figsize=(10, 8))
sns.jointplot(data=dfn, x='sys_created_by_num', y='opened_by_num', kind='kde')
plt.title('Joint Plot For CreatedBy Vs OpenedBy')
plt.tight_layout()
plt.show()

sns.jointplot(data=dfn, x='priority_num', y='completion_time_days', hue='urgency')
plt.title('Joint Plot For Priority Vs Completion_Time_Days By Urgency')
plt.tight_layout()
plt.show()

sns.rugplot(data=dfn, x='priority_num',y='completion_time_days', hue='urgency',height=0.1, linewidth=2, alpha=0.5)
plt.title('Rug Plot')
plt.tight_layout()
plt.show()

sns.lmplot(data=dfn, x='reassignment_count', y='sys_mod_count')
plt.title('LM Plot')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hexbin(dfn['Month_Opened'],dfn['completion_time_days'],gridsize=30, cmap='viridis')
plt.colorbar(label='count')
plt.xlabel('Month_Opened')
plt.ylabel('Completion Time (Days)')
plt.title('Hexbin Plot')
plt.show()


# In[41]:


###Normality Test

ks_test_completiontime = kstest(dfn['completion_time_days'], 'norm', args=(dfn['completion_time_days'].mean(), dfn['completion_time_days'].std()))

# Perform Shapiro-Wilk Test for normality
shapiro_test_completiontime = shapiro(dfn['completion_time_days'])


# Perform D'Agostino's K² Test for normality
statistic, p_value = normaltest(dfn['completion_time_days'])

table = PrettyTable()

# Set the column names
table.field_names = ["Normality Test", "Statistic", "p-value", "Result"]

# Populate the table with results
table.add_row(["K-S Test Completion_Time (Days)", f"{ks_test_completiontime.statistic:.2f}", f"{ks_test_completiontime.pvalue:.2e}", "Not Normal"]),
table.add_row(["Shapiro Test Completion_Time (Days)", f"{shapiro_test_completiontime [0]:.2f}", f"{shapiro_test_completiontime[1]:.2e}", "Not Normal"])
table.add_row(["D'Agostino's K² Completion_Time (Days)", f"{statistic:.2f}", f"{p_value:.2e}", "Not Normal"])

table.align = "r"
table.title="Normality Test For Completion_Time (Days)"
print(table)

qq_data = stat.probplot(dfn[f].dropna(), dist="norm")

# Create Q-Q plot
plt.figure(figsize=(8, 6))
plt.scatter(qq_data[0][0], qq_data[0][1], alpha=0.5)
plt.title("Q-Q Plot")
plt.xlabel("Theoretical quantiles")
plt.ylabel("Ordered Values")
plt.grid(True)
plt.plot([min(qq_data[0][0]), max(qq_data[0][0])], [min(qq_data[0][1]), max(qq_data[0][1])], color='red', linestyle='--')
plt.show()


# In[ ]:




