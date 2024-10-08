#!/usr/bin/env python
# coding: utf-8

# ### Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


# to supress warnings

import warnings 
warnings.filterwarnings('ignore')


# ### Load Datasets

# In[3]:


df_test = pd.read_csv(r"C:\Users\Prajat\Desktop\UA\testbc7185d.csv")
df_test


# In[4]:


df_sentiment = pd.read_csv(r"C:\Users\Prajat\Desktop\UA\sentiment_statisticscc1e57a.csv")
df_sentiment


# In[5]:


df_reasons = pd.read_csv(r"C:\Users\Prajat\Desktop\UA\reason18315ff.csv")
df_reasons


# In[6]:


df_customers = pd.read_csv(r"C:\Users\Prajat\Desktop\UA\customers2afd6ea.csv")
df_customers


# In[7]:


df_calls = pd.read_csv(r"C:\Users\Prajat\Desktop\UA\callsf0d4f5a.csv")
df_calls


# ### Combining Data

# In[8]:


df_combined1 = pd.merge(df_sentiment, df_reasons, on = 'call_id', how = 'outer')
df_combined1


# In[9]:


df_combined2 = pd.merge(df_combined1, df_calls, on = 'call_id', how = 'outer')
df_combined2


# In[10]:


df = pd.merge(df_combined2, df_customers, on = 'customer_id', how = 'left')
df


# ### Defining Combined Data

# #### 1. Display Top 5 Rows of the Dataset

# In[11]:


df.head(5)


# #### 2. Checking Last 3 Rows of the Dataset

# In[12]:


df.tail(3)


# #### 3. Finding Shape of the Dataset i.e, Number of Rows & Number of Columns¶

# In[13]:


df.shape


# In[14]:


print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# #### 4. Get Informaation about the Dataset like Total Number of Rows, Total Number of Columns, Datatypes of each Column and Memory Requirement

# In[15]:


df.info()


# In[16]:


df.dtypes


# #### 5. Dealing with Datatypes

# In[17]:


df.columns


# In[18]:


df.head(5)


# In[ ]:





# In[19]:


# df['call_start_datetime'] = df['call_start_datetime'].str.replace('-', '/')


# In[20]:


df['call_start_datetime'] = pd.to_datetime(df['call_start_datetime'], format = '%m/%d/%Y %H:%M')


# In[21]:


df['call_start_date'] = df['call_start_datetime'].dt.date
df['call_start_time'] = df['call_start_datetime'].dt.time


# In[22]:


df.head()


# In[23]:


df.columns


# In[ ]:





# In[24]:


# df['agent_assigned_datetime'] = df['agent_assigned_datetime'].str.replace('-', '/')


# In[25]:


df['agent_assigned_datetime'] = pd.to_datetime(df['agent_assigned_datetime'], format = '%m/%d/%Y %H:%M')


# In[26]:


df['agent_assigned_date'] = df['agent_assigned_datetime'].dt.date
df['agent_assigned_time'] = df['agent_assigned_datetime'].dt.time


# In[27]:


df.head()


# In[ ]:





# In[28]:


# df['call_end_datetime'] = df['call_end_datetime'].str.replace('-', '/')


# In[29]:


df['call_end_datetime'] = pd.to_datetime(df['call_end_datetime'], format = '%m/%d/%Y %H:%M')


# In[30]:


df['call_end_date'] = df['call_end_datetime'].dt.date
df['call_end_time'] = df['call_end_datetime'].dt.time


# In[31]:


df.head()


# In[ ]:





# #### Taking care of Null Values

# In[32]:


df.dtypes


# In[33]:


df.isnull().all()


# In[34]:


df.isnull().any()


# In[35]:


df.isnull().sum()


# In[36]:


df['agent_tone'].fillna('Unknown', inplace = True)


# In[37]:


df['primary_call_reason'].fillna('Unknown', inplace = True)


# In[38]:


df['average_sentiment'].describe()


# In[39]:


df['average_sentiment'].fillna(df['average_sentiment'].median(), inplace = True)


# In[40]:


df['elite_level_code'].fillna(df['elite_level_code'].median(), inplace = True)


# In[41]:


df['elite_level_code'] = df['elite_level_code'].astype(int)


# In[42]:


df.head(10)


# In[43]:


df.isnull().sum()


# #### Overall Statistics about Dataframe

# In[44]:


df.describe(include='object')


# In[45]:


for col in df.describe(include='object').columns:
    print(col)
    print(df[col].unique())
    print('-'*50)


# In[46]:


#summary statistics

df.describe(include = 'all')


# In[47]:


df.describe()


# Data Cleaning

# In[151]:


df['primary_call_reason'].unique()


# In[152]:


#removing extra spaces

df['primary_call_reason'] = df['primary_call_reason'].str.strip()


# In[160]:


#mapping similar values

cleaning_map = {
    'Voluntary  Cancel': 'Voluntary Cancel',
    'Voluntary   Cancel': 'Voluntary Cancel',
    'Voluntary Change': 'Voluntary Change',
    'Voluntary   Change': 'Voluntary Change',
    'Mileage   Plus': 'Mileage Plus',
    'Mileage  Plus': 'Mileage Plus',
    'Post-Flight': 'Post Flight',
    'Check-In': 'Check In',
    'Traveler   Updates': 'Traveler Updates',
    'Traveler  Updates': 'Traveler Updates',
    'Digital   Support': 'Digital Support',
    'Digital  Support': 'Digital Support',
    'Products & Services': 'Products and Services',
    'Other  Topics': 'Other Topics'
}


# In[161]:


df['primary_call_reason'] = df['primary_call_reason'].replace(cleaning_map)


# In[162]:


df['primary_call_reason'].unique()


# In[ ]:





# ### DELIVERABLES

# 1. Long average handle time (AHT) affects both efficiency and customer satisfaction. Explore the factors contributing to extended call durations, such as agent performance, call types, and sentiment. Identify key drivers of long AHT and AST, especially during high volume call periods. Additionally, could you quantify the percentage difference between the average handling time for the most frequent and least frequent call reasons?
# 

# ##### NOTE
# 
# **AHT (Average Handle Time)**:
# 
# Time from when the agent picks up the call to when they hang up
# 
# Formula: AHT = Total Handle Time / Total Number of Calls
# 
# 
# 
# 
# **AST (Average Speed to Answer)**:
# 
# Time spent by the customer in queue till the agent answers the call
# 
# Formula: AST = Total Waiting Time / Total Number of Calls

# In[48]:


df.columns


# Factors contributing to extended call durations:
#     
#     1. Agent Performance, Agent Tone and Agent Experience
#     2. Call Type
#     3. Sentiment
#     4. Primary Call Reason
#     5. Call Timing and Shift Timing? Is it peak hour or non peak hour?
#     6. Customer Tone
#     7. Customer Loyalty Status
#     8. Silence Percentage

# In[49]:


#AHT

df['handle_time'] = (df['call_end_datetime'] - df['agent_assigned_datetime']).dt.total_seconds()


# In[50]:


#AST

df['waiting_time'] = (df['agent_assigned_datetime'] - df['call_start_datetime']).dt.total_seconds()


# In[51]:


# factor_columns = ['agent_id_x', 'agent_tone', 'average_sentiment', 'primary_call_reason', 'customer_tone', 'elite_level_code',  'silence_percent_average']


# In[52]:


# for factor in factor_columns:
#    print(df.groupby(factor)[['handle_time', 'waiting_time']].mean())


# In[53]:


# factor_columns


# ### **Agent Performance**

# In[54]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean()
agent_performance


# In[55]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean().sort_values(by = ['handle_time'])
agent_performance


# **Analysis**: Handle time of Agent 5475592 is minimum while Handle time of Agent 102574 is maximum

# In[52]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean().sort_values(by = ['waiting_time'])
agent_performance


# In[82]:


handle_time = agent_performance['handle_time'].head(10)


# In[83]:


waiting_time = agent_performance['waiting_time'].head(10)


# In[84]:


mini_agent_performance = agent_performance.head(10)
mini_agent_performance


# In[91]:


#plotting the data

plt.figure(figsize = (20,8))
plt.title('Relation between Waiting and Handling Time of Agents',fontsize = 30)
plt.bar(agent_performance.head(10).index, handle_time, label = 'Handle Time')
plt.bar(agent_performance.head(10).index, waiting_time, label = 'Waiting Time')
plt.legend(fontsize = 20)

plt.xlabel('Agent ID', fontsize = 20)
plt.ylabel('Time (in seconds)', fontsize = 20)

plt.show()


# In[89]:


import matplotlib.pyplot as plt

# Select the top 10 agents with the highest handling time
top_10_agents = agent_performance.nlargest(10, 'handle_time')

# Extract handle time and waiting time for these agents
handle_time_top10 = top_10_agents['handle_time']
waiting_time_top10 = top_10_agents['waiting_time']

# Plotting
plt.figure(figsize=(20, 8))
plt.title('Relation between Waiting and Handling Time of Top 10 Agents', fontsize=30)

# Create stacked bar plot
plt.bar(top_10_agents.index, handle_time_top10, label='Handle Time')
plt.bar(top_10_agents.index, waiting_time_top10, bottom=handle_time_top10, label='Waiting Time')

# Add legend and labels
plt.legend(fontsize=20)
plt.xlabel('Agent ID', fontsize=20)
plt.ylabel('Time (in seconds)', fontsize=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()


# In[78]:


plt.figure(figsize=(10, 6))
plt.title('Relation between Waiting and Handling Time of Agents', fontsize = 20)

# Creating box plot
sns.boxplot(data=agent_performance[['handle_time', 'waiting_time']])

plt.xlabel('Metrics', fontsize = 15)
plt.ylabel('Time (in seconds)', fontsize = 15)

plt.show()


# ### **Agent Experience**

# In[56]:


#Agent Experience - Number of calls handled by each agent

df['agent_experience'] = df.groupby('agent_id_x')['call_id'].transform('count')


# In[57]:


# Group by agent experience levels and calculate AHT and AST

experience_groups = pd.qcut(df['agent_experience'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])


# In[58]:


df['experience_level'] = experience_groups


# In[59]:


experience_performance = df.groupby('experience_level')[['handle_time', 'waiting_time']].mean()
experience_performance


# In[61]:


import matplotlib.pyplot as plt

# Plotting a bar chart to compare handle time and waiting time by experience level
experience_levels = experience_performance.index
handle_time = experience_performance['handle_time']
waiting_time = experience_performance['waiting_time']

plt.figure(figsize=(10, 6))
plt.bar(experience_levels, handle_time, width=0.4, label='Handle Time', color='skyblue', align='center')
plt.bar(experience_levels, waiting_time, width=0.4, label='Waiting Time', color='orange', align='edge')

plt.title('Handle Time vs Waiting Time by Agent Experience Level', fontsize=18)
plt.xlabel('Experience Level', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.legend(fontsize=12)
plt.show()


# In[70]:


df.groupby('experience_level').count()


# Analysis: Can provide more trainings to agents who have medium and low experince level

# In[ ]:





# ### **Agent Tone**

# In[72]:


agent_tone_performance = df.groupby('agent_tone')[['handle_time', 'waiting_time']].mean()
agent_tone_performance


# In[75]:


import matplotlib.pyplot as plt

# Extracting data for plotting
agent_tones = agent_tone_performance.index
handle_time = agent_tone_performance['handle_time']
waiting_time = agent_tone_performance['waiting_time']

# Plotting Handle Time and Waiting Time for each agent tone
plt.figure(figsize=(10, 6))
plt.bar(agent_tones, handle_time, width=0.4, label='Handle Time', color='skyblue', align='center')
plt.bar(agent_tones, waiting_time, width=0.4, label='Waiting Time', color='orange', align='edge')

plt.title('Handle Time vs Waiting Time by Agent Tone', fontsize=18)
plt.xlabel('Agent Tone', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(fontsize=12)
plt.show()


# Sentiment

# In[77]:


sentiment_performance = df.groupby('average_sentiment')[['handle_time', 'waiting_time']].mean()
sentiment_performance


# In[78]:


import matplotlib.pyplot as plt

# Extracting data for plotting
sentiment = sentiment_performance.index
handle_time = sentiment_performance['handle_time']
waiting_time = sentiment_performance['waiting_time']

# Plotting Handle Time vs Sentiment and Waiting Time vs Sentiment
plt.figure(figsize=(10, 6))
plt.scatter(sentiment, handle_time, color='blue', label='Handle Time', s=100, alpha=0.6)
plt.scatter(sentiment, waiting_time, color='orange', label='Waiting Time', s=100, alpha=0.6)

# Adding titles and labels
plt.title('Sentiment vs Handle Time and Waiting Time', fontsize=18)
plt.xlabel('Average Sentiment', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.legend(title='Time Type', fontsize=12)
plt.grid(True)

# Show plot
plt.show()


# ### **Primary Call Reason**

# In[80]:


primary_reason_performance = df.groupby('primary_call_reason')[['handle_time', 'waiting_time']].mean()
primary_reason_performance


# In[82]:


import seaborn as sns
import matplotlib.pyplot as plt

# Reset the index for better readability in the heatmap
primary_reason_performance_reset = primary_reason_performance.reset_index()

# Create a heatmap for handle time and waiting time
plt.figure(figsize=(14, 8))
sns.heatmap(primary_reason_performance.pivot_table(values=['handle_time', 'waiting_time'], 
                                                   index=primary_reason_performance.index), 
            annot=True, cmap='Blues', linewidths=0.5, fmt='.1f')

plt.title('Average Handle Time and Waiting Time by Primary Call Reason', fontsize=18)
plt.xlabel('Time Metrics', fontsize=14)
plt.ylabel('Primary Call Reason', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Reset the index for easier plotting with seaborn
primary_reason_performance_reset = primary_reason_performance.reset_index()

# Melt the dataframe for better visualization
primary_reason_melted = primary_reason_performance_reset.melt(id_vars='primary_call_reason', 
                                                              value_vars=['handle_time', 'waiting_time'],
                                                              var_name='Time Type', 
                                                              value_name='Time')

# Plotting a box plot
plt.figure(figsize=(15, 8))
sns.boxplot(x='primary_call_reason', y='Time', hue='Time Type', data=primary_reason_melted, palette='Set2')

plt.title('Handle Time and Waiting Time by Primary Call Reason', fontsize=18)
plt.xlabel('Primary Call Reason', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.xticks(rotation=90)
plt.legend(title='Time Type', fontsize=12)
plt.show()


# ### **Call Timing and Shift Timing? Is it peak hour or non peak hour?**

# In[85]:


df['call_hour'] = df['call_start_datetime'].dt.hour


# In[91]:


peak_hour = df.groupby('call_start_time').count()
peak_hour


# In[106]:


# peak_hour.sort_values(by = 'call_id', ascending = False).head(10)


# In[109]:


timelist = peak_hour.sort_values(by='call_id', ascending=False).head(10).index.tolist()


# In[111]:


df['is_peaktime'] = df['call_start_time'].isin(timelist)


# In[113]:


df.columns


# In[120]:


timing_performance = df.groupby('is_peaktime')[['handle_time', 'waiting_time']].mean()
timing_performance


# In[123]:


import matplotlib.pyplot as plt

# Extract data for plotting
peak_status = ['Non-Peak', 'Peak']
handle_time = timing_performance['handle_time']
waiting_time = timing_performance['waiting_time']

# Creating a figure with two subplots for handle time and waiting time
plt.figure(figsize=(12, 6))

# Plot for Handle Time vs Peak Hour
plt.subplot(1, 2, 1)
plt.plot(peak_status, handle_time, marker='o', color='blue', label='Handle Time', linewidth=2)
plt.title('Peak Hour vs Handle Time', fontsize=16)
plt.xlabel('Peak Hour Status', fontsize=12)
plt.ylabel('Handle Time (in seconds)', fontsize=12)
plt.grid(True)

# Plot for Waiting Time vs Peak Hour
plt.subplot(1, 2, 2)
plt.plot(peak_status, waiting_time, marker='o', color='orange', label='Waiting Time', linewidth=2)
plt.title('Peak Hour vs Waiting Time', fontsize=16)
plt.xlabel('Peak Hour Status', fontsize=12)
plt.ylabel('Waiting Time (in seconds)', fontsize=12)
plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()


# ### **Customer Tone**

# In[125]:


customer_tone_performance = df.groupby('customer_tone')[['handle_time', 'waiting_time']].mean()
customer_tone_performance


# In[126]:


import matplotlib.pyplot as plt

# Extracting data for plotting
customer_tones = customer_tone_performance.index
handle_time = customer_tone_performance['handle_time']
waiting_time = customer_tone_performance['waiting_time']

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot for Handle Time
plt.plot(customer_tones, handle_time, marker='o', label='Handle Time', color='blue', linewidth=2)

# Plot for Waiting Time
plt.plot(customer_tones, waiting_time, marker='x', label='Waiting Time', color='orange', linewidth=2)

# Adding titles and labels
plt.title('Handle Time and Waiting Time by Customer Tone', fontsize=18)
plt.xlabel('Customer Tone', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)

# Add a legend
plt.legend(title='Time Type', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# ### **Customer Loyality Status**

# In[96]:


df.columns


# In[129]:


loyalty_status_performance = df.groupby('elite_level_code')[['handle_time', 'waiting_time']].mean()
loyalty_status_performance


# In[130]:


import matplotlib.pyplot as plt

# Extract data for plotting
elite_levels = loyalty_status_performance.index
handle_time = loyalty_status_performance['handle_time']
waiting_time = loyalty_status_performance['waiting_time']

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot for Handle Time
plt.plot(elite_levels, handle_time, marker='o', label='Handle Time', color='blue', linewidth=2)

# Plot for Waiting Time
plt.plot(elite_levels, waiting_time, marker='x', label='Waiting Time', color='orange', linewidth=2)

# Adding titles and labels
plt.title('Handle Time and Waiting Time by Loyalty Status (Elite Level)', fontsize=18)
plt.xlabel('Elite Level Code (Loyalty Status)', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.grid(True)

# Add a legend
plt.legend(title='Time Type', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# ### **Silenece Percentage**

# In[132]:


silence_performance = df.groupby(pd.cut(df['silence_percent_average'], bins = 4))[['handle_time', 'waiting_time']].mean()
silence_performance


# In[133]:


import matplotlib.pyplot as plt

# Extract data for plotting
silence_bins = silence_performance.index.astype(str)  # Convert the index to string for better labeling
handle_time = silence_performance['handle_time']
waiting_time = silence_performance['waiting_time']

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot for Handle Time
plt.plot(silence_bins, handle_time, marker='o', label='Handle Time', color='blue', linewidth=2)

# Plot for Waiting Time
plt.plot(silence_bins, waiting_time, marker='x', label='Waiting Time', color='orange', linewidth=2)

# Adding titles and labels
plt.title('Handle Time and Waiting Time by Silence Percentage in Call', fontsize=18)
plt.xlabel('Silence Percentage Range', fontsize=14)
plt.ylabel('Time (in seconds)', fontsize=14)
plt.grid(True)

# Add a legend
plt.legend(title='Time Type', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


# ### **Key drivers of long AHT and AST (During High Call Volume Periods)**

# In[134]:


# Calculating AHT and AST across the entire dataset

total_handle_time = df['handle_time'].sum()
total_calls = len(df)


# In[135]:


AHT = total_handle_time / total_calls
AST = df['waiting_time'].sum() / total_calls

print(f"Average Handle Time (AHT): {AHT} seconds")
print(f"Average Speed to Answer (AST): {AST} seconds")


# ### **Quantifying the Percentage Difference Between AHT for Most and Least Frequent Call Reasons**

# In[139]:


# Finding the most and least frequent call reasons

most_frequent_reason = df['primary_call_reason'].value_counts().idxmax()
least_frequent_reason = df['primary_call_reason'].value_counts().idxmin()


# In[140]:


# Calculating AHT for both

most_frequent_AHT = df[df['primary_call_reason'] == most_frequent_reason]['handle_time'].mean()
least_frequent_AHT = df[df['primary_call_reason'] == least_frequent_reason]['handle_time'].mean()


# In[141]:


# Calculating percentage difference

percentage_diff = ((most_frequent_AHT - least_frequent_AHT) / least_frequent_AHT) * 100

print(f"Percentage difference in AHT between most and least frequent call reasons: {percentage_diff:.2f}%")


# In[ ]:





# 2. We often observe self-solvable issues unnecessarily escalating to agents, increasing their workload. Analyse the transcripts and call reasons to identify granular reasons associated to recurring problems that could be resolved via self-service options in the IVR system. Propose specific improvements to the IVR options to effectively reduce agent intervention in these cases, along with solid reasoning to support your recommendations.
# 
# 
# Analysze:
# i. Primary Reason to Call
# ii. Call Transcript

# In[142]:


#importing library for the extraction of keywords from transcript

from sklearn.feature_extraction.text import CountVectorizer


# In[143]:


# Converting call transcripts into a document-term matrix

vectorizer = CountVectorizer(stop_words = 'english', max_features = 100)
X = vectorizer.fit_transform(df['call_transcript'])


# In[144]:


# Top keywords related to recurring issues

top_keywords = vectorizer.get_feature_names_out()
print("Top Keywords in Call Transcripts: ", top_keywords)


# In[163]:


# Group by primary call reason to identify recurring problems

call_reason_counts = df['primary_call_reason'].value_counts()
print(call_reason_counts)


# In[164]:


reason_counts = df['primary_call_reason'].value_counts()
print(reason_counts)
total_calls = len(df)
reason_percentages = (reason_counts / total_calls) * 100

other_reasons_percentage = reason_percentages[reason_percentages < 1].sum()
other_reasons_count = reason_counts[reason_percentages < 1].sum()
major_reasons = reason_percentages[reason_percentages >= 1]
major_reasons_counts = reason_counts[reason_percentages >= 1]

labels = list(major_reasons.index) + ['Other']
sizes = list(major_reasons_counts) + [other_reasons_count]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution of Call Reasons')
plt.show()


# In[166]:


self_solvable = [
    'Booking', 'Digital Support', 'Schedule Change', 'Products and Services', 'Other Topics'
]

agent_needed = [
    'IRROPS', 'Baggage', 'Upgrade', 'Mileage Plus', 'Checkout', 'Voluntary Cancel', 'Voluntary Change',
    'Post Flight', 'Check In', 'Communications', 'Disability', 'Unaccompanied Minor', 'Traveler Updates', 'ETC'
]


# In[167]:


df['issue_type'] = df['primary_call_reason'].apply(
    lambda x: 'self-solvable' if x in self_solvable else 'agent-needed'
)


# In[168]:


df['issue_type'].value_counts()


# In[ ]:





# In[171]:



import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Preprocessing function to clean the transcript text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = text.lower().strip()  # Lowercase and remove surrounding spaces
    return text

df['call_transcript'] = df['call_transcript'].apply(preprocess_text)

# Vectorizing the transcript using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['call_transcript'])


# In[172]:


# Apply KMeans clustering to the TF-IDF matrix
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Checking how clusters correspond to call reasons
cluster_summary = df.groupby('cluster')['primary_call_reason'].value_counts()
print(cluster_summary)


# In[ ]:


# Applying LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# Get the top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

tfidf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda, tfidf_feature_names, 10)


# In[174]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Applying sentiment analysis to the transcripts
df['sentiment'] = df['transcript'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])

# Checking average sentiment by cluster and call reason
sentiment_summary = df.groupby('cluster')['sentiment'].mean()
print(sentiment_summary)


# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming the dataframe 'df' contains the following columns:
# - 'primary_call_reason'
# - 'sentiment'
# - 'silence_percent_average'
# - 'customer_tone'
# - 'handle_time'
# - 'elite_level_code' (Customer Loyalty Status)

# Step 1: Clean and preprocess text-based features
df['primary_call_reason'] = df['primary_call_reason'].str.strip()  # Clean up call reason

# Apply sentiment analysis if sentiment is not available (if sentiment is not pre-calculated, use a sentiment analyzer)
# Assuming 'sentiment' is already a column, otherwise use a sentiment analyzer here

# Step 2: Encode categorical variables (Primary Call Reason and Customer Tone)
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_call_reason = one_hot_encoder.fit_transform(df[['primary_call_reason']])
encoded_customer_tone = one_hot_encoder.fit_transform(df[['customer_tone']])

# Step 3: Combine all features into a single feature matrix
X = pd.concat([
    pd.DataFrame(encoded_call_reason),  # One-hot encoded primary call reason
    pd.DataFrame(encoded_customer_tone),  # One-hot encoded customer tone
    df[['sentiment', 'silence_percent_average', 'handle_time', 'elite_level_code']]  # Numerical features
], axis=1)

# Step 4: Standardize the numerical features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply dimensionality reduction (PCA) for visualization purposes (optional but useful for visualizing clusters)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[ ]:


# Step 6: Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Step 7: Visualize the clusters using PCA-reduced data
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', s=50)
plt.title('Cluster Visualization for Call Reasons')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

# Check the cluster composition by primary call reason
cluster_composition = df.groupby('cluster')['primary_call_reason'].value_counts()
print(cluster_composition)


# In[ ]:


# Check how the features like sentiment, handle time, etc., are distributed across clusters
cluster_summary = df.groupby('cluster')[['sentiment', 'silence_percent_average', 'handle_time', 'elite_level_code']].mean()
print(cluster_summary)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[165]:


# Filter most frequent call reasons and cross-check with transcripts for self-solvable patterns

self_solvable_reasons = df[df['primary_call_reason'].isin(call_reason_counts.index[:5])]
print(self_solvable_reasons[['primary_call_reason', 'call_transcript']].head())


# In[ ]:





# In[ ]:





# In[119]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[120]:


# Extract TF-IDF features from the call transcripts

vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 500)
X_transcripts = vectorizer.fit_transform(df['call_transcript'])


# In[121]:


# Display the top words based on TF-IDF

top_words = vectorizer.get_feature_names_out()
print(top_words[:10])  # Preview of top words


# In[122]:


# Analyze average sentiment and tone distributions per call reason

sentiment_analysis = df.groupby('primary_call_reason')['average_sentiment'].mean()
tone_analysis = df.groupby('primary_call_reason')[['agent_tone', 'customer_tone']].value_counts()

print(sentiment_analysis)
print(tone_analysis)


# In[123]:


# Extract hour, weekday, and month from the call_start_datetime column

df['call_hour'] = df['call_start_datetime'].dt.hour
df['call_weekday'] = df['call_start_datetime'].dt.weekday
df['call_month'] = df['call_start_datetime'].dt.month


# In[124]:


# Analyze how call times relate to call reasons

time_analysis = df.groupby(['call_hour', 'primary_call_reason']).size().unstack(fill_value=0)
print(time_analysis)


# In[126]:


df.columns


# In[128]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Example: Convert the sparse matrix to dense and then to DataFrame
X_transcripts_dense = pd.DataFrame(X_transcripts.toarray())

# Combine the dense matrix with the DataFrame
X = pd.concat([X_transcripts_dense, df[['average_sentiment', 'call_hour', 'elite_level_code']]], axis=1)

# Target variable
y = df['primary_call_reason']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




