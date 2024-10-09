#!/usr/bin/env python
# coding: utf-8

# # **SKYHACK 2.0**
# 
# 
# ### **Team Name**: Jet Set Analyst
# ### **Team Members Name**: Heena Saini and Kavya Sethia
# ### **College Name**: IGDTUW
#                                                                                                                                                                                                                                                                  
# 

# ## Problem Statement

# As United Airlines continues its journey to become the best airline in the history of aviation, it is crucial to provide world-class customer service, for which one of the key areas of focus is our call center operations. Call centers play a critical role in ensuring customer issues are resolved quickly and efficiently, but we face challenges in improving metrics such as Average Handle Time (AHT) and Average Speed to Answer (AST).
# 
# Your task is to optimize these key call center metrics, helping reduce resolution times and providing faster, more efficient service to our customers. You are required to analyze our existing call center data to identify inefficiencies, determine the drivers of long AHT and AST, and suggest strategies to enhance customer satisfaction, reduce escalations, and improve overall operational efficiency.
# 
# **Deliverables**:
# 
# 1. Long average handle time (AHT) affects both efficiency and customer satisfaction. Explore the factors contributing to extended call durations, such as agent performance, call types, and sentiment. Identify key drivers of long AHT and AST, especially during high volume call periods. Additionally, could you quantify the percentage difference between the average handling time for the most frequent and least frequent call reasons?
# 
# 
# 
# 2. We often observe self-solvable issues unnecessarily escalating to agents, increasing their workload. Analyse the transcripts and call reasons to identify granular reasons associated to recurring problems that could be resolved via self-service options in the IVR system. Propose specific improvements to the IVR options to effectively reduce agent intervention in these cases, along with solid reasoning to support your recommendations.
# 
# 
# 
# 3. Understanding the primary reasons for incoming calls is vital for enhancing operational efficiency and improving customer service. Accurately categorizing call reasons enables the call center to streamline processes, reduce manual tagging efforts, and ensure that customers are directed to the appropriate resources. In this context, analyze the dataset to uncover patterns that can assist in understanding and identifying these primary call reasons. Please outline your approach, detailing the data analysis techniques and feature identification methods you plan to use. **Optional task, you may utilize the `test.csv` file to generate and submit your predictions**.

# 
# 
# 
# ### STEPS:
# 
# #### 1. Data Understanding
# #### 2. Data Handling
# #### 3. Data Analyzing
# #### 4. Data Visualisation
# #### 5. Uncover Insights
# 
# 
# 
#                       
# 

# ##  Importing Required Libraries

# In[136]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[137]:


# to supress warnings

import warnings 
warnings.filterwarnings('ignore')


# ##  Load Datasets

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


# ## Combining Data

# In[8]:


df_combined1 = pd.merge(df_sentiment, df_reasons, on = 'call_id', how = 'outer')
df_combined1


# In[9]:


df_combined2 = pd.merge(df_combined1, df_calls, on = 'call_id', how = 'outer')
df_combined2


# In[10]:


df = pd.merge(df_combined2, df_customers, on = 'customer_id', how = 'left')
df


# ## Defining Combined Data

# ### 1. Display Top 5 Rows of the Dataset

# In[12]:


df.head(5)


# ### 2. Checking Last 3 Rows of the Dataset

# In[13]:


df.tail(3)


# ### 3. Finding Shape of the Dataset i.e, Number of Rows & Number of Columns¶

# In[14]:


df.shape


# In[15]:


print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ### 4. Get Informaation about the Dataset like Total Number of Rows, Total Number of Columns, Datatypes of each Column and Memory Requirement

# In[16]:


df.info()


# In[17]:


df.dtypes


# ## Dealing with Datatypes

# In[18]:


df.columns


# In[19]:


df.head(5)


# In[ ]:





# In[20]:


# df['call_start_datetime'] = df['call_start_datetime'].str.replace('-', '/')


# In[21]:


df['call_start_datetime'] = pd.to_datetime(df['call_start_datetime'], format = '%m/%d/%Y %H:%M')


# In[22]:


df['call_start_date'] = df['call_start_datetime'].dt.date
df['call_start_time'] = df['call_start_datetime'].dt.time


# In[23]:


df.head()


# In[24]:


df.columns


# In[ ]:





# In[25]:


# df['agent_assigned_datetime'] = df['agent_assigned_datetime'].str.replace('-', '/')


# In[26]:


df['agent_assigned_datetime'] = pd.to_datetime(df['agent_assigned_datetime'], format = '%m/%d/%Y %H:%M')


# In[27]:


df['agent_assigned_date'] = df['agent_assigned_datetime'].dt.date
df['agent_assigned_time'] = df['agent_assigned_datetime'].dt.time


# In[28]:


df.head()


# In[ ]:





# In[29]:


# df['call_end_datetime'] = df['call_end_datetime'].str.replace('-', '/')


# In[30]:


df['call_end_datetime'] = pd.to_datetime(df['call_end_datetime'], format = '%m/%d/%Y %H:%M')


# In[31]:


df['call_end_date'] = df['call_end_datetime'].dt.date
df['call_end_time'] = df['call_end_datetime'].dt.time


# In[32]:


df.head()


# In[ ]:





# ## Taking Care of Null Values

# In[33]:


df.dtypes


# In[34]:


df.isnull().all()


# In[35]:


df.isnull().any()


# In[36]:


df.isnull().sum()


# In[37]:


df['agent_tone'].fillna('Unknown', inplace = True)


# In[38]:


df['primary_call_reason'].fillna('Unknown', inplace = True)


# In[39]:


df['average_sentiment'].describe()


# In[40]:


df['average_sentiment'].fillna(df['average_sentiment'].median(), inplace = True)


# In[41]:


df['elite_level_code'].fillna(df['elite_level_code'].median(), inplace = True)


# In[42]:


df['elite_level_code'] = df['elite_level_code'].astype(int)


# In[43]:


df.head(10)


# In[45]:


df.isnull().sum()


# ## Data Cleaning

# In[46]:


df['primary_call_reason'].unique()


# In[47]:


#removing extra spaces

df['primary_call_reason'] = df['primary_call_reason'].str.strip()


# In[165]:


#mapping similar values

cleaning_map = {
    'Voluntary  Cancel': 'Voluntary Cancel',
    'Voluntary   Cancel': 'Voluntary Cancel',
    'Voluntary Change': 'Voluntary Change',
    'Voluntary   Change': 'Voluntary Change',
    'Voluntary  Change': 'Voluntary Change',
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


# In[166]:


df['primary_call_reason'] = df['primary_call_reason'].replace(cleaning_map)


# In[167]:


df['primary_call_reason'].unique()


# ## Overall Statistics about Dataframe

# In[51]:


df.describe(include='object')


# In[52]:


for col in df.describe(include='object').columns:
    print(col)
    print(df[col].unique())
    print('-'*50)


# In[53]:


#summary statistics

df.describe(include = 'all')


# In[54]:


df.describe()


# In[ ]:





# # **DELIVERABLE - 1**

# 1. Long average handle time (AHT) affects both efficiency and customer satisfaction. Explore the factors contributing to extended call durations, such as agent performance, call types, and sentiment. Identify key drivers of long AHT and AST, especially during high volume call periods. Additionally, could you quantify the percentage difference between the average handling time for the most frequent and least frequent call reasons?
# 

# #### NOTE:
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

# In[55]:


df.columns


# **Factors contributing to extended call durations**:
#     
#     1. Agent Performance
#     2. Agent Tone and Agent Experience
#     3. Sentiment
#     4. Primary Call Reason
#     5. Call Timing and Shift Timing? Is it peak hour or non peak hour?
#     6. Customer Tone
#     7. Customer Loyalty Status
#     8. Silence Percentage

# In[56]:


#AHT

df['handle_time'] = (df['call_end_datetime'] - df['agent_assigned_datetime']).dt.total_seconds()


# In[58]:


#AST

df['waiting_time'] = (df['agent_assigned_datetime'] - df['call_start_datetime']).dt.total_seconds()


# In[59]:


# factor_columns = ['agent_id_x', 'agent_tone', 'average_sentiment', 'primary_call_reason', 'customer_tone', 'elite_level_code',  'silence_percent_average']


# In[60]:


# for factor in factor_columns:
#    print(df.groupby(factor)[['handle_time', 'waiting_time']].mean())


# In[61]:


# factor_columns


# ## **FACTOR : 1**

# ### **Agent Performance**

# In[65]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean()
agent_performance


# In[63]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean().sort_values(by = ['handle_time'])
agent_performance


# ### **Analysis**: 
# 
# Handle time of Agent 547592 is minimum while Handle time of Agent 102574 is maximum

# In[66]:


agent_performance = df.groupby('agent_id_x')[['handle_time', 'waiting_time']].mean().sort_values(by = ['waiting_time'])
agent_performance


# ### **Analysis**: 
# 
# Waiting time of Agent 616988 is minimum while Waiting time of Agent 547592 is maximum

# ### **Insights**: 
# 
# 1. It's possible that Agent 547592 is very efficient (hence the low handle time) and popular or in demand (which explains the high waiting time). This could mean that more customers prefer waiting for this agent because they trust their expertise, even if it means a longer wait.
# 
# 
# 2. Agent 547592 might also be more experienced or better trained, allowing them to handle issues faster, despite being in high demand.
# 
# 
# 3. Agent 102574, on the other hand, may be taking more time to handle customer queries, possibly due to inexperience or dealing with more complex issues.
# 
# 
# **Here, we can say that high demand can lead to longer waiting times, but efficiency can still result in shorter handle times**

# ## **FACTOR : 2**

# ### **Agent Experience**

# In[74]:


#Agent Experience - Number of calls handled by each agent

df['agent_experience'] = df.groupby('agent_id_x')['call_id'].transform('count')


# In[75]:


# Group by agent experience levels and calculate AHT and AST

experience_groups = pd.qcut(df['agent_experience'], q = 4, labels = ['Low', 'Medium', 'High', 'Very High'])


# In[76]:


df['experience_level'] = experience_groups


# In[77]:


experience_performance = df.groupby('experience_level')[['handle_time', 'waiting_time']].mean()
experience_performance


# ### **Analysis**: 
# 
# 1. Agents with Low experience have a handle time of 722 seconds and a waiting time of 436 seconds.
# 
# 
# 2. Agents with Very High experience have a handle time of 675 seconds and a waiting time of 437 seconds.

# ### **Insights**: 
# 
# 1. As the experience level increases, the handle time generally decreases. This suggests that more experienced agents are better at resolving issues faster, possibly because they’ve dealt with similar situations more frequently.
# 
# 
# 2. The waiting time remains fairly constant across all experience levels, which could indicate that waiting times are more dependent on external factors (e.g., customer demand or availability) than on agent experience.
# 
# **Here, We can focus on improving the skills of less experienced agents to match the performance of high-experience agents**.

# In[78]:


# Plotting a bar chart to compare handle time and waiting time by experience level

experience_levels = experience_performance.index
handle_time = experience_performance['handle_time']
waiting_time = experience_performance['waiting_time']

plt.figure(figsize = (10, 6))
plt.bar(experience_levels, handle_time, width = 0.4, label = 'Handle Time', color = 'skyblue', align = 'center')
plt.bar(experience_levels, waiting_time, width = 0.4, label = 'Waiting Time', color = 'orange', align = 'edge')

plt.title('Handle Time vs Waiting Time by Agent Experience Level', fontsize=18)
plt.xlabel('Experience Level', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.legend(fontsize = 12)

plt.show()


# In[79]:


df.groupby('experience_level').count()


# Analysis: Can provide more trainings to agents who have medium and low experince level

# ### **Agent Tone**

# In[80]:


agent_tone_performance = df.groupby('agent_tone')[['handle_time', 'waiting_time']].mean()
agent_tone_performance


# ### **Analysis**:
# 
# 1. Agents with a polite tone have the lowest handle time (220 seconds), meaning they resolve issues faster.
# 
# 
# 2. Agents with a neutral tone have the highest handle time (750 seconds).
# 
# 
# 3. The waiting time is relatively higher for agents with an unknown tone (446 seconds) and polite tone (461 seconds), while it's lower for agents with an angry (425 seconds) or frustrated tone (424 seconds).

# ### **Insights**:
# 
# 1. Agents who are polite seem to resolve issues the fastest, likely because customers may feel more comfortable and satisfied, leading to quicker resolutions.
# 
# 
# 2. Interestingly, agents with a neutral tone have the longest handle times, suggesting that a neutral approach may not engage the customer as effectively, leading to more drawn-out conversations.
# 
# 
# 3. While angry and frustrated tones show slightly lower waiting times, it doesn't necessarily lead to faster issue resolution, as their handle times are higher compared to polite agents.
# 
# **Here, by providing training the agents to maintain a polite and friendly tone can help improve efficiency.**

# In[81]:


# Plotting Handle Time and Waiting Time for each agent tone

agent_tones = agent_tone_performance.index
handle_time = agent_tone_performance['handle_time']
waiting_time = agent_tone_performance['waiting_time']


plt.figure(figsize = (10, 6))
plt.bar(agent_tones, handle_time, width = 0.4, label = 'Handle Time', color = 'skyblue', align = 'center')
plt.bar(agent_tones, waiting_time, width = 0.4, label = 'Waiting Time', color = 'orange', align = 'edge')

plt.title('Handle Time vs Waiting Time by Agent Tone', fontsize = 18)
plt.xlabel('Agent Tone', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.xticks(rotation = 45)
plt.legend(fontsize = 12)

plt.show()


# ## **FACTOR : 3**

# ### **Sentiment**

# In[82]:


sentiment_performance = df.groupby('average_sentiment')[['handle_time', 'waiting_time']].mean()
sentiment_performance


# ### **Analysis**: 
# 
# 1. Handle time (blue dots) shows a wide range, especially when the sentiment is slightly negative (around -0.5). Handle times are scattered from very low to extremely high (~1750 seconds) in this sentiment range.
# 
# 
# 2. Waiting time (orange dots) remains fairly consistent across different sentiment levels, with only minor fluctuations. It stays mostly in the 200-600 seconds range, regardless of whether the sentiment is positive or negative.
# 

# ### **Insights**: 
# 
# 1. When the conversation is neutral or slightly negative, the handle time seems to vary the most. This could mean that when customers are somewhat dissatisfied or unclear, agents might take much longer to resolve the issue, possibly due to more complex discussions or indecisiveness from the customer.
# 
# 
# 2. Interestingly, extremely negative or extremely positive sentiment does not seem to result in longer handle times. This suggests that extreme emotions (either positive or negative) might lead to quicker resolutions, either because the issue is simple or the customer is more direct about their needs.
# 
# 
# 3. Waiting time stays relatively stable across all sentiment levels, indicating that customer sentiment doesn’t affect how long they wait to speak to an agent
# 
# **Here, the special attention might be needed for customers who display neutral or slightly negative sentiment to resolve their issues more efficiently**

# In[84]:


# Plotting Handle Time vs Sentiment and Waiting Time vs Sentiment

sentiment = sentiment_performance.index
handle_time = sentiment_performance['handle_time']
waiting_time = sentiment_performance['waiting_time']

plt.figure(figsize=(10, 6))
plt.scatter(sentiment, handle_time, color = 'blue', label = 'Handle Time', s = 100, alpha = 0.6)
plt.scatter(sentiment, waiting_time, color = 'orange', label = 'Waiting Time', s = 100, alpha = 0.6)

plt.title('Sentiment vs Handle Time and Waiting Time', fontsize = 18)
plt.xlabel('Average Sentiment', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.legend(title = 'Time Type', fontsize = 12)
plt.grid(True)


plt.show()


# ## **FACTOR : 4**

# ### **Primary Call Reason**

# In[85]:


primary_reason_performance = df.groupby('primary_call_reason')[['handle_time', 'waiting_time']].mean()
primary_reason_performance


# ### **Analysis**:
# 
# 1. Checkout has the longest handle time (1016.9 seconds) and high waiting time (724.7 seconds). This suggests that checkout issues are complex and may require more agent intervention.
# 
# 
# 2. Voluntary Cancel has a relatively high handle time (721.9 seconds) and a moderate waiting time (539.5 seconds), meaning it also involves complex interactions.
# 
# 
# 3. Communications, Schedule Change, and Booking have significantly lower waiting times (around 240 seconds) and moderate handle times (between 372 to 490 seconds).
# 
# 
# 4. Issues like Baggage have a shorter handle time (333.6 seconds) but higher waiting time (542.1 seconds), meaning customers wait longer for a relatively quick resolution.

# ### **Insights**:
# 
# 1. Checkout and Voluntary Cancel take the longest to resolve, indicating these areas might benefit from improved processes or tools to reduce agent workload and simplify customer interactions.
# 
# 
# 2. Baggage and Unaccompanied Minor show high waiting times, possibly indicating that these issues are lower in priority or that more resources should be allocated to reduce wait times.
# 
# 
# 3. Communications and Schedule Changes are generally quicker to resolve and have short waiting times, meaning these issues are likely more straightforward and may be ideal candidates for self-service options through IVR systems or online tools.
# 
# **Here, complex issues like Checkout and Voluntary Cancel may need more focus on process improvement, while quicker and simpler tasks like Schedule Changes could be handled more effectively through self-service to reduce agent intervention**

# In[86]:


# Creating a heatmap 

primary_reason_performance_reset = primary_reason_performance.reset_index()


plt.figure(figsize = (14, 8))
sns.heatmap(primary_reason_performance.pivot_table(values = ['handle_time', 'waiting_time'], index = primary_reason_performance.index), 
            annot = True, cmap = 'Blues', linewidths = 0.5, fmt = '.1f')

plt.title('Average Handle Time and Waiting Time by Primary Call Reason', fontsize=18)
plt.xlabel('Time Metrics', fontsize =  14)
plt.ylabel('Primary Call Reason', fontsize = 14)
plt.xticks(rotation = 45)
plt.tight_layout()

plt.show()


# ## **FACTOR : 5**

# ### **Call Timing and Shift Timing? Is it peak hour or non peak hour?**

# In[88]:


df['call_hour'] = df['call_start_datetime'].dt.hour


# In[89]:


peak_hour = df.groupby('call_start_time').count()
peak_hour


# In[90]:


# peak_hour.sort_values(by = 'call_id', ascending = False).head(10)


# In[91]:


timelist = peak_hour.sort_values(by='call_id', ascending=False).head(10).index.tolist()


# In[92]:


df['is_peaktime'] = df['call_start_time'].isin(timelist)


# In[93]:


df.columns


# In[94]:


timing_performance = df.groupby('is_peaktime')[['handle_time', 'waiting_time']].mean()
timing_performance


# ### **Analysis**:
# 
# 1. Handle Time slightly increases during peak hours (from 697.05 seconds to 697.16 seconds). This shows that calls take a bit longer to handle when the call center is busier.
# 
# 
# 2. Waiting Time is lower during peak hours (drops from 437.09 seconds to 435.55 seconds), meaning customers wait less to get connected to an agent during peak times.

# ### **Insights**:
# 
# 1. The increase in handle time during peak hours is very small, suggesting that the agents are able to manage calls efficiently, even when demand is higher.
# 
# 
# 2. Interestingly, the waiting time decreases during peak hours, which might indicate that more resources (agents) are allocated during peak times, allowing for quicker customer service, or the system prioritizes faster connections during busy periods.
# 
# **Here, peak hours don’t significantly impact handle time, but they do reduce waiting time slightly, which is a positive indicator for resource management during busy periods**

# In[96]:


# Creating a figure with two subplots for handle time and waiting time

peak_status = ['Non-Peak', 'Peak']
handle_time = timing_performance['handle_time']
waiting_time = timing_performance['waiting_time']


plt.figure(figsize = (12, 6))

# Handle Time vs Peak Hour
plt.subplot(1, 2, 1)
plt.plot(peak_status, handle_time, marker = 'o', color = 'blue', label = 'Handle Time', linewidth = 2)
plt.title('Peak Hour vs Handle Time', fontsize = 16)
plt.xlabel('Peak Hour Status', fontsize = 12)
plt.ylabel('Handle Time (in seconds)', fontsize = 12)
plt.grid(True)

# Waiting Time vs Peak Hour
plt.subplot(1, 2, 2)
plt.plot(peak_status, waiting_time, marker = 'o', color = 'orange', label = 'Waiting Time', linewidth = 2)
plt.title('Peak Hour vs Waiting Time', fontsize = 16)
plt.xlabel('Peak Hour Status', fontsize = 12)
plt.ylabel('Waiting Time (in seconds)', fontsize = 12)
plt.grid(True)


plt.tight_layout()
plt.show()


# ## **FACTOR : 6**

# ### **Customer Tone**

# In[97]:


customer_tone_performance = df.groupby('customer_tone')[['handle_time', 'waiting_time']].mean()
customer_tone_performance


# ### **Analysis**:
# 
# 1. Handle time is highest when the customer has a neutral tone (707.6 seconds) and lowest when the customer is polite (689.7 seconds).
# 
# 
# 2. Waiting time remains fairly consistent across all tones, ranging from 436 to 438 seconds.

# ### **Insights**:
# 
# 1. Customers with a neutral tone seem to have the longest handle times. This could indicate that neutral-toned customers are more reserved or unclear about their issues, which requires more back-and-forth to resolve their problems.
# 
# 
# 2. Interestingly, customers who are polite have the shortest handle times, suggesting that polite customers might communicate their needs more clearly, leading to faster resolutions.
# 
# 
# 3. Waiting time is consistent across different customer tones, indicating that the customer's emotional state doesn’t impact how long they have to wait to speak to an agent.
# 
# **Here, we can say that clear and positive communication from the customer side might help resolve issues faster**

# In[102]:


#Creating a plot for Handle Time and Waiting Time by Customer Tone

customer_tones = customer_tone_performance.index
handle_time = customer_tone_performance['handle_time']
waiting_time = customer_tone_performance['waiting_time']


plt.figure(figsize=(10, 6))

# Handle Time
plt.plot(customer_tones, handle_time, marker = 'o', label = 'Handle Time', color = 'blue', linewidth = 2)

# Waiting Time
plt.plot(customer_tones, waiting_time, marker = 'x', label = 'Waiting Time', color = 'orange', linewidth = 2)

plt.title('Handle Time and Waiting Time by Customer Tone', fontsize=18)
plt.xlabel('Customer Tone', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.xticks(rotation = 45)
plt.grid(True)
plt.legend(title = 'Time Type', fontsize = 12)


plt.tight_layout()
plt.show()


# ## **FACTOR : 7**

# ### **Customer Loyality Status**

# In[100]:


df.columns


# In[101]:


loyalty_status_performance = df.groupby('elite_level_code')[['handle_time', 'waiting_time']].mean()
loyalty_status_performance


# ### **Analysis**:
# 
# 1. Handle time increases as the elite level increases. Customers with elite level 5 have the longest handle time (896.9 seconds), while customers with elite level 0 have a shorter handle time (696 seconds).
# 
# 
# 2. Waiting time decreases as the elite level increases. Elite level 5 customers wait the least (411.4 seconds), while elite level 0 customers wait the longest (438.6 seconds).

# ### **Insights**:
# 
# 1. Higher loyalty customers (elite levels 4 and 5) tend to have longer handle times. This could indicate that these customers are given more personalized attention, or their issues might be more complex due to the value they bring to the business.
# 
# 
# 2. However, these high-loyalty customers also experience shorter waiting times, suggesting that they are prioritized in the queue and attended to more quickly, likely due to their importance to the business.
# 
# 
# 3. Lower loyalty customers (elite level 0) have shorter handle times but longer waiting times, suggesting that they are not given the same priority as elite customers.

# In[104]:


# Creating a plot for Handle Time and Waiting Time by Loyalty Status (Elite Level)

elite_levels = loyalty_status_performance.index
handle_time = loyalty_status_performance['handle_time']
waiting_time = loyalty_status_performance['waiting_time']


plt.figure(figsize=(10, 6))

# Handle Time
plt.plot(elite_levels, handle_time, marker = 'o', label = 'Handle Time', color = 'blue', linewidth = 2)

# Waiting Time
plt.plot(elite_levels, waiting_time, marker = 'x', label = 'Waiting Time', color = 'orange', linewidth = 2)

plt.title('Handle Time and Waiting Time by Loyalty Status (Elite Level)', fontsize = 18)
plt.xlabel('Elite Level Code (Loyalty Status)', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.grid(True)
plt.legend(title ='Time Type', fontsize = 12)


plt.tight_layout()
plt.show()


# ## **FACTOR : 8**

# ### **Silenece Percentage**

# In[105]:


silence_performance = df.groupby(pd.cut(df['silence_percent_average'], bins = 4))[['handle_time', 'waiting_time']].mean()
silence_performance


# ### **Analysis**:
# 
# 1. Handle time increases significantly as the silence percentage increases. For the lowest silence percentage range (0-24.5%), the handle time is 424.75 seconds, while for the highest range (73.5%-98%), it reaches a staggering 1493.25 seconds.
# 
# 
# 2. Waiting time, on the other hand, remains fairly consistent, ranging between 431 and 439 seconds, with the highest silence percentage range having a slightly lower waiting time.

# ### **Insights**:
# 
# 1. As the silence percentage increases, the handle time dramatically increases. This suggests that long periods of silence during a call may be linked to complex or unresolved issues where both the customer and agent might be uncertain about how to proceed, leading to longer conversations.
# 
# 
# 2. The waiting time does not seem to be significantly impacted by silence percentage, meaning that how long a customer waits before speaking to an agent doesn’t affect how much silence there is during the call.
# 
# **Here, reducing silence by providing more guidance or clarity during the call could help reduce the overall handling time**

# In[106]:


# Creating plot for Handle Time and Waiting Time by Silence Percentage in Call

silence_bins = silence_performance.index.astype(str)  # Convert the index to string for better labeling
handle_time = silence_performance['handle_time']
waiting_time = silence_performance['waiting_time']

plt.figure(figsize=(10, 6))

# Handle Time
plt.plot(silence_bins, handle_time, marker = 'o', label = 'Handle Time', color = 'blue', linewidth = 2)

# Waiting Time
plt.plot(silence_bins, waiting_time, marker = 'x', label = 'Waiting Time', color = 'orange', linewidth = 2)

plt.title('Handle Time and Waiting Time by Silence Percentage in Call', fontsize=18)
plt.xlabel('Silence Percentage Range', fontsize = 14)
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.grid(True)
plt.legend(title = 'Time Type', fontsize = 12)

plt.tight_layout()
plt.show()


# In[164]:


# Calculate correlation matrix
filtered_df = df[df['is_peaktime']]
filtered_df = filtered_df.select_dtypes(include='number')

corr = filtered_df.corr()
plt.figure(figsize=(20, 10))  # Set the size of the figure
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink': .8})
plt.title('Heatmap of Correlation Matrix')
plt.show()


# ### **Key drivers of long AHT and AST (During High Call Volume Periods)**

# **Finally we can say that, based on the above analysis, the following factors contribute to longer AHT and AST, particularly during high-volume call periods:**
# 
# 1. Customer Tone
# 2. Agent Experience
# 3. Primary Call for Reason
# 4. Peak Hours
# 5. Silence Percentage
# 6. Customer Loyalty Status

# **Common Insights and Uncovering New Findings:**
# 
# 1. Long handle times are driven by unclear communication (neutral tone), complex call reasons (checkout), and high silence percentages.
# 
# 
# 2. Peak hour efficiency: While waiting times decrease, handle times increase slightly due to higher pressure.
# 
# 
# 3. Improving clarity through agent training, self-service options, and structured call flows can reduce handle times.
# 
# 
# **Uncovering Insights:**
# 
# 1. Silence percentage is a critical factor for long handle times across all call types. Reducing silence through better agent guidance and conversation strategies is a key area for improvement.
# 
# 
# 2. Communication clarity is crucial. Enhancing customer-agent interactions during neutral tone and complex issues can significantly reduce AHT.

# ### **Quantifying the Percentage Difference Between AHT for Most and Least Frequent Call Reasons**

# In[109]:


# Calculating AHT and AST across the entire dataset

total_handle_time = df['handle_time'].sum()
total_calls = len(df)


# In[110]:


AHT = total_handle_time / total_calls
AST = df['waiting_time'].sum() / total_calls

print(f"Average Handle Time (AHT): {AHT} seconds")
print(f"Average Speed to Answer (AST): {AST} seconds")


# In[ ]:





# In[111]:


# Finding the most and least frequent call reasons

most_frequent_reason = df['primary_call_reason'].value_counts().idxmax()
least_frequent_reason = df['primary_call_reason'].value_counts().idxmin()


# In[112]:


# Calculating AHT for both

most_frequent_AHT = df[df['primary_call_reason'] == most_frequent_reason]['handle_time'].mean()
least_frequent_AHT = df[df['primary_call_reason'] == least_frequent_reason]['handle_time'].mean()


# In[113]:


# Calculating percentage difference

percentage_diff = ((most_frequent_AHT - least_frequent_AHT) / least_frequent_AHT) * 100

print(f"Percentage difference in AHT between most and least frequent call reasons: {percentage_diff:.2f}%")


# # **DELIVERABLE - 2**

# We often observe self-solvable issues unnecessarily escalating to agents, increasing their workload. Analyse the transcripts and call reasons to identify granular reasons associated to recurring problems that could be resolved via self-service options in the IVR system. Propose specific improvements to the IVR options to effectively reduce agent intervention in these cases, along with solid reasoning to support your recommendations.
# 

# In[115]:


# Group by primary call reason to identify recurring problems

call_reason_counts = df['primary_call_reason'].value_counts()
print(call_reason_counts)


# In[118]:


#Converting into % and creating a pie chart

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


# In[117]:


#Creating a visual for Customer Tone Distribution by Call Reason

plt.figure(figsize = (12, 7))
sns.countplot(data = df, x = 'customer_tone', hue = 'primary_call_reason', palette = "magma")
plt.xticks(fontsize = 12)
plt.xlabel("Customer Tone", fontsize = 14)
plt.ylabel("Count", fontsize = 14)
plt.title("Customer Tone Distribution by Call Reason", fontsize = 16, weight = 'bold', pad = 20)
plt.legend(title = "Primary Call Reason", bbox_to_anchor = (1, 1), loc = "upper left", fontsize =12)

plt.tight_layout(pad=2)
plt.show()


# In[119]:


plt.figure(figsize=(12, 7))
sns.countplot(data=df, x='elite_level_code', hue='primary_call_reason', palette="magma")
plt.xticks(fontsize=12)
plt.xlabel("Loyalty Status (Elite Level)", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Loyalty Status Impact on Call Reason", fontsize=16, weight='bold', pad=20)
plt.legend(title="Primary Call Reason", bbox_to_anchor=(1, 1), loc="upper left", fontsize=12)
plt.tight_layout(pad=2)
plt.show()


# In[131]:


primary_call_data = df.groupby('primary_call_reason').agg(
    avg_sentiment=('average_sentiment', 'mean'),
    avg_AHT=('handle_time', 'mean'),
    avg_AST=('waiting_time', 'mean')
).reset_index()
print(primary_call_data)


# In[134]:


#Creating a visual of Average Handle Time by Primary Call Reason

plt.figure(figsize=(12, 6))
sns.barplot(x='primary_call_reason', y='avg_AHT', data=primary_call_data)
plt.xticks(rotation=90)
plt.title('Average Handle Time by Primary Call Reason')
plt.show()


# In[135]:


# Creating a visual for Average Sentiment by Primary Call Reason

plt.figure(figsize=(12, 6))
sns.barplot(x='primary_call_reason', y='avg_sentiment', data=primary_call_data)
plt.xticks(rotation=90)
plt.title('Average Sentiment by Primary Call Reason')
plt.show()


# **After analysizing the above information, we are classifying the Call Reasons into Self Service/IVR and Agent Needed based on the following factors:**
# 
# 1. Average Sentiment: If sentiment is negative or low, customers may require more personalized agent support.
# 
# 
# 2. AHT (Average Handle Time): Shorter AHT indicates simpler issues that could be handled through self-service.
# 
# 
# 3. AST (Average Speed to Answer): If AST is high, the call reason might be suitable for IVR to reduce waiting time.
# 

# In[128]:


self_solvable = [ 'Baggage', 'Booking', 'Seating', 'Schedule Change', 'Voluntary Change', 'Other Topics', 
                 'Products and Services', 'Digital Support', 'Traveler Updates','Voluntary Cancel']

# These call reasons are frequent and straightforward, with lower sentiment and manageable AHT, making them ideal candidates for automation


# In[129]:


agent_needed = ['IRROPS', 'Mileage Plus', 'Checkout', 'Post Flight', 'Upgrade', 'Check In',
                'Communications', 'Disability', 'Unaccompanied Minor', 'ETC']

# These call reasons are more complex, involve sensitive customer interactions, or have higher AHT and negative sentiment, requiring human intervention.


# In[124]:


df['issue_type'] = df['primary_call_reason'].apply(
    lambda x: 'self-solvable' if x in self_solvable else 'agent-needed'
)


# In[125]:


df['issue_type'].value_counts()


# ### **Proposed Improvements to IVR Options to Reduce Agent Intervention**

# ### **1. Baggage:**
# 
# **Avg Sentiment:** -0.019 (neutral)  
# **AHT:** 333 seconds (low)  
# **AST:** 542 seconds (high)
# 
# **Improvement:** Introduce automated baggage tracking and status updates in the IVR.
# 
# **How:** Customers can input their baggage ID or flight number to get real-time updates on the location of their luggage.  
# **Reasoning:** Baggage status inquiries are repetitive and straightforward, ideal for automation. This will reduce call volume to agents and improve customer satisfaction by providing immediate answers.
# 
# 
# 
# 
# 
# ### **2. Booking:**
# 
# **Avg Sentiment:** -0.013 (neutral)  
# **AHT:** 427 seconds (moderate)  
# **AST:** 240 seconds (low)
# 
# **Improvement:** Implement booking modification options within the IVR.
# 
# **How:** Allow customers to modify or confirm their bookings by inputting their booking reference numbers, without agent intervention.  
# **Reasoning:** Basic booking changes, such as date or seat selection, are simple and frequent. Automating these interactions will streamline operations and reduce agent intervention for repetitive tasks.
# 
# 
# 
# 
# 
# ### **3. Seating:**
# 
# **Avg Sentiment:** -0.003 (neutral)  
# **AHT:** 475 seconds (moderate)  
# **AST:** 571 seconds (high)
# 
# **Improvement:** Automate seat selection and changes through IVR.
# 
# **How:** Provide an option for customers to select or change seats via IVR by entering their booking reference. The system can then offer available seats and confirm changes.  
# **Reasoning:** This is a frequent request that can be easily handled through automation. Automating seating arrangements reduces the need for agents to manage simple seat-related inquiries.
# 
# 
# 
# 
# 
# ### **4. Schedule Change:**
# 
# **Avg Sentiment:** -0.034 (slightly negative)  
# **AHT:** 490 seconds (moderate)  
# **AST:** 241 seconds (low)
# 
# **Improvement:** Provide automated flight schedule changes.
# 
# **How:** Allow customers to check for changes in their flight schedule and make adjustments directly through IVR, without speaking to an agent.  
# **Reasoning:** Schedule changes are frequent and often don’t require agent intervention. Automating this process allows customers to manage their schedules quickly and conveniently.
# 
# 
# 
# 
# 
# ### **5. Voluntary Change:**
# 
# **Avg Sentiment:** -0.034 (slightly negative)  
# **AHT:** 490 seconds (moderate)  
# **AST:** 241 seconds (low)
# 
# **Improvement:** Automate voluntary flight changes via IVR.
# 
# **How:** Allow customers to modify their flight details (e.g., changing flight times or dates) via IVR by selecting from available options.  
# **Reasoning:** Voluntary changes are often simple, especially when customers just want to change their booking details. Automating these changes will reduce agent load and provide faster service to customers.
# 
# 
# 
# 
# 
# ### **6. Other Topics:**
# 
# **Avg Sentiment:** -0.004 (neutral)  
# **AHT:** 350 seconds (low)  
# **AST:** 239 seconds (low)
# 
# 
# **Improvement:** Provide a self-service portal for miscellaneous requests via IVR.
# 
# 
# **How:** Offer options for common but miscellaneous issues (such as inquiring about airport services) in the IVR, directing customers to the appropriate resources or FAQs.  
# **Reasoning:** Most of these miscellaneous issues are easily resolvable with self-service or information portals, reducing the need for agents to handle these calls.
# 
# 
# 
# 
# 
# ### **7. Products and Services:**
# 
# **Avg Sentiment:** -0.034 (slightly negative)  
# **AHT:** 747 seconds (moderate)  
# **AST:** 302 seconds (moderate)
# 
# **Improvement:** Automate inquiries about products and services in the IVR.
# 
# **How:** Provide detailed automated information about available products and services (e.g., loyalty programs, onboard services) through IVR.  
# **Reasoning:** Inquiries about products and services are simple and can be automated. Automating responses will save agents time while still providing customers with the necessary information.
# 
# 
# 
# 
# 
# ### **8. Digital Support:**
# 
# **Avg Sentiment:** -0.042 (slightly negative)  
# **AHT:** 372 seconds (moderate)  
# **AST:** 506 seconds (high)
# 
# **Improvement:** Offer automated troubleshooting and FAQ support via IVR.
# 
# **How:** Create a digital support section in the IVR system where customers can access troubleshooting steps for common issues.  
# **Reasoning:** Many digital support inquiries are repetitive and can be resolved through automated processes or FAQ-based self-service, reducing the need for human intervention.
# 
# 
# 
# 
# 
# ### **9. Traveler Updates:**
# 
# **Avg Sentiment:** 0.007 (neutral)  
# **AHT:** 393 seconds (low)  
# **AST:** 690 seconds (high)
# 
# **Improvement:** Automate real-time travel updates in IVR.
# 
# **How:** Provide an option for real-time flight status, delay information, and travel alerts through IVR, based on flight number or booking reference.  
# **Reasoning:** Traveler updates are a frequent and simple request that can be fully automated. Providing real-time updates will reduce the number of calls directed to agents and improve the customer experience.
# 
# 
# 
# 
# 
# ### **10. Voluntary Cancel:**
# 
# **Avg Sentiment:** -0.031 (slightly negative)  
# **AHT:** 722 seconds (moderate)  
# **AST:** 540 seconds (moderate)
# 
# **Improvement:** Allow customers to cancel flights via IVR.
# 
# **How:** Implement a flight cancellation option where customers can cancel their booking directly through the IVR.  
# **Reasoning:** Simple cancellations don’t require human intervention. Automating this will save time for both customers and agents, especially when customers know they want to cancel without requiring further information.

# In[ ]:





# # **DELIVERABLE - 3**

# Understanding the primary reasons for incoming calls is vital for enhancing operational efficiency and improving customer service. Accurately categorizing call reasons enables the call center to streamline processes, reduce manual tagging efforts, and ensure that customers are directed to the appropriate resources. In this context, analyze the dataset to uncover patterns that can assist in understanding and identifying these primary call reasons. Please outline your approach, detailing the data analysis techniques and feature identification methods you plan to use. Optional task, you may utilize the `test.csv` file to generate and submit your predictions.

# In[138]:


from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[139]:


# Converting reasons to number to make it easier for creating correlation metric
reason_to_num = {
    'Seating': 1,
    'IRROPS': 2,
    'Voluntary Change': 3,
    'Mileage Plus': 4,
    'Unknown': 5,
    'Communications': 6,
    'Voluntary Cancel': 7,
    'ETC': 8,
    'Upgrade': 9,
    'Check In': 10,
    'Schedule Change': 11,
    'Products and Services': 12,
    'Booking': 13,
    'Other Topics': 14,
    'Post Flight': 15,
    'Digital Support': 16,
    'Baggage': 17,
    'Disability': 18,
    'Voluntary  Change': 3,
    'Checkout': 20,
    'Traveler Updates': 21,
    'Unaccompanied Minor': 22
}

# Creating map to convert customer tonr and agent tone into numbers

tone_mapping = {
    'polite': 0,
    'angry': 1,
    'frustrated': 2,
    'neutral': 3,
    'calm': 4,
    
}

# Applying the mapping to the 'customer_tone' column

df['customer_tone_numeric'] = df['customer_tone']
df['agent_tone_numeric'] = df['agent_tone'].map(tone_mapping)
df['primary_call_reason_num'] = df['primary_call_reason'].map(reason_to_num)
df['primary_call_reason_num'].unique()
df['agent_tone_numeric'].fillna(df['agent_tone_numeric'].median(), inplace = True)
df['agent_tone_numeric'].unique()


# In[142]:


# Creating a Correlation Heatmap for understanding the relation between different features

plt.figure(figsize = (10, 8))
correlation_matrix = df[['call_id', 'customer_tone_numeric', 'agent_tone_numeric', 'average_sentiment', 'silence_percent_average', 
                      'elite_level_code', 'handle_time', 'waiting_time', 'agent_experience', 'primary_call_reason_num' ]].corr()
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation Heatmap')

plt.show()


# In[143]:


sns.pairplot(df[['call_id', 'customer_tone_numeric', 'agent_tone_numeric', 'average_sentiment', 'silence_percent_average', 
                 'elite_level_code', 'handle_time', 'waiting_time', 'agent_experience', 'primary_call_reason_num' ]])

plt.title('Pair Plot of Features')

plt.show()


# ### **Analysis**:
# 
# Primary call reasons and waiting time are related. 
# 
# **On the basis of this observation, we decided to use KMeans Model to categorize call reasons on the basis of AST(waiting time)**

# In[145]:


one_hot_encoder = OneHotEncoder(sparse = False)

# Apply the OneHotEncoder on the desired columns

encoded_call_reason = one_hot_encoder.fit_transform(df[['primary_call_reason']])


# In[147]:


X = pd.concat([
    pd.DataFrame(encoded_call_reason),
    df[[ 'waiting_time']]], axis = 1)


# In[148]:


X.columns = X.columns.astype(str)

# Applying the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[150]:


#Note: used pca to reduce dimensions

pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)


# In[152]:


inertia = []
K_values = range(1, 40)  # Trying K from 1 to 40

for k in K_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the inertia values
plt.plot(K_values, inertia, 'bx-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal K')

plt.show()


# ### **Analysis**:
# 
# After using Elbow Method, we came to a conclusion that we should use k = 23
# 
# Reason: As there is a massive drop in inertia at k = 23

# In[153]:


kmeans = KMeans(n_clusters = 23, random_state = 42)
df['cluster'] = kmeans.fit_predict(X_scaled)


# In[156]:


cluster_composition = df.groupby('cluster')['primary_call_reason'].unique()
print(cluster_composition)


# In[157]:


for cluster, reasons in cluster_composition.items():
    print(f"Cluster {cluster}:")
    
    for reason in reasons:
        print(f"  - {reason}")


# In[159]:


#Creating PCA Scatter plot

plt.figure(figsize = (5, 3))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c = df['cluster'], cmap = 'viridis', s = 50)
plt.title('Cluster Visualization for Call Reasons')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label = 'Cluster')

plt.show()


# In[160]:


sil_score = silhouette_score(X_scaled, df['cluster'])
print(f'Silhouette Score: {sil_score}')


# ### **Analysis**:
# 
# In this case, the silhouette score is 0.87, which is very high. This indicates that clustering algorithm has performed well, as the data points are well-separated and assigned to the correct clusters.

# In[161]:


db_index = davies_bouldin_score(X_scaled, df['cluster'])
print(f'Davies-Bouldin Index: {db_index}')


# ### **Analysis**:
# 
# In this case, the DB Index is 0.167, which is very low, suggesting that the clusters are very distinct and compact, which is a sign of good clustering performance.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




