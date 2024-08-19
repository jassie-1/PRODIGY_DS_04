import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

twitts_train = pd.read_csv('C:\\Users\\DELL\\Desktop\\Dataset\\twitter_training.csv')
twitts_valid = pd.read_csv('C:\\Users\\DELL\\Desktop\\Dataset\\twitter_validation.csv')
# Naming each column
column_name=['TweetID','Entity','Sentiment','Tweet_Content']
twitts_train.columns=column_name
twitts_valid.columns=column_name
# Combining 2 dataframes to 1 dataframe
twitts=pd.concat([twitts_train,twitts_valid],ignore_index=False)
twitts.head()
twitts.columns.tolist()
twitts.info()
# Data Cleaning
twitts.isnull().sum()
twitts.duplicated().sum()
twitts.dropna(inplace=True)
twitts.drop_duplicates(inplace=True)
print(twitts.isnull().sum())
print("Duplicate Values:",twitts.duplicated().sum())
# Dropping Irrelevant columns:
twitts.drop(columns=['TweetID','Tweet_Content'],inplace=True)
twitts.head()
twitts.info()
# Data Visualization
entity_content=twitts['Entity'].value_counts()
entity_content.plot(kind='pie', autopct='%1.1f%%', figsize=(10, 12))
plt.title('Distribution of Entities')
plt.show()
#Sentiment Analysis
sentiment_content=twitts['Sentiment'].value_counts()
color= plt.get_cmap('viridis')
colors = [color(i) for i in np.linspace(0, 1, len(sentiment_content))]
sentiment_content.plot(kind='bar',color=colors,grid=True)
reactions_entities = pd.crosstab(twitts['Entity'],twitts['Sentiment'])
reactions_entities.plot(kind='bar', figsize=(16, 6),grid=True)
plt.show()
