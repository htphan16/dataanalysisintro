import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv('titanic_data.csv')
grouped_data1 = titanic_df.groupby(['Pclass'])
grouped_data2 = titanic_df.groupby(['Sex'])
grouped_data3 = titanic_df.groupby(pd.cut(titanic_df['Age'], np.arange(0, 90, 10)))
grouped_data4 = titanic_df.groupby(pd.cut(titanic_df['Fare'], np.arange(0, 1000, 200)))
grouped_data5 = titanic_df.groupby('Embarked')
grouped_data6 = titanic_df.groupby('SibSp')
grouped_data7 = titanic_df.groupby('Parch')
grouped_data8 = titanic_df.groupby([pd.cut(titanic_df['Age'], np.arange(0, 90, 10)),'Sex'])

data1 = grouped_data1.mean()['Survived']
data2 = grouped_data2.mean()['Survived']
data3 = grouped_data3.mean()['Survived']
data4 = grouped_data4.mean()['Survived'].dropna()
data5 = grouped_data5.mean()['Survived']
data6 = grouped_data6.mean()['Survived']
data7 = grouped_data7.mean()['Survived']
data8 = grouped_data8.mean()['Survived'].fillna(0)

#Bar charts
#data1.plot.bar()
#data2.plot.bar()
#data3.plot.bar()
#data4.plot.bar()
#data5.plot.bar()
#data6.plot.bar()
#data7.plot.bar()
#plt.show()





def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.
    
    correlation = average of (x in standard units) times (y in standard units)
    
    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''
    std_x = (x - x.mean()) / x.std(ddof = 0)
    std_y = (y - y.mean()) / y.std(ddof = 0)
    return (std_x*std_y).mean()

'''print(correlation(titanic_df['Survived'], titanic_df['Fare']))
print(correlation(titanic_df['Survived'], titanic_df['Age']))
print(correlation(titanic_df['Survived'], titanic_df['Pclass']))
print(correlation(titanic_df['Survived'], titanic_df['SibSp']))
print(correlation(titanic_df['Survived'], titanic_df['Parch']))'''


#Age: the younger children are more likely to survive than grown adults.
#Fare: The higher the fare one paid, the more likely one survived.
#Pclass: The higher the class one was in, the more likely one survived.
#Sex: Females are more likely to survive than males.
#Embarked: those who embarked at Cherbourg are more likely to survive than those who embarked at Queenstown or Southampton
#No significant correlation between number of siblings/spouses and number pf parents/children on whether one is more likely to survive
# Strongest correlation of survival rate as a function of class of passengers