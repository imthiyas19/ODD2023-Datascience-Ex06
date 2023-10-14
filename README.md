# ODD2023-Datascience-Ex06
## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.
## Program:
```
Developed By: M.Mohammed imthiyas
Register No: 212222230083
Importing libraries and reading csv file:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
Before Transformation:
```
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/e842df9e-36af-4ba5-adb6-58e6213ae170)
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/7c8a9e9e-c950-4bd2-83f0-80ca3b06cff2)
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/533aa5bf-0173-4dcc-aba1-3efccea8556c)
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/717025d8-5bdc-4434-8257-7b3a04d10ae1)




   
Log Transformation:
```

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()


df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
 ![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/09e329f9-5442-4b82-b723-2a9b6298bb86)
 ![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/0a92f0c7-41cd-4bcd-9fd8-355b80955800)


Reciprocal Transformation:
```
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/74c0886e-7ba5-406e-8deb-4fdc1fdfa859)


SquareRoot Transformation:
```
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/b0fda908-674e-4a12-97f9-a45e4b572fd3)


Power Transformation:
```
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```

 ![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/317758f9-1ce5-4be6-83c8-7f4a8c3421ec)

Quantile Transformation:
```

qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()
```
![image](https://github.com/imthiyas19/ODD2023-Datascience-Ex06/assets/120353416/c7fd00f5-ca24-45e6-8c31-9c6ee9cbba0e)

Result:
Thus feature transformation is done for the given dataset.
