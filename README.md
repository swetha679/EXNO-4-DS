# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
REG NO:212224040345
NAME :B R SWETHA NIVASINI 
```

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

```

![Screenshot 2025-04-19 072844](https://github.com/user-attachments/assets/9407acef-c4e4-42de-aef3-b0d1b6d50435)


```
df.dropna()
```

![Screenshot 2025-04-19 072929](https://github.com/user-attachments/assets/fb40a273-bbf5-4418-a908-d0136ddb5758)

```
max_vals=np.max(np.abs(df[['Height','Weight']]),axis=0)
max_vals
```

![Screenshot 2025-04-19 073003](https://github.com/user-attachments/assets/5d463e5c-194b-4b7c-a3db-4a7fe09388fd)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1=df.copy()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![Screenshot 2025-04-19 073038](https://github.com/user-attachments/assets/47bc674e-a267-4931-9d98-3e9babfd6f0a)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']] = scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![Screenshot 2025-04-19 073151](https://github.com/user-attachments/assets/ef596a12-2b34-45f3-a678-997e7b25512c)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2=df.copy()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```

![Screenshot 2025-04-19 073259](https://github.com/user-attachments/assets/8ece4115-92d7-44f5-a879-f1cbf0666147)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

![Screenshot 2025-04-19 073400](https://github.com/user-attachments/assets/a5b43c12-3ffe-4e52-9847-23857b636398)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![Screenshot 2025-04-19 073459](https://github.com/user-attachments/assets/cdc2f553-c72a-43e5-8207-5c3e90541fdb)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![Screenshot 2025-04-19 073821](https://github.com/user-attachments/assets/67b0db78-18f0-4334-8c16-a77f083d04c3)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif 
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]

}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']] 
y=df['Target']
selector = SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_feature=x.columns[selected_feature_indices]
print("Selected Feature:")
print(selected_feature)

```

![Screenshot 2025-04-19 073914](https://github.com/user-attachments/assets/43100a6c-c445-4ca8-b486-528c702d6498)



































































































































# RESULT:
Thus for the given dat Feature Scaling and Feature Selection process is sucsessfully performed.
