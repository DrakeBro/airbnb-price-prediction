#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




#Visual setting
pd.set_option('display.width', None)

#reading file

df = pd.read_csv('dataSP23.csv')

#clearing unnecessary columns
df = df.drop([
    'id', 'name', 'host_id', 'host_name',
    'last_review', 'number_of_reviews',
    'last_review', 'reviews_per_month', 'neighbourhood'
], axis=1)

# Check for missing values
df.isnull().sum()

# Check for duplicates
df.duplicated().sum()


#detect any outliers in the price column using the IQR method
def check_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

# Encode categorical variables
le = LabelEncoder()

df['neighbourhood_group'] = le.fit_transform(df['neighbourhood_group'])
df['room_type']           = le.fit_transform(df['room_type'])

print(df.head())

#train and test split

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_safe, y_train, y_safe = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Add scale

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

#training the model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#evaluate the linear regression model
y_pred = regressor.predict(X_test)

print('R2 score: ', r2_score(y_test, y_pred))
print('Accuracy of linear regression classifier on test set: {:.2f}'.format(
    100 * (regressor.score(X_test, y_test))
))

git 
# Scatterplots: features vs. price
fig, axs = plt.subplots(1, 3, sharey=True)
df.plot(kind='scatter', x='neighbourhood_group', y='price', ax=axs[0], figsize=(16, 8))
df.plot(kind='scatter', x='room_type',           y='price', ax=axs[1])
df.plot(kind='scatter', x='minimum_nights',      y='price', ax=axs[2])

# Distribution of price
plt.figure(figsize=(10, 6))
plt.hist(df['price'], bins=30)
plt.xlabel('price')
plt.ylabel('count')
plt.show()

# Summary statistics
print(df.describe())