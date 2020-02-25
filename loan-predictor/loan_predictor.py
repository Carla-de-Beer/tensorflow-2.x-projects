# Carla de Beer
# Created: February 2020
# Keras API project to predict whether a customer is likely to repay a loan, or to default on it.
# Based on the Udemy course: Complete TensforFlow 2 and Keras Deep Learning Bootcamp:
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp
# The project uses a subset of the LendingClub DataSet obtained from Kaggle:
# https://www.kaggle.com/wordsforthewise/lending-club

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# #########################################################
# Data Loading
# #########################################################

data_info = pd.read_csv('data/lending_club_info.csv', index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])


def feature_info(col_name):
    print(data_info.loc[col_name]['Description'])


feature_info('mort_acc')

df = pd.read_csv('data/lending_club_loan_data.csv')
df.info()

# #########################################################
# Exploratory Data Analysis
# #########################################################

# Create a countplot with loan_status on the x-axis.
sns.countplot(x='loan_status', data=df)
plt.savefig('images/countplot loan_status')
plt.show()
# => Use precision and recall rather than accuracy because the dataset is imbalanced.

# Create a histogram of the loan_amnt column.
plt.figure(figsize=(12, 4))
sns.distplot(df['loan_amnt'], kde=False, bins=40)
plt.savefig('images/distplot loan_amnt')
plt.xlim(0, 45000)

# Examine the correlation between the continuous feature variables.
# Calculate the correlation between all continuous numeric variables.
df.corr()

# Visualize this using a heatmap.
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.savefig('images/heatmap correlation')
plt.show()

# There is an almost perfect correlation with the "installment" feature.
# Create a scatterplot relating installment and loan_amnt.
feature_info('installment')
feature_info('loan_amnt')
sns.scatterplot(x='installment', y='loan_amnt', data=df)
plt.savefig('images/scatterplot installment')
plt.show()

# Create a boxplot showing the relationship between loan_status and loan_amnt.
sns.boxplot(df['loan_status'], df['loan_amnt'])
plt.savefig('images/boxplot loan_status loan_amnt')
plt.show()

# Calculate the summary statistics for the loan_amnt, grouped by the loan_status.
df.groupby('loan_status')['loan_amnt'].describe()

# Explore the grade and sub-grade columns that are attributed to the loans.
df['grade'].sort_values().unique().tolist()
df['sub_grade'].sort_values().unique().tolist()

# Create a countplot per grade.
sns.countplot(x='grade', data=df, hue='loan_status')
plt.savefig('images/countplot grade')
plt.show()

# Display a count plot per sub-grade.
# Create a similar plot, but set hue="loan_status"
plt.figure(figsize=(12, 4))
order = df['sub_grade'].sort_values().unique().tolist()
sns.countplot(x='sub_grade', data=df, order=order, palette='coolwarm')
plt.savefig('images/countplot sub_grade 1')
plt.show()

plt.figure(figsize=(12, 4))
sns.countplot(x='sub_grade', data=df, order=order, palette='coolwarm', hue='loan_status')
plt.savefig('images/countplot sub_grade 2')
plt.show()

# It looks like F and G sub-grades don't get paid back that often.
# Isolate those and recreate the countplot for those sub-grades.
plt.figure(figsize=(12, 4))
f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, hue='loan_status', order=subgrade_order)
plt.savefig('images/countplot sub_grade 3')
plt.show()


# Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and
# a 0 if it was "Charged Off".

def convert_status(status):
    if status == 'Fully Paid':
        return 1
    elif status == 'Charged Off':
        return 0


df['loan_repaid'] = df['loan_status'].apply(convert_status)
print(df[['loan_repaid', 'loan_status']])

# Create a bar plot showing the correlation of the numeric features to the new loan_repaid column.
# Drop loan_repaid because it is perfectly correlated with itself.
df.corr()['loan_repaid'].drop('loan_repaid').sort_values().plot(kind='bar')
plt.savefig('images/plot corr')
plt.show()

# #########################################################
# Data PreProcessing: Remove or fill any missing data.
# Remove unnecessary or repetitive features.
# Convert categorical string features to dummy variables.
# #########################################################

# #########################################################
# Missing Data
# Use a variety of factors to decide whether or not they would be useful,
# to see if we should keep, discard, or fill in the missing data.
# #########################################################

# What is the length of the dataframe?
print(df.shape[0])

# Create a Series that displays the total count of missing values per column.
df.isnull().sum()

# Convert this Series to be in term of percentage of the total DataFrame
df.isnull().sum().div(df.shape[0]).mul(100)

# Examine emp_title and emp_length to see whether it will be okay to drop them.
feature_info('emp_title')
feature_info('emp_length')

# How many unique employment job titles are there?
df['emp_title'].value_counts()
df['emp_title'].nunique()

# Realistically there are too many unique job titles to try to convert this to a dummy variable feature,
# so remove the emp_title column.
df = df.drop(['emp_title'], axis=1)
df['emp_length'].sort_values(ascending=False).unique().tolist()

# Create a count plot of the emp_length feature column.
emp_length_order = df['emp_length'].sort_values().unique().tolist()
del emp_length_order[-1]
print(emp_length_order)

emp_length_new_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
                        '8 years', '9 years', '10+ years',
                        ]

data = df.sort_values(by='emp_length', ascending=False)
plt.figure(figsize=(12, 4))
sns.countplot(x='emp_length', data=data, order=emp_length_new_order)
plt.savefig('images/countplot emp_length 1')
plt.show()

# Plot out the countplot with a hue separating Fully Paid vs Charged Off
plt.figure(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, order=emp_length_new_order, hue='loan_status')
plt.savefig('images/countplot emp_length 2')
plt.show()

# The percentage of charge offs per category is required,
# i.e. determine what percent of people per employment category didn't pay back their loan.
print(df['emp_length'])
print(df['loan_status'])

A = []
for i in emp_length_order:
    x = (df['emp_length'] == i) & (df['loan_status'] == 'Charged Off')
    A.append(x.sum())
print(A)

B = []
for i in emp_length_order:
    x = (df['emp_length'] == i) & (df['loan_status'] == 'Fully Paid')
    B.append(x.sum())
print(B)

df['emp_length'].value_counts()

div = [ai / bi for ai, bi in zip(A, B)]
print(div)

loan_default_df = pd.DataFrame(data=div, index=emp_length_order, columns=['emp_length'])
print(loan_default_df)
loan_default_df.plot(kind='bar', legend=False)
plt.savefig('images/bar plot')
plt.show()

# Charge off rates are extremely similar across all employment lengths, so drop the emp_length column.
df = df.drop(['emp_length'], axis=1)
df.isnull().sum()

# Review the title column vs the purpose column.
df['purpose'].head(10)
df['title'].head(10)

# The title column is simply a string subcategory/description of the purpose column, so drop the title column.
feature_info('purpose')
feature_info('title')
df = df.drop(['title'], axis=1)

# Find out what the mort_acc feature represents
feature_info('mort_acc')

# Create a value_counts of the mort_acc column.
df['mort_acc'].value_counts()

# There are many ways we could deal with this missing data.
# We could attempt to build a simple model to fill it in,
# based on the other columns to see which most highly correlates to mort_acc
df.corr()['mort_acc'].sort_values(0)

# The total_acc feature correlates with the mort_acc.
# Group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry.
total_acc_mean = df.groupby('total_acc').mean()['mort_acc']
print(total_acc_mean)


# Fill in the missing mort_acc values based on their total_acc value.
# If the mort_acc is missing, fill in that missing value with the mean value corresponding to
# its total_acc value from the Series we created above.


def fill_missing_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_mean[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_missing_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

df.isnull().sum()

# revol_util and the pub_rec_bankruptcies have missing data points,
# but they account for less than 0.5% of the total data.
# Remove the rows that are missing those values in those columns.
df = df.dropna()
df.isnull().sum()

# #########################################################
# Categorical Variables and Dummy Variables
# #########################################################

# List all the columns that are currently non-numeric.
print(df.select_dtypes(['object']).columns)

# term feature
# Convert the term feature into either a 36 or 60 integer numeric data type.
feature_info('term')
df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))
df['term'].value_counts()

# grade feature
# Grade is part of sub_grade, so drop the grade feature.
df = df.drop(['grade'], axis=1)

# Convert the sub-grade into dummy variables.
# Then concatenate these new columns to the original dataframe.
# Drop the original sub-grade column.
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
print(dummies)

df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)
print(df.columns)

# verification_status, application_type,initial_list_status, purpose
# Convert these columns: ['verification_status', 'application_type', 'initial_list_status', 'purpose']
# into dummy variables and concatenate them with the original dataframe.
dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                         drop_first=True)
df = pd.concat(
    [df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1), dummies], axis=1)

# home_ownership2
df['home_ownership'].value_counts()

# Convert these to dummy variables, but replace NONE and ANY with OTHER, so that we end up with just 4 categories.
# Then concatenate them with the original dataframe.
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
df['home_ownership'].value_counts()

dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

# address
# Feature engineer a zip code column from the address in the data set.
# Create a column called 'zip_code' that extracts the zip code from the address column.
df['zip_code'] = df['address'].apply(lambda address: int(address[-5:]))
print(df['zip_code'])

df.head()
df['zip_code'].value_counts()

# Make this zip_code column into dummy variables using pandas.
# Concatenate the result and drop the original zip_code column along with dropping the address column.
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = pd.concat([df.drop(['zip_code'], axis=1), dummies], axis=1)
df.head()
df = df.drop('address', axis=1)

# issue_d
# Wouldn't know beforehand whether or not a loan would be issued when using our model,
# so in theory we wouldn't have an issue_date. Drop this feature.
feature_info('issue_d')
df = df.drop('issue_d', axis=1)

# earliest_cr_line
# Extract the year from this feature and convert it to a numeric feature.
# Set this new data to a feature column called 'earliest_cr_year' and drop the earliest_cr_line feature.
feature_info('earliest_cr_line')
df['earliest_cr_line'].value_counts()
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
df['earliest_cr_line'].value_counts()

# #########################################################
# Train Test Split
# #########################################################

# Drop the load_status column we created earlier, since its a duplicate of the loan_repaid column.
# We'll use the loan_repaid column since its already in 0s and 1s.
df = df.drop('loan_status', axis=1)

# Set X and y variables to the .values of the features and label.
X = df.drop('loan_repaid', axis=1).values
print(X)

y = df['loan_repaid'].values
print(y)

# df = df.sample(frac=0.1,random_state=101)
print(len(df))

# Perform a train/test split with test_size=0.2 and a random_state of 101.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# #########################################################
# Data Normalisation
# #########################################################

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #########################################################
# Model Creation
# #########################################################

# Build a sequential model to will be trained on the data.
print(X_train.shape)

model = Sequential()

model.add(Dense(units=78, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=39, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=19, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

# Fit the model to the training data.
# Also add in the validation data for later plotting. Use batching because we have a large dataset.
model.fit(x=X_train, y=y_train, epochs=150, batch_size=256, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save the model.
model.save('training_run_1.h5')

# #########################################################
# Model Performance Evaluation
# #########################################################

# Plot out the validation loss versus the training loss.
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.savefig('images/plot losses')
plt.show()

# Create predictions from the X_test set and display a classification report and confusion matrix for the X_test set.
predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

df['loan_repaid'].value_counts()
df['loan_repaid'].value_counts()[1] / len(df)

# Imbalanced data set => don't be fooled by the accuracy:
# if the model always reports true, it will be 80% correct all the time.
# Need to improve the recall and f1-score values for label '0'.

# Given the customer below, would you offer this person a loan?
random.seed(101)
random_ind = random.randint(0, len(df))

new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]
print(new_customer)

new_customer = scaler.transform(new_customer.values.reshape(1, 78))

model.predict_classes(new_customer)

# Now check, did this person actually end up paying back their loan?
print(df.iloc[random_ind]['loan_repaid'])
