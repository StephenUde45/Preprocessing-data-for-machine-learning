# Preprocessing-data-for-machine-learning
Preprocessed a csv file to get ready to input into a machine learning algorithm

# Python libraries
import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\UDESP\Downloads\PreProcessing Test.csv") # Sets the variable equal to the csv file
print(dataset.shape) # Tells me the rows by coloumns of the data set

x = dataset[['ID', 'Age', 'Income', 'Gender']].values # .value takes out the cell number. Prints out the independent variables
# print(x)

y = dataset[['Purchased']].values # Printing out the depedent variables
# print(y)

# Python library
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Replaces the blank cells which will later be filled by taking the mean of the column where the blank cell is
imputer.fit(x[:,1:3]) # Telling python to go through all rows in columns 1 through 3

x[:,1:3] = imputer.transform(x[:,1:3]) # Fills in the empty cells with the mean of that specific column
# print(x)

# Python library
from sklearn.preprocessing import LabelEncoder # Transforms words into numbers
labelEncoder_x = LabelEncoder() # States there are words in the independent variable x that need to be converted to numbers
x[:,3] = labelEncoder_x.fit_transform(x[:,3]) # Transforms the words in the third column to numbers for the computer to read it
# print(x)

labelEncoder_y = LabelEncoder() # States there are words in the dependent variable y that need to be converted to numbers
y[:,0] = labelEncoder_y.fit_transform(y[:,0]) # Transforms the words in the fourth column to numbers for the computer to read it
# print(y)

# Might not be needed for this dataset 
# Dummy Variables to make sure the machine learning algorithm knows that the numbers replacing the words aren't greater or less than each oher
# Python library
from sklearn.preprocessing import OneHotEncoder # Transforms the numbers represented as words into an array so the machine learning algorithm doesn't confuse the numbers as being less or greater than each other
oneHotEncoder1 = OneHotEncoder() # Creates an array for the words that have been converted to numbers
oneHotEncoder1.fit_transform(dataset.Gender.values.reshape(-1,1)).toarray() # Creates an array for the genders column so the computer doesn't mistakin the numbers being greater or less than one another

oneHotEncoder2 = OneHotEncoder()
oneHotEncoder2.fit_transform(dataset.Purchased.values.reshape(-1,1)).toarray() # Creates an array for the purchased column

# SPlitting data to training and test sets 
# Python library
from sklearn.model_selection import train_test_split # Splits the datasets and it is being utilzed below
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) # Train model with 80 percent and test model with 20 percent. Random state is used as
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# Python library
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) # Scales the training part of the input
x_test = sc_x.fit_transform(x_test) # Scales the testing part of the input

print(x_train)
print(x_test)

# Preprocessing step done. No apply to any machine learning algorithm
