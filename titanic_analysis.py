import pandas as pd

# Load the training and testing datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preview the datasets
print("Training Data:")
print(train_data.head())

print("\nTesting Data:")
print(test_data.head())

# Check for missing values in the training data
print("\nMissing values in training data:")
print(train_data.isnull().sum())

# Check for missing values in the testing data
print("\nMissing values in testing data:")
print(test_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

print("\nAfter handling missing values:")
print("Missing values in training data:")
print(train_data.isnull().sum())
print("\nMissing values in testing data:")
print(test_data.isnull().sum())



#converting non numerical data into numeric ones 
# cinversion for sex
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
# conversion for boarding place 
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_data['Embarked'] = test_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# removing names , passenfer id and ticket ( gives to much unimportant data )
train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

#Splitting the training data
X_train = train_data.drop('Survived', axis=1)  # Features
y_train = train_data['Survived']  # Target

#previewing the data 
print("\nFeatures (X_train):")
print(X_train.head())

print("\nTarget (y_train):")
print(y_train.head())


#importing model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Trainng model 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Predicting 
train_predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)

print("\nTraining Accuracy:", accuracy)

test_predictions = model.predict(test_data)

print("\nTest Predictions:")
print(test_predictions[:10])  # Show the first 10 predictions

# preparing for submission in csv format 
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],  # Use the original test file to get PassengerId
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")