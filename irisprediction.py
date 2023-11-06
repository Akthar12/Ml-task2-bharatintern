import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

#load the dataset
data = pd.read_csv("Iris.csv")

# separate features (x) and target (y)
x = data.drop('Species', axis=1)
y = data['Species']

# convert species labels into numerical using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

# Create a Logistic regression model
model = LogisticRegression()

# Create a Logistic regression model
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)


#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#Generate classification repor
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Suppose you want to predict the species for a new flower with given features
new_flower_features = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})
