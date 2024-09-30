import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/sonali gupta31102003/Downloads/titanic.csv")

df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0], 'Fare': df['Fare'].median()}, inplace=True)
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df = pd.get_dummies(df, drop_first=True)

for col in ['Age', 'Fare']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]


missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    sns.barplot(x=missing_values.index, y=missing_values.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title('Missing Values in Each Column')
    plt.show()

# Visualize Survived distribution
sns.set_style("whitegrid")
sns.countplot(x="Survived", data=df)
plt.title('Survival Count')
plt.show()

# Plot Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True, color='red')
plt.title('Age Distribution')
plt.show()

# Plot Fare Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['Fare'], bins=20, kde=True, color='purple')
plt.title('Fare Distribution')
plt.show()

# Visualize survival count by different variables
catplot_titles = [
    ('Pclass', 'Survived', 'Survival Count by Passenger Class'),
    ('Sex_male', 'Survived', 'Survival Count by Gender'),
    ('Embarked_Q', 'Survived', 'Survival Count by Embarkation Point')
]
for x, hue, title in catplot_titles:
    sns.catplot(data=df, x=x, hue=hue, kind='count', palette='husl', height=4, aspect=1.5).set_titles(title)
    plt.show()

boxplot_titles = [
    ('Pclass', 'Age', 'Survived', 'Age Distribution across Passenger Classes with Survival'),
    ('Pclass', 'Fare', 'Survived', 'Fare Distribution across Passenger Classes with Survival')
]
for x, y, hue, title in boxplot_titles:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette='husl')
    plt.title(title)
    plt.show()

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

X, y = df.drop('Survived', axis=1), df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nConfusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report: \n{classification_report(y_test, y_pred)}")
print(f"\nAccuracy Score: \n{accuracy_score(y_test, y_pred)}")

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False)
print(f"\nFeature Importances:\n{feature_importances}")

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.title('Feature Importance')
plt.show()
