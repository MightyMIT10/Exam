# Multivariate Bernoulli distribution

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMS Spam dataset
sms_df = pd.read_csv('Spam.csv', encoding='latin-1')

# Extract features and Label
X = sms_df['v2']
y = sms_df['v1']

# Convert Labels to binary (1 for spam, e for ham)q
y = y.map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a vectorizer to convert text data into a binary feature matrix
vectorizer=CountVectorizer(binary=True)
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

# Initialize the Multivariate Bernoulli Naive Bayes classifier 
clf = BernoulliNB()
# Train the classifier on the training data 
clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred=clf.predict(X_test)

# Calculate the accuracy of the classifier 
accuracy = accuracy_score (y_test, y_pred) 
print("Accuracy:", accuracy)

confusion_mat = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Print classification report
print("Classification Report: \n", classification_report (y_test, y_pred))








# Multivariate Bernoulli distribution

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMS Spam dataset
sms_df = pd.read_csv('Spam.csv', encoding='latin-1')

# Extract features and Label
X = sms_df['v2']
y = sms_df['v1']

# Convert Labels to binary (1 for spam, e for ham)
y = y.map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a vectorizer to convert text data into a binary feature matrix
vectorizer=CountVectorizer(binary=True)
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

# Initialize the Multivariate Bernoulli Naive Bayes classifier 
clf = MultinomialNB()
# Train the classifier on the training data 
clf.fit(X_train, y_train)
# Make predictions on the test data
y_pred=clf.predict(X_test)

# Calculate the accuracy of the classifier 
accuracy = accuracy_score (y_test, y_pred) 
print("Accuracy:", accuracy)

confusion_mat = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Print classification report
print("Classification Report: \n", classification_report (y_test, y_pred))





# Multivariate Bernoulli distribution

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMS Spam dataset
sms_df = pd.read_csv('Spam.csv', encoding='latin-1')

# Extract features and Label
X = sms_df['v2']
y = sms_df['v1']

# Convert Labels to binary (1 for spam, 0 for ham)
y = y.map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a vectorizer to convert text data into a binary feature matrix
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Convert the sparse matrix to a dense numpy array
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Initialize the Gaussian Naive Bayes classifier 
clf = GaussianNB()

# Train the classifier on the training data (with dense matrix)
clf.fit(X_train_dense, y_train)

# Make predictions on the test data (with dense matrix)
y_pred = clf.predict(X_test_dense)

# Calculate the accuracy of the classifier 
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion_mat = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print("Classification Report: \n", classification_report(y_test, y_pred))
