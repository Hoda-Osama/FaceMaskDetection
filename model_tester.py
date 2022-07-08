
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


features = np.load('features.npy')
labels = np.load('labels.npy')

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=2209)

# Using a decision tree classifier
clf = DecisionTreeClassifier(
    criterion='entropy', random_state=2209)
clf = clf.fit(train_features, train_labels)

clf_predictions = clf.predict(test_features)
clf_accuracy = accuracy_score(test_labels, clf_predictions)
clf_confusion = confusion_matrix(test_labels, clf_predictions)
clf_report = classification_report(test_labels, clf_predictions)

# Saving the results to a text file
f = open('model_comparison.txt', 'a')

f.write('Using Decision Tree:')
f.write('\n\nAccuracy = %f\n\n' % clf_accuracy)
f.write('-' * 80)
f.write('\n\nConfusion Matrix = ' + clf_confusion.__repr__() + '\n\n')
f.write('-' * 80)
f.write('\n\nClassification Report:\n\n' + clf_report)
f.write('-' * 80)
f.close()


# Using a random forest classifier
clf1 = RandomForestClassifier(random_state=2209)
clf1 = clf1.fit(train_features, train_labels)

clf1_predictions = clf1.predict(test_features)
clf1_accuracy = accuracy_score(test_labels, clf1_predictions)
clf1_confusion = confusion_matrix(test_labels, clf1_predictions)
clf1_report = classification_report(test_labels, clf1_predictions)

# Creating a file to save the results in
f = open('model_comparison.txt', 'a')

f.write('\n\nUsing Random Forest:')
f.write('\n\nAccuracy = %f\n\n' % clf1_accuracy)
f.write('-' * 80)
f.write('\n\nConfusion Matrix = ' + clf1_confusion.__repr__() + '\n\n')
f.write('-' * 80)
f.write('\n\nClassification Report:\n\n' + clf1_report)
f.write('-' * 80)
f.close()

# Using a bagging classifier
clf2 = BaggingClassifier(base_estimator=clf, random_state=2209)
clf2 = clf2.fit(train_features, train_labels)

clf2_predictions = clf2.predict(test_features)
clf2_accuracy = accuracy_score(test_labels, clf2_predictions)
clf2_confusion = confusion_matrix(test_labels, clf2_predictions)
clf2_report = classification_report(test_labels, clf2_predictions)

# Creating a file to save the results in
f = open('model_comparison.txt', 'a')

f.write('\n\nUsing Bagging Classifier:')
f.write('\n\nAccuracy = %f\n\n' % clf2_accuracy)
f.write('-' * 80)
f.write('\n\nConfusion Matrix = ' + clf2_confusion.__repr__() + '\n\n')
f.write('-' * 80)
f.write('\n\nClassification Report:\n\n' + clf2_report)
f.write('-' * 80)
f.close()


# Using adaboost classifier
clf3 = AdaBoostClassifier(base_estimator=clf, random_state=2209)
clf3 = clf3.fit(train_features, train_labels)

clf3_predictions = clf3.predict(test_features)
clf3_accuracy = accuracy_score(test_labels, clf3_predictions)
clf3_confusion = confusion_matrix(test_labels, clf3_predictions)
clf3_report = classification_report(test_labels, clf3_predictions)

# Creating a file to save the results in
f = open('model_comparison.txt', 'a')

f.write('\n\nUsing AdaBoost Classifier:')
f.write('\n\nAccuracy = %f\n\n' % clf3_accuracy)
f.write('-' * 80)
f.write('\n\nConfusion Matrix = ' + clf3_confusion.__repr__() + '\n\n')
f.write('-' * 80)
f.write('\n\nClassification Report:\n\n' + clf3_report)
f.write('-' * 80)
f.close()
