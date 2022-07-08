
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Loading the features and labels required for all experiments
features = np.load('features.npy')
features_smooth = np.load('feature_smooth.npy')
less_features = np.load('less_features.npy')
less_features_smooth = np.load('less_feature_smooth.npy')
labels = np.load('labels.npy')


# This function trains and tests a bagging classifier with given parameters for each experiment
def supervised_experiment(selected_features, test_ratio, filename, title):
    global labels

    # Splitting the dataset into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        selected_features, labels, test_size=test_ratio, random_state=2209)
    # Using a random forest classifier
    clf = RandomForestClassifier(random_state=2209)
    clf = clf.fit(train_features, train_labels)

    clf_predictions = clf.predict(test_features)
    clf_accuracy = accuracy_score(test_labels, clf_predictions)
    clf_confusion = confusion_matrix(test_labels, clf_predictions)
    clf_report = classification_report(test_labels, clf_predictions)

    # Creating a file to save the results in
    f = open(filename, 'w')

    f.write(title)
    f.write('\n\nAccuracy = %f\n\n' % clf_accuracy)
    f.write('-' * 80)
    f.write('\n\nConfusion Matrix = ' + clf_confusion.__repr__() + '\n\n')
    f.write('-' * 80)
    f.write('\n\nClassification Report:\n\n' + clf_report)
    f.close()


# This function applies K-Means clustering for the unsupervised experiment
def unsupervised_experiment(selected_features, test_ratio, filename, title):
    global labels

    # Splitting the dataset into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        selected_features, labels, test_size=test_ratio, random_state=2209)

    # Set 'n_clusters' to 2 since we have 2 classes
    kmeans = KMeans(n_clusters=2, random_state=2209)

    # Training the model without providing the labels
    kmeans.fit(train_features)

    kmeans_predictions = kmeans.predict(test_features)
    kmeans_accuracy = accuracy_score(test_labels, kmeans_predictions)
    kmeans_confusion = confusion_matrix(test_labels, kmeans_predictions)
    kmeans_report = classification_report(test_labels, kmeans_predictions)

    # Creating a file to save the results in
    f = open(filename, 'w')

    f.write(title)
    f.write('\n\nAccuracy = %f\n\n' % kmeans_accuracy)
    f.write('-' * 80)
    f.write('\n\nConfusion Matrix = ' + kmeans_confusion.__repr__() + '\n\n')
    f.write('-' * 80)
    f.write('\n\nClassification Report:\n\n' + kmeans_report)
    f.close()


# Experiment 1: Training the model with the complete dataset with 80% training
supervised_experiment(features, 0.2, 'experiment1_results.txt',
                      'Using Complete Dataset with Random Forest Classifier:')


# Experiment 2: The same but with 50% smaller training data (which means 60% testing)
supervised_experiment(features, 0.6, 'experiment2_results.txt',
                      'Using Complete Dataset with Random Forest Classifier and 60% Testing Data:')


# Experiment 3: Using the complete dataset but with less features for each image
supervised_experiment(less_features, 0.2, 'experiment3_results.txt',
                      'Using Complete Dataset with Random Forest Classifier and Less Features:')


# Experiment 4: Using the complete dataset but with smoothed images (erosion)
supervised_experiment(features_smooth, 0.2, 'experiment4_results.txt',
                      'Using Complete Dataset with Random Forest Classifier and Smoothed Images:')


# Experiment 5: Using K-means clustering with the complete dataset for unsupervised learning
unsupervised_experiment(features, 0.2, 'experiment5_results.txt',
                        'Using Complete Dataset with K-Means Clustering:')
