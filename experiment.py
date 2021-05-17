import numpy
import pandas
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score


dataset_features = [
    'Klasa',
    'ID w klasie',
    'Temperatura',
    'Anemia',
    'Stopień krwawienia',
    'Miejsce krwawienia',
    'Bóle kości',
    'Wrażliwość mostka',
    'Powiększenie węzłów chłonnych',
    'Powiększenie wątroby i śledziony',
    'Centralny układ nerwowy',
    'Powiększenie jąder',
    'Uszkodzenie w sercu, płucach, nerce',
    'Gałka oczna',
    'Poziom WBC (leukocytów)',
    'Obniżenie RBC (erytrocytów)',
    'Liczba płytek krwi',
    'Niedojrzałe komórki (blastyczne)',
    'Stan pobudzenia szpiku',
    'Główne komórki w szpiku',
    'Poziom limfocytów',
    'Reakcja',
]


numpy.set_printoptions(precision=3)

dataset = pandas.read_csv(r'bialaczka.csv', sep=",", header=None)
dataset.columns = dataset_features

# X - set of features
X = dataset.drop(columns=['Klasa', 'ID w klasie'])
# y - set of classes
y = dataset['Klasa']

# tested classifiers
clfs = {
    'k3euclidean': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
    'k5euclidean': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'k9euclidean': KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
    'k3manhattan': KNeighborsClassifier(n_neighbors=3, metric='manhattan'),
    'k5manhattan': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'k9manhattan': KNeighborsClassifier(n_neighbors=9, metric='manhattan')
}

# number of splits in K-fold
n_splits = 2
# number of repetitions in K-fold
n_repeats = 5
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

number_of_features = len(X.columns)
number_of_folds = n_splits*n_repeats
scores = numpy.zeros((len(clfs), number_of_features, number_of_folds))

# first we need to choose features to take into considerations
for feature_index in range(0, number_of_features):
    features_counter = feature_index + 1

    # then perform training and testing fold by fold for each classifier
    for fold, (train, test) in enumerate(rkf.split(X, y)):
        selector = SelectKBest(score_func=chi2, k=features_counter)
        selected_data = selector.fit_transform(X, y)
        for clf_idx, clf_name in enumerate(clfs):
            X_train, X_test = selected_data[train], selected_data[test]
            y_train, y_test = y[train], y[test]
            # it is good habit to use clone here
            clf = clone(clfs[clf_name])
            # training part
            clf.fit(X_train, y_train)
            # prediction part
            y_pred = clf.predict(X_test)
            # calculating accuracy
            scores[clf_idx, feature_index, fold] = accuracy_score(y_test, y_pred)

means = numpy.mean(scores, axis=2)
deviations = numpy.std(scores, axis=2)
for clf_id, clf_name in enumerate(clfs):
    print(f"Classifier: {clf_name}")
    for feature_index in range(0, number_of_features):
        tmp_classifier_mean = means[clf_id, feature_index]
        print("Number of features: %d, mean: %.3f, standard deviation: (%.3f)" % (
        feature_index + 1, tmp_classifier_mean, deviations[clf_id, feature_index]))

best_mean = numpy.max(means)
best_mean_clf_id = numpy.argmax(numpy.max(means, axis=1))
best_mean_feature_index = numpy.argmax(numpy.max(means, axis=0))

print(
    f"\nBest result globally: {best_mean}, with classifier {list(clfs.keys())[best_mean_clf_id]} and feature_count: {best_mean_feature_index + 1}")
#print(scores)
numpy.save('scores', scores)
