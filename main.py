import numpy
import pandas
import matplotlib.pyplot as pypl
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


def convert_excel_to_csv():
    read_file = pandas.read_excel(r'bialaczka.XLS')
    read_file.to_csv(r'bialaczka.csv', index=None, header=False)


def generate_ranking():
    dataset = pandas.read_csv(r'bialaczka.csv', sep=",", header=None)
    dataset.columns = dataset_features
    dataset.info()

    X = dataset.drop(columns=['Klasa', 'ID w klasie'])
    y = dataset['Klasa']

    features_amount = X.shape[1]
    selector = SelectKBest(score_func=chi2, k=features_amount)
    selector.fit(X, y)
    ranking = [
        (name, round(score, 2))
        for name, score in zip(X.columns, selector.scores_)
    ]

    ranking.sort(reverse=False, key=lambda f: f[1])
    for i, feature in enumerate(ranking, 1):
        print(f"{i}. {feature[0]} {feature[1]}")

    fig = pypl.figure(figsize=(10, 12))
    fig.add_axes([0.3, 0.1, 0.3, 0.8])
    pypl.barh(range(len(ranking)), [feature[1] for feature in ranking], align='center')
    pypl.yticks(range(len(ranking)), [feature[0] for feature in ranking])
    pypl.xlabel('Wartość testu chi-squared')
    pypl.grid(axis='x')
    pypl.savefig(fname='ranking.eps')
    pypl.show()


def make_experiment():
    dataset = pandas.read_csv(r'bialaczka.csv', sep=",", header=None)
    dataset.columns = dataset_features

    X = dataset.drop(columns=['Klasa', 'ID w klasie'])
    y = dataset['Klasa']

    clfs = {
        'k3euclidean': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
        'k5euclidean': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'k9euclidean': KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
        'k3manhattan': KNeighborsClassifier(n_neighbors=3, metric='manhattan'),
        'k5manhattan': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
        'k9manhattan': KNeighborsClassifier(n_neighbors=9, metric='manhattan')
    }

    n_splits = 2
    n_repeats = 5
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    number_of_features = len(X.columns)
    scores = numpy.zeros((len(clfs), number_of_features, n_splits*n_repeats))

    for feature_index in range(0, number_of_features):
        features_counter = feature_index + 1
        selector = SelectKBest(score_func=chi2, k=features_counter)
        selected_data = selector.fit_transform(X, y)
        # print(selector.get_support(indices=True))
        for fold, (train, test) in enumerate(rkf.split(selected_data, y)):
            for clf_idx, clf_name in enumerate(clfs):
                X_train, X_test = selected_data[train], selected_data[test]
                y_train, y_test = y[train], y[test]
                clf = clone(clfs[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                scores[clf_idx, feature_index, fold] = accuracy_score(y_test, y_pred)

    # print(scores)
    numpy.save('scores', scores)

    means = numpy.mean(scores, axis=2)
    stds = numpy.std(scores, axis=2)

    for clf_id, clf_name in enumerate(clfs):
        print(f"classifier:{clf_name}")
        for feature_index in range(0, number_of_features):
            current_classifier_mean = means[clf_id, feature_index]
            print("features: %d, mean: %.3f, std: (%.2f)" % (
            feature_index + 1, current_classifier_mean, stds[clf_id, feature_index]))

    best_mean = numpy.max(means)
    best_clf_id = numpy.argmax(numpy.max(means, axis=1))
    best_feature_index = numpy.argmax(numpy.max(means, axis=0))

    print(
        f"\nBest result: {best_mean} with classifier {list(clfs.keys())[best_clf_id]} and feature_count equal "
        f"{best_feature_index + 1}")


if __name__ == '__main__':
    # convert_excel_to_csv()
    # generate_ranking()
    make_experiment()