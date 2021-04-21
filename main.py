import numpy
import pandas
import matplotlib.pyplot as pypl
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind
from tabulate import tabulate

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
        selector = SelectKBest(score_func=chi2, k=features_counter)
        selected_data = selector.fit_transform(X, y)
        # print(selector.get_support(indices=True))
 
        # then perform training and testing fold by fold for each classifier
        for fold, (train, test) in enumerate(rkf.split(selected_data, y)):
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

    #print(scores)
    numpy.save('scores', scores)

    # calculate means and standard deviations for comparison
    means = numpy.mean(scores, axis=2)
    deviations = numpy.std(scores, axis=2)

    for clf_id, clf_name in enumerate(clfs):
        print(f"Classifier: {clf_name}")
        for feature_index in range(0, number_of_features):
            tmp_classifier_mean = means[clf_id, feature_index]
            print("Number of features: %d, mean: %.3f, standard deviation: (%.3f)" % (feature_index + 1, tmp_classifier_mean, deviations[clf_id, feature_index]))

    best_mean = numpy.max(means)
    best_mean_clf_id = numpy.argmax(numpy.max(means, axis=1))
    best_mean_feature_index = numpy.argmax(numpy.max(means, axis=0))

    print(f"\nBest result globally: {best_mean}, with classifier {list(clfs.keys())[best_clf_id]} and feature_count: {best_feature_index + 1}")

    # we need to pick best number of features from each classifier
    best_feature_indeces = numpy.zeros(len(clfs))
    # we pick best number of features by its mean
    print("\nChosen best means:")
    for clf_id, clf_name in enumerate(clfs):
        # pick index of the highest mean within the classifier
        best_feature_index = numpy.argmax(means[clf_id])
        # pick highest mean within the classifier
        best_mean = numpy.max(means[clf_id])
        best_feature_indeces[clf_id] = best_feature_index
        print("Classifier: %s" % (clf_name))
        print("Number of features: %d mean: %.3f" % (best_feature_index + 1, best_mean))
    
    scores_stat = numpy.zeros((len(clfs), number_of_folds))
    for clf_id, clf_name in enumerate(clfs):
        scores_stat[clf_id] = scores[clf_id,best_feature_indeces[clf_id].astype(int)]

    print("Scores for statistics:")
    print(scores_stat)

    # now t-student statistic test
    alfa = .05
    t_statistic = numpy.zeros((len(clfs), len(clfs)))
    p_value = numpy.zeros((len(clfs), len(clfs)))

    # get t-statistics values
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores_stat[i], scores_stat[j])
    

    # print t-statistics and p-values in tables
    headers = [clf_name for clf_name in clfs.keys()]
    names_column = numpy.array([[clf_name] for clf_name in clfs.keys()])
    t_statistic_table = numpy.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = numpy.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    # calculate the advantage from t-statistics (if t_statistic>0 print 1 else 0) and display in a table
    advantage = numpy.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(numpy.concatenate((names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    # calculate significance (if p-value<alpha print 1 else 0) and display in the table
    significance = numpy.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(numpy.concatenate((names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)

    # get final results
    stat_better = significance * advantage
    stat_better_table = tabulate(numpy.concatenate((names_column, stat_better), axis=1), headers)
    print("\nStatistically significantly better:\n", stat_better_table)


if __name__ == '__main__':
    # convert_excel_to_csv()
    # generate_ranking()
    make_experiment()