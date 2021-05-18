import numpy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

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

clfs = {
    'k3euclidean': KNeighborsClassifier(n_neighbors=3, metric='euclidean'),
    'k5euclidean': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'k9euclidean': KNeighborsClassifier(n_neighbors=9, metric='euclidean'),
    'k3manhattan': KNeighborsClassifier(n_neighbors=3, metric='manhattan'),
    'k5manhattan': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'k9manhattan': KNeighborsClassifier(n_neighbors=9, metric='manhattan')
}
feature_range = np.arange(1, len(dataset_features)-1)

scores = numpy.load('scores.npy')
means = numpy.mean(scores, axis=2)
means = (means*100)

k3euclidean = patches.Patch(color='grey', label='euclidean, k=3')
k5euclidean = patches.Patch(color='green', label='euclidean, k=5')
k9euclidean = patches.Patch(color='red', label='euclidean, k=9')
k3manhattan = patches.Patch(color='yellow', label='manhattan, k=3')
k5manhattan = patches.Patch(color='blue', label='manhattan, k=5')
k9manhattan = patches.Patch(color='pink', label='manhattan, k=9')


euclidean_classifiers = [clf for clf in clfs.items() if clf[1].metric == 'euclidean']
manhattan_classifiers = [clf for clf in clfs.items() if clf[1].metric == 'manhattan']


def draw_all():
    legend = [k3euclidean, k5euclidean, k9euclidean, k3manhattan, k5manhattan, k9manhattan]

    plt.figure(figsize=(6, 8))
    for clf_id, clf in enumerate(clfs):
        plt.plot(feature_range, means[clf_id], legend[clf_id]._original_facecolor)

    plt.legend(handles=legend, loc=2)

    axes = plt.gca()
    axes.set_xlim([1, 20])
    axes.set_ylim([1, 35])
    x_ticks = numpy.arange(1, 20, 1)
    y_tics = numpy.arange(5, 40, 5)
    plt.xticks(x_ticks)
    plt.yticks(y_tics)
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

    plt.xlabel('Liczba najlepszych cech', fontsize=16)
    plt.ylabel('Dokładność klasyfikacji', fontsize=16)
    plt.grid(axis='x')
    plt.grid(axis='y')

    plt.savefig(fname='all_classifiers.eps')
    plt.show()


def draw_metric(classifiers):
    metric = classifiers[0][1].metric

    grey = patches.Patch(color='grey', label=f'{metric}, k=3')
    green = patches.Patch(color='green', label=f'{metric}, k=5')
    red = patches.Patch(color='red', label=f'{metric}, k=9')
    legend = [grey, green, red]

    plt.figure(figsize=(6, 8))
    for index, clf in enumerate(classifiers):
        clf_id = list(clfs).index(clf[0])
        plt.plot(feature_range, means[clf_id], legend[index]._original_facecolor)

    plt.legend(handles=legend, loc=2)

    axes = plt.gca()
    axes.set_xlim([1, 20])
    axes.set_ylim([1, 35])
    x_ticks = numpy.arange(1, 20, 1)
    y_tics = numpy.arange(5, 40, 5)
    plt.xticks(x_ticks)
    plt.yticks(y_tics)
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

    plt.xlabel('Liczba najlepszych cech', fontsize=16)
    plt.ylabel('Dokładność klasyfikacji', fontsize=16)
    plt.grid(axis='x')
    plt.grid(axis='y')

    plt.savefig(fname=f'{metric}_plot.eps')
    plt.show()


# draw_all()
# draw_metric(manhattan_classifiers)
draw_metric(euclidean_classifiers)