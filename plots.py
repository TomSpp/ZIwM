import numpy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

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

k3euclidean = Line2D(feature_range, means[0], c='black', label='euclidean, k=3', linestyle='--')
k5euclidean = Line2D(feature_range, means[1], c='blue', label='euclidean, k=5', linestyle='--')
k9euclidean = Line2D(feature_range, means[2], c='red', label='euclidean, k=9', linestyle='--')
k3manhattan = Line2D(feature_range, means[3], c='black', label='manhattan, k=3')
k5manhattan = Line2D(feature_range, means[4], c='blue', label='manhattan, k=5')
k9manhattan = Line2D(feature_range, means[5], c='red', label='manhattan, k=9')


legend = [k3euclidean, k5euclidean, k9euclidean, k3manhattan, k5manhattan, k9manhattan]

plt.figure(figsize=(6, 8))
for clf_id, clf in enumerate(clfs):
    plt.plot(feature_range, means[clf_id], color=legend[clf_id].get_color(), linestyle=legend[clf_id].get_linestyle())

plt.legend(handles=legend, loc=2)

axes = plt.gca()
axes.set_xlim([1, 20])
axes.set_ylim([1, 35])
x_ticks = numpy.arange(1, 21, 1)
y_tics = numpy.arange(5, 40, 5)
plt.xticks(x_ticks)
plt.yticks(y_tics)
axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

plt.xlabel('Liczba cech', fontsize=12)
plt.ylabel('Dokładność klasyfikacji', fontsize=12)
plt.grid(axis='x')
plt.grid(axis='y')

plt.savefig(fname='all_classifiers.eps')
plt.show()
