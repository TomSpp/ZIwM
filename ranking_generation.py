import pandas
import matplotlib.pyplot as pypl
from sklearn.feature_selection import SelectKBest, chi2

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