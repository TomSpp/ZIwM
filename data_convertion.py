import pandas

read_file = pandas.read_excel(r'bialaczka.XLS')
read_file.to_csv(r'bialaczka.csv', index=None, header=False)