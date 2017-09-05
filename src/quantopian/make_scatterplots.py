import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def main():
    data = pd.read_csv('./data/training_data.csv', index_col=0)

    X = data.drop(['OUTPUT'], axis=1)
    y = data['OUTPUT']

    vars = X.columns
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            x_var, y_var = vars[i], vars[j]

            print('%s (STD: %s), %s (STD: %s)' % (x_var, X[x_var].std(), y_var, X[y_var].std()))

            plt.figure()
            sc = plt.scatter(X[x_var], X[y_var], c=y, cmap='RdBu')
            plt.colorbar(sc)
            plt.xlabel(x_var)
            plt.ylabel(y_var)
            plt.tight_layout()
            plt.savefig('./plots/scatterplots/%s_vs_%s.png' % (x_var, y_var))


if __name__ == '__main__':
    main()