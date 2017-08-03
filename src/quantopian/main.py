import pandas as pd


def main():
    all_pairs = pd.read_csv('./data/pedro-2017-08-03-all-pairs.csv', parse_dates=True, index_col=0)

    all_pairs_in = all_pairs['2013':'2015'].copy()
    all_pairs_out = all_pairs['2016':'2016'].copy()

    print(all_pairs_in.shape, all_pairs_out.shape)


if __name__ == '__main__':
    main()