import pandas as pd


original = pd.read_csv("tf_saved_main.csv")

dev = pd.read_csv("tf_saved.csv")


original.columns
dev.columns

original.columns.isin(dev.columns).sum()

dev = dev.reindex(columns=original.columns)

dev.equals(original)

original.dtypes
dev.dtypes


for col in original.columns:
    if not original[col].equals(dev[col]):
        print(col)


dev.SFA.compare(original.SFA)
