import pandas as pd


original = pd.read_csv("tf_saved_main.csv")

dev = pd.read_csv("tf_saved.csv")


original.columns
dev.columns

original.columns.isin(dev.columns).sum()

# dev = dev.reindex(columns=original.columns)

dev.equals(original)

original.dtypes
dev.dtypes


for col in original.columns:
    if not original[col].equals(dev[col]):
        print(col)


dev.NLROM.compare(original.NLROM)
dev.teamId.compare(original.teamId)

original.at[12366, "SFA"]
original.loc[12366]
dev.at[12366, "SFA"]
dev.loc[12366]


original[original.dailyDataDate == 20180101]
dev[dev.dailyDataDate == 20180101]
