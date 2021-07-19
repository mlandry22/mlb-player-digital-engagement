import numpy as np
import pandas as pd
from pathlib import Path

input_file_path = Path('./data')



for file in ['train']:
    # drop playerTwitterFollowers, teamTwitterFollowers from example_test
    df = pd.read_csv(input_file_path / f"{file}.csv").dropna(axis=1,how='all')
    daily_data_nested_df_names = df.drop('date', axis = 1).columns.values.tolist()

    for df_name in daily_data_nested_df_names:
        date_nested_table = df[['date', df_name]]

        date_nested_table = (date_nested_table[
          ~pd.isna(date_nested_table[df_name])
          ].
          reset_index(drop = True)
          )

        daily_dfs_collection = []

        for date_index, date_row in date_nested_table.iterrows():
            daily_df = pd.read_json(date_row[df_name])

            daily_df['dailyDataDate'] = date_row['date']

            daily_dfs_collection = daily_dfs_collection + [daily_df]

        # Concatenate all daily dfs into single df for each row
        unnested_table = (pd.concat(daily_dfs_collection,
          ignore_index = True).
          # Set and reset index to move 'dailyDataDate' to front of df
          set_index('dailyDataDate').
          reset_index()
          )
        #print(f"{file}_{df_name}.pickle")
        #display(unnested_table.head(3))
        unnested_table.to_pickle(f"./data/{file}_{df_name}.pickle")
        #print('\n'*2)
