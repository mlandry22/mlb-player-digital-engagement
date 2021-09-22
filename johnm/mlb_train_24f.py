
# run this script after running mlb_dataprep_24c.py

import os
import pandas as pd
import numpy as np
import lightgbm as lgb


tf = pd.read_csv("./tf_saved_725f.csv")

##
tf['season'] =  tf.season.fillna(method='ffill').fillna(method='bfill').astype(int)

excl_cols = ['hitterId','pitcherId_x','pitcherId_y', 'playerId','gamePk','dailyDataDate',
            'engagementMetricsDate','target1','target2','target3','target4',
             "gameDate", "gameTimeUTC", "teamName", "playerName", "jerseyNum",
             "positionName", "positionType", "target1_p_mean", "target2_p_mean",
             "target3_p_mean","target4_p_mean",
             "gameTimeUTC_team_box_score", "gameDate_team_box_score",
             "dailyDataDate_team_box_score","dailyDataDate_team_standings",
             "dailyDataDate_lead", 'playerForTestSetAndFuturePreds','hr_rank',
             'team_games_played', 'dayNight','homeWinPct','awayWinPct','homeScore',
             'awayScore','homeWinner',
             "ddd_month"]
use_cols = [col for col in tf.columns if col not in excl_cols]


###  correlated columns
# tf_samp = tf.sample(frac=0.01)[use_cols]
# corr_matrix = tf_samp[use_cols].corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# correlates = [column for column in upper.columns if any(upper[column] == 1)]


correlates = ['hitBatsmen',
 'no_hitter',
 'home_team_box_score',
 'hitBatsmen_team_box_score',
 'season_team_standings',
 'sportGamesBack',
 'nlWins',
 'nlLosses',
 'errors_1_games_ago']

tf = tf.drop(columns=correlates)
use_cols = [col for col in tf.columns if col not in (excl_cols+correlates)]


params = {    'num_leaves': 73,
              'min_data_in_leaf': 10,
              'objective': 'regression_l1',
              'max_depth': -1,
            #   'learning_rate': 0.05,
              "boosting": 'gbdt',
              "feature_fraction": 0.3,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 12345,
              'min_sum_hessian_in_leaf': 0.00,
              'min_gain_to_split': 0.0,
              "max_bin": 255,
              "metric": 'mae',
              "multi_error_top_k": 1,
              "num_classes": 1,
              "lambda_l1": 0.0,
              "lambda_l2": 0.0,
              'alpha': 0.01, # only for Huber and Quantile regression
              "verbosity": 1,
              "nthread": 16,
              "random_state": 12345,
#              "top_rate": 0.8, # only for GOSS
#              "other_rate": 0.2, # only for GOSS
              'boost_from_average': False,
              'reg_sqrt': False, # Change to false for regression_l2
              "device_type": 'gpu',
              "gpu_device_id": 0
              }  ##gpu



model_dict = {}
for target in [
    'target2',
    'target4',
    'target3',
    'target1'
]:
    LR_BASE = 0.001 # if target ### =="target2" else 0.02
    LR_EVAL = 1000

    params['learning_rate'] = LR_BASE
    TREES = 17600  # if target=='target1' else 8700


    lgb_dtrain = lgb.Dataset(tf[use_cols], tf[target], free_raw_data=False)

    np.random.seed(123456)
    lgb_model = lgb.train(params,
                    train_set = lgb_dtrain,
                    num_boost_round = TREES, # 200000,
                    verbose_eval=LR_EVAL,
                    # eval_train_metric=True,
                    # show_stdv=True,
                    # return_cvbooster=True
                    )
    if not os.path.exists("./alldata"):
        os.makedirs("./alldata")
    lgb_model.save_model(f"./alldata/lgb_{target}_dubs_tripsX_all.txt")

