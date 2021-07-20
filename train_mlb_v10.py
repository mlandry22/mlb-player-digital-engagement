#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 19:26:54 2021

@author: branden
"""
import os
import pandas as pd
import numpy as np
import pickle as pkl
import itertools
from tqdm import tqdm
from collections import defaultdict
import lightgbm as lgb
import re
from fuzzywuzzy import fuzz
import numba as nb
from numba import njit
import time


@njit
def nb_cumsum(arr):
    return arr.cumsum()

@njit
def nb_sum(arr):
    return arr.sum()


def unnest(data, name):

    date_nested_table = data[['date', name]]

    date_nested_table = (date_nested_table[
      ~pd.isna(date_nested_table[name])
      ].
      reset_index(drop = True)
      )

    daily_dfs_collection = []

    for date_index, date_row in date_nested_table.iterrows():
        daily_df = pd.read_json(date_row[name])

        daily_df['dailyDataDate'] = date_row['date']

        daily_dfs_collection = daily_dfs_collection + [daily_df]

    if daily_dfs_collection:
        # Concatenate all daily dfs into single df for each row
        unnested_table = (pd.concat(daily_dfs_collection,
          ignore_index = True).
          # Set and reset index to move 'dailyDataDate' to front of df
          set_index('dailyDataDate').
          reset_index()
          )
        return unnested_table
    else:
        return pd.DataFrame()


def get_unnested_data_dict(data, daily_data_nested_df_names):
    df_dict = {}
    for df_name in daily_data_nested_df_names:
        df_dict[df_name] = unnest(data, df_name)
    return df_dict

def get_unnested_data(data, colnames):
    return (unnest(data, df_name) for df_name in colnames)


## Find win expectancy and volatility given inning, out, base, run situation.
## https://gist.github.com/JeffSackmann/815377
## no. of runs that score with HR in diff. base situations
baseHr = {1: 1,
          2: 2,
          3: 2,
          4: 3,
          5: 2,
          6: 3,
          7: 3,
          8: 4
          }

tangoRunExp = {'60': {1: 0.51400000000000001, 2: 0.19400000000000001, 3: 0.14999999999999999, 4: 0.076999999999999999, 5: 0.036999999999999998, 6: 0.017000000000000001, 7: 0.0060000000000000001, 8: 0.0030000000000000001, 9: 0.001, 10: 0.001, 'm': -0.216, 'b': 0.247}, '61': {1: 0.59599999999999997, 2: 0.17599999999999999, 3: 0.13200000000000001, 4: 0.057000000000000002, 5: 0.024, 6: 0.0089999999999999993, 7: 0.0040000000000000001, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.11600000000000001, 'b': 0.40600000000000003}, '62': {1: 0.55900000000000005, 2: 0.20599999999999999, 3: 0.158, 4: 0.051999999999999998, 5: 0.017000000000000001, 6: 0.0050000000000000001, 7: 0.002, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.19900000000000001, 'b': 0.82799999999999996}, '82': {1: 0.27300000000000002, 2: 0.35499999999999998, 3: 0.17000000000000001, 4: 0.13800000000000001, 5: 0.041000000000000002, 6: 0.014999999999999999, 7: 0.0050000000000000001, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.20899999999999999, 'b': 0.78900000000000003}, '80': {1: 0.311, 2: 0.247, 3: 0.17000000000000001, 4: 0.14399999999999999, 5: 0.070999999999999994, 6: 0.031, 7: 0.012999999999999999, 8: 0.0080000000000000002, 9: 0.0030000000000000001, 10: 0.002, 'm': -0.127, 'b': 0.193}, '81': {1: 0.39700000000000002, 2: 0.24399999999999999, 3: 0.151, 4: 0.123, 5: 0.050999999999999997, 6: 0.021000000000000001, 7: 0.0080000000000000002, 8: 0.0030000000000000001, 9: 0.001, 10: 0.0, 'm': -0.14199999999999999, 'b': 0.40200000000000002}, '20': {1: 0.42399999999999999, 2: 0.29899999999999999, 3: 0.14999999999999999, 4: 0.071999999999999995, 5: 0.032000000000000001, 6: 0.012999999999999999, 7: 0.0050000000000000001, 8: 0.002, 9: 0.001, 10: 0.0, 'm': -0.27800000000000002, 'b': 0.71599999999999997}, '21': {1: 0.44400000000000001, 2: 0.32600000000000001, 3: 0.13700000000000001, 4: 0.056000000000000001, 5: 0.021999999999999999, 6: 0.0089999999999999993, 7: 0.0030000000000000001, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.26800000000000002, 'b': 0.86299999999999999}, '22': {1: 0.45300000000000001, 2: 0.374, 3: 0.11600000000000001, 4: 0.039, 5: 0.012, 6: 0.0050000000000000001, 7: 0.001, 8: 0.0, 9: 0.0, 10: 0.0, 'm': -0.19400000000000001, 'b': 0.97399999999999998}, '42': {1: 0.49399999999999999, 2: 0.23699999999999999, 3: 0.18099999999999999, 4: 0.059999999999999998, 5: 0.017999999999999999, 6: 0.0070000000000000001, 7: 0.002, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.14399999999999999, 'b': 0.84099999999999997}, '40': {1: 0.36199999999999999, 2: 0.25600000000000001, 3: 0.19400000000000001, 4: 0.104, 5: 0.048000000000000001, 6: 0.02, 7: 0.0089999999999999993, 8: 0.0040000000000000001, 9: 0.002, 10: 0.001, 'm': -0.16700000000000001, 'b': 0.45000000000000001}, '41': {1: 0.40100000000000002, 2: 0.25800000000000001, 3: 0.20300000000000001, 4: 0.083000000000000004, 5: 0.034000000000000002, 6: 0.012999999999999999, 7: 0.0050000000000000001, 8: 0.002, 9: 0.001, 10: 0.0, 'm': -0.17399999999999999, 'b': 0.66900000000000004}, '72': {1: 0.185, 2: 0.54800000000000004, 3: 0.16900000000000001, 4: 0.067000000000000004, 5: 0.023, 6: 0.0060000000000000001, 7: 0.002, 8: 0.0, 9: 0.0, 10: 0.0, 'm': -0.095000000000000001, 'b': 0.78500000000000003}, '71': {1: 0.41299999999999998, 2: 0.32800000000000001, 3: 0.13800000000000001, 4: 0.072999999999999995, 5: 0.029000000000000001, 6: 0.010999999999999999, 7: 0.0050000000000000001, 8: 0.002, 9: 0.001, 10: 0.0, 'm': -0.311, 'b': 0.47799999999999998}, '70': {1: 0.315, 2: 0.35599999999999998, 3: 0.16800000000000001, 4: 0.085999999999999993, 5: 0.043999999999999997, 6: 0.017999999999999999, 7: 0.0070000000000000001, 8: 0.0040000000000000001, 9: 0.002, 10: 0.0, 'm': -0.22900000000000001, 'b': 0.26100000000000001}, '11': {1: 0.59999999999999998, 2: 0.24299999999999999, 3: 0.097000000000000003, 4: 0.036999999999999998, 5: 0.014, 6: 0.0060000000000000001, 7: 0.002, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.29599999999999999, 'b': 0.98799999999999999}, '12': {1: 0.67300000000000004, 2: 0.222, 3: 0.070999999999999994, 4: 0.023, 5: 0.0070000000000000001, 6: 0.002, 7: 0.001, 8: 0.0, 9: 0.0, 10: 0.0, 'm': -0.16300000000000001, 'b': 1.014}, '32': {1: 0.68600000000000005, 2: 0.20399999999999999, 3: 0.072999999999999995, 4: 0.025000000000000001, 5: 0.0080000000000000002, 6: 0.002, 7: 0.001, 8: 0.0, 9: 0.0, 10: 0.0, 'm': -0.107, 'b': 0.83199999999999996}, '31': {1: 0.59399999999999997, 2: 0.23400000000000001, 3: 0.104, 4: 0.042000000000000003, 5: 0.017000000000000001, 6: 0.0060000000000000001, 7: 0.0030000000000000001, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.191, 'b': 0.69299999999999995}, '30': {1: 0.56599999999999995, 2: 0.22600000000000001, 3: 0.114, 4: 0.052999999999999999, 5: 0.023, 6: 0.01, 7: 0.0040000000000000001, 8: 0.002, 9: 0.001, 10: 0.0, 'm': -0.35799999999999998, 'b': 0.55900000000000005}, '51': {1: 0.73699999999999999, 2: 0.152, 3: 0.067000000000000004, 4: 0.027, 5: 0.010999999999999999, 6: 0.0040000000000000001, 7: 0.001, 8: 0.001, 9: 0.0, 10: 0.0, 'm': -0.27000000000000002, 'b': 0.47699999999999998}, '50': {1: 0.65400000000000003, 2: 0.185, 3: 0.088999999999999996, 4: 0.041000000000000002, 5: 0.017999999999999999, 6: 0.0080000000000000002, 7: 0.0030000000000000001, 8: 0.001, 9: 0.001, 10: 0.0, 'm': -0.37, 'b': 0.35499999999999998}, '52': {1: 0.73199999999999998, 2: 0.17699999999999999, 3: 0.059999999999999998, 4: 0.021000000000000001, 5: 0.0070000000000000001, 6: 0.002, 7: 0.001, 8: 0.0, 9: 0.0, 10: 0.0, 'm': -0.047, 'b': 0.76300000000000001}}

def getRunsInn(rpinn):
    runsinn = {0:   1/((rpinn*.761)+1),
               1:   (rpinn*(0.761**2))/(((rpinn*.761)+1)**2)
               }

    for i in range(2, 11):
        v = (rpinn*(0.761**2)*(((rpinn*.761) - 0.761 + 1)**(i-1)))/(((rpinn*.761)+1)**(i+1))
        runsinn[i] = v
    return runsinn


def getRunExp(rpinn, runsinn):
    runExp = {'10': runsinn
              }
    for i in range(0, 3):
        for j in range(1, 9):
            k = str(j) + str(i)
            if k == '10':   continue
            runExp[k] = {0: ((tangoRunExp[k]['m']*rpinn) + tangoRunExp[k]['b'])
                         }
            for r in range(1, 11):
                runExp[k][r] = ((1 - runExp[k][0])*tangoRunExp[k][r])
    return runExp

def getInnWinexp(runExp):
    ## Chance of home team winning with zero
    ## outs at the beg. of each inning

    innWinexp = {'101': {0: 0.5
                      }
              }

    for i in range(-25, 0):
        innWinexp['101'][i] = 0
    for i in range(1, 26):
        innWinexp['101'][i] = 1

    for i in range(9, 0, -1):
        for j in range(2, 0, -1):
            if j == 2:  next = str(i+1) + '1'
            else:   next = str(i) + '2'
            this = str(i) + str(j)
            innWinexp[this] = {}
            if j == 2:
                for k in range(-25, 26):
                    p = 0
                    if i == 9 and k > 0:
                        innWinexp[this][k] = 1
                        continue
                    else:   pass
                    for m in range(0, 11):
                        if k+m > 25:    iw = 1
                        else:   iw = innWinexp[next][k+m]
                        p += runExp['10'][m]*iw
                    innWinexp[this][k] = p
            else:
                for k in range(-25, 26):
                    p = 0
                    for m in range(0, 11):
                        if k-m < -25:   iw = 0
                        else:   iw = innWinexp[next][k-m]
                        p += runExp['10'][m]*iw
                    innWinexp[this][k] = p
    return innWinexp


def getWinexp(innWinexp, runExp, inn, half, base, outs, rdiff):
    if inn > 9: inn = 9
    innkey = str(inn) + str(half)
    if outs > 2:    outs = 2
    sitkey = str(base) + str(outs)
    if half == 2:  next = str(inn+1) + '1'
    else:   next = str(inn) + '2'
    if sitkey == '10':  ## beginning of half inning
        if rdiff > 25:  rdiff = 25
        elif rdiff < -25:   rdiff = -25
        else:   pass
        Winexp = innWinexp[innkey][rdiff]
    elif half == 1:
        Winexp = 0
        for i in range(10, -1, -1):
            if rdiff-i < -25:   iw = 0
            elif rdiff-i > 25:  iw = 1
            else:   iw = innWinexp[next][rdiff-i]
            Winexp += runExp[sitkey][i]*iw
    else:
        Winexp = 0
        for i in range(0, 11):
            if rdiff-i < -25:   iw = 0
            elif rdiff+i > 25:   iw = 1
            else:   iw = innWinexp[next][rdiff+i]
            Winexp += runExp[sitkey][i]*iw
    return Winexp

def getVol(innWinexp, runExp, inn, half, base, outs, rdiff):
    ## changes if strikeout:
    if outs == 2:
        outsK = 0
        baseK = 1
        if half == 1:
            halfK = 2
            innK = inn
        else:
            halfK = 1
            innK = inn + 1
    else:
        outsK = outs + 1
        baseK, halfK, innK = base, half, inn
    WinexpK = getWinexp(innWinexp, runExp, innK, halfK, baseK, outsK, rdiff)
    ## changes if homerun
    if half == 1:
        rdiff -= baseHr[base]
    else:
        rdiff += baseHr[base]
    base = 1
    WinexpHr = getWinexp(innWinexp, runExp, inn, half, base, outs, rdiff)
    return (abs(WinexpHr - WinexpK))/0.133

def rpgToInnWinexp(rpg):
    rpinn = float(rpg)/9 ## r/inn
    runsinn = getRunsInn(rpinn)
    runExp = getRunExp(rpinn, runsinn)
    innWinexp = getInnWinexp(runExp)
    return innWinexp, runExp

def winnexp_feature(x):
    return getWinexp(innWinexp, runExp, x['inning'], x['halfInning_index'], x['base_state'], x['outs_beg'], x['run_diff'])

# Set up win expectancy variables
rpg = 4.5
innWinexp, runExp = rpgToInnWinexp(rpg)


def game_score_james(x):
    '''
    #     • Start with 50 points
    #     • Add 1 point for each out recorded (or 3 points per inning)
    #     • Add 2 points for each inning completed after the fourth
    #     • Add 1 additional point for every strikeout
    #     • Remove 2 points for each hit allowed
    #     • Remove 4 points for each earned run allowed
    #     • Remove 2 points for each unearned run allowed
    #     • Remove 1 point for each walk allowed
    '''
    score = 50
    score += x['outsPitching']
    score += 2*(x['inningsPitched'] - 4)
    score += x['strikeOutsPitching']
    score -= 2*x['hitsPitching']
    score -= 4*x['earnedRuns']
    score -= 2*(x['runsPitching'] - x['earnedRuns'])
    score -= (x['baseOnBallsPitching']+x['hitByPitchPitching'])
#     score = 50 + x['outsPitching'] + 2*(x['inningsPitched'] - 4) + x['strikeOutsPitching'] - 2*x['hitsPitching'] - 4*x['earnedRuns'] - 2*(x['runsPitching'] - x['earnedRuns']) - (x['baseOnBallsPitching']+x['hitByPitchPitching'])
    return score


def game_score_tango(x):
    '''
    Game Score formula (updated by Tom Tango)
    # • Start with 40 points
    # • Add 2 points for each out recorded (or 6 points per inning)
    # • Add 1 additional point for every strikeout
    # • Remove 2 points for each walk allowed
    # • Remove 2 points for each hit allowed
    # • Remove 3 points for each run allowed (earned or unearned)
    # • Remove 6 additional points for each home run allowed
    '''
    score = 40
    score += 2*x['outsPitching']
    score += x['strikeOutsPitching']
    score -= 2*(x['baseOnBallsPitching']+x['hitByPitchPitching'])
    score -= 2*x['hitsPitching']
    score -= 3*x['runsPitching']
    score -= 6*x['homeRunsPitching']
    return score

##################################################################################################
## Functions for extracting and matching ejected player names and getting their playerId
##################################################################################################
# Need to map names to the players.csv or playerBoxScores playerIds
def find_closest_playerName(playerName, players):
    players['fuzz_score'] = [fuzz.WRatio(playerName, x) for x in players['playerName']]
    best_match = players.loc[players['fuzz_score']==players['fuzz_score'].max(), 'playerName'].iloc[0]

    return best_match

def find_playerId(x, players, rosters_players):
    # rosters_players is a merge of the rosters df and the players df on the playerId
    tmp = players[players['playerName']==x['playerName']]
    if tmp.shape[0]==1:
        return tmp['playerId'].iloc[0]
    else:
        # If there are two players with the same name in players, then use the daily roster data to find the player on the matching team
        return rosters_players.loc[(rosters_players['dailyDataDate']==x['dailyDataDate']) & (rosters_players['teamId']==x['teamId']) & (rosters_players['playerName']==x['playerName']), 'playerId'].iloc[0]

##################################################################################################



train = pd.read_csv("./data/train.csv")

print(time.perf_counter())

eng = pkl.load(open("./data/train_nextDayPlayerEngagement.pickle","rb"))
games = pkl.load(open("./data/train_games.pickle","rb"))
rosters = pkl.load(open("./data/train_rosters.pickle","rb"))
p_box_scores = pkl.load(open("./data/train_playerBoxScores.pickle","rb"))
t_box_scores = pkl.load(open("./data/train_teamBoxScores.pickle","rb"))
transactions = pkl.load(open("./data/train_transactions.pickle","rb"))
standings = pkl.load(open("./data/train_standings.pickle","rb"))
awards = pkl.load(open("./data/train_awards.pickle","rb"))
events = pkl.load(open("./data/train_events.pickle","rb"))
p_twitter = pkl.load(open("./data/train_playerTwitterFollowers.pickle","rb"))
t_twitter = pkl.load(open("./data/train_teamTwitterFollowers.pickle","rb"))

teams = pd.read_csv("./data/teams.csv")
players = pd.read_csv("./data/players.csv")

schedule_21 = pd.read_csv("./data/schedule_2021.csv")
schedule_21['gameDate'] = pd.to_datetime(schedule_21['gameDate'])
schedule_21 = schedule_21.sort_values('gameDate')

##################################################################################################
## Regex's for extracting and matching ejected player names and getting their playerId
##################################################################################################
team_names = list(teams['teamName'].unique()) + ["Diamondbacks"]
team_regex = re.compile('|'.join(map(re.escape, team_names)))
team_full_names = list(teams['name'].unique()) + list(teams['teamName'].unique())
team_full_regex = re.compile('|'.join(map(re.escape, team_full_names)))
coaching_names = ["Assistant Hitting Coach", "Manager", "Bench Coach", "Interim Manager", "Hitting Coach", "First Base Coach", "Pitching Coach", "bench caoch", "assistant hitting coach", "Third Base Coach", "catching coach", "field coordinator", "first base coach", "hitting coach", "major league coach", "manager", "pitching coach", "third base coach", "bench coach"]
coaching_regex = re.compile('|'.join(map(re.escape, coaching_names)))
positions = ['pitcher','catcher','first baseman','second baseman', 'third baseman','shortstop','left fielder','center fielder','right fielder', 'designated hitter']
pos_regex = re.compile('|'.join(map(re.escape, positions)))
##################################################################################################

train_colnames = ['nextDayPlayerEngagement',
            'games',
            'rosters',
            'playerBoxScores',
            'teamBoxScores',
            'transactions',
            'standings',
            'awards',
            'events',
            'playerTwitterFollowers',
            'teamTwitterFollowers']

hitter_history_feats = ['hits','doubles','triples','homeRuns','rbi','totalBases', 'plateAppearances','strikeOuts','baseOnBalls','hitByPitch', 'atBats','sacFlies']
pitcher_history_feats = ['gamesPlayedPitching', 'gamesStartedPitching','inningsPitched', 'pitchesThrown', 'winsPitching', 'runsPitching', 'homeRunsPitching', 'strikeOutsPitching','earnedRuns', 'blownSaves', 'holds']
fielder_history_feats = ['errors']
##################################################################################################
## Mappings
##################################################################################################
team_mapping = teams.set_index('teamName')['id'].to_dict()
team_mapping['Diamondbacks'] = 109

player_mapping = p_box_scores[['playerId','playerName']].drop_duplicates()

##################################################################################################

##################################################################################################
## Target aggregate features
##################################################################################################
player_aggs = eng.groupby('playerId')[['target1','target2','target3','target4']].agg({'target1': ['median','var'],
                                                                                        'target2': ['median','var'],
                                                                                        'target3': ['median','var'],
                                                                                        'target4': ['median','var']}).round(6)
player_aggs.columns = ["_".join(x) for x in player_aggs.columns.ravel()]
player_aggs = player_aggs.reset_index()

player_medians = eng.groupby('playerId')[['target1','target2','target3','target4']].median().round(6).reset_index()
player_medians = player_medians.rename({'target1': 'target1_p_median',
                                        'target2': 'target2_p_median',
                                        'target3': 'target3_p_median',
                                        'target4': 'target4_p_median'}, axis=1)

player_variances = eng.groupby('playerId')[['target1','target2','target3','target4']].var().round(6).reset_index()
player_variances = player_variances.rename({'target1': 'target1_p_var',
                                        'target2': 'target2_p_var',
                                        'target3': 'target3_p_var',
                                        'target4': 'target4_p_var'}, axis=1)

player_means = eng.groupby('playerId')[['target1','target2','target3','target4']].mean().round(6).reset_index()
player_means = player_means.rename({'target1': 'target1_p_mean',
                                        'target2': 'target2_p_mean',
                                        'target3': 'target3_p_mean',
                                        'target4': 'target4_p_mean'}, axis=1)

game_day_player_means = eng.merge(p_box_scores[['dailyDataDate','playerId']])
game_day_player_means = game_day_player_means.groupby('playerId')[['target1','target2','target3','target4']].mean().round(6).reset_index()


game_day_player_means = eng.merge(p_box_scores[['dailyDataDate','playerId', 'gamePk']], how='left')
game_day_player_means['game_played'] = game_day_player_means['gamePk'].notnull().astype(int)
off_day_player_means = game_day_player_means[game_day_player_means['game_played']==0].groupby(['playerId'])[['target1','target2','target3','target4']].mean().round(6).reset_index()
off_day_player_means = off_day_player_means.rename({'target1': 'target1_p_mean_off_day',
                                        'target2': 'target2_p_mean_off_day',
                                        'target3': 'target3_p_mean_off_day',
                                        'target4': 'target4_p_mean_off_day'}, axis=1)

game_day_player_means = game_day_player_means[game_day_player_means['game_played']==1].groupby(['playerId'])[['target1','target2','target3','target4']].mean().round(6).reset_index()
game_day_player_means = game_day_player_means.rename({'target1': 'target1_p_mean_game_day',
                                        'target2': 'target2_p_mean_game_day',
                                        'target3': 'target3_p_mean_game_day',
                                        'target4': 'target4_p_mean_game_day'}, axis=1)

game_day_player_vars = eng.merge(p_box_scores[['dailyDataDate','playerId']])
game_day_player_vars = game_day_player_vars.groupby('playerId')[['target1','target2','target3','target4']].var().round(6).reset_index()
game_day_player_vars = game_day_player_vars.rename({'target1': 'target1_p_var_game_day',
                                        'target2': 'target2_p_var_game_day',
                                        'target3': 'target3_p_var_game_day',
                                        'target4': 'target4_p_var_game_day'}, axis=1)

##################################################################################################
position_freq = p_box_scores['positionType'].fillna(-999).value_counts(normalize=True).to_dict()

position_target_agg = eng.merge(p_box_scores[['dailyDataDate','playerId','gamePk','gameTimeUTC','positionType']], how='left')
dh_games = position_target_agg[position_target_agg[['dailyDataDate','playerId']].duplicated(keep=False)].sort_values('gameTimeUTC')[['dailyDataDate','playerId','gamePk']].reset_index(drop=True)
dh_last_game = dh_games[dh_games[['dailyDataDate','playerId']].duplicated(keep='first')] #games to remove
position_target_agg = position_target_agg[~(position_target_agg['playerId'].isin(dh_last_game['playerId']) & position_target_agg['gamePk'].isin(dh_last_game['gamePk']))]
position_freq = position_target_agg['positionType'].fillna(-999).value_counts(normalize=True).to_dict()
position_target_agg = position_target_agg.groupby('positionType')[['target1','target2','target3','target4']].agg({'target1': ['median','var'],
                                                                                        'target2': ['median','var'],
                                                                                        'target3': ['median','var'],
                                                                                        'target4': ['median','var']}).round(6)
position_target_agg.columns = ["_".join(x + ('position',)) for x in position_target_agg.columns.ravel()]
position_target_agg = position_target_agg.reset_index()


pitchers = players[players['primaryPositionName']=="Pitcher"]
players['playerForTestSetAndFuturePreds'] = players['playerForTestSetAndFuturePreds'].fillna(-1).astype(int)
players['value'] = 1
player_country_dummies = pd.pivot_table(players, values='value', index=['playerId'], columns=['birthCountry'], aggfunc='sum', fill_value=0).reset_index()


games['gameDate'] = pd.to_datetime(games['gameDate'])
games = games.sort_values('gameDate')
schedule = pd.concat([games[['dailyDataDate', 'gamePk','homeId', 'gameDate']].rename({'homeId': 'teamId'}, axis=1),
                      games[['dailyDataDate', 'gamePk','awayId', 'gameDate']].rename({'awayId': 'teamId'}, axis=1)])
schedule = schedule.sort_values('gameDate')
schedule = schedule[schedule['dailyDataDate']<20210401]
schedule = pd.concat([schedule, schedule_21[['dailyDataDate','teamId','gameDate']]])
schedule['gameDate'] = pd.to_datetime(schedule['gameDate'])




player_history = defaultdict(list)
# List of daily training observation frames to be concatenated
t = []
prior_day_pos_player_pitching = pd.DataFrame()
prior_day_rosters = pd.DataFrame()
team_win_history = {}
win_streaks = {}
hitter_history_dict = {}
fielder_history_dict = {}
pitcher_history_dict = {}
#pitcher_history_df = pd.DataFrame()
print(time.perf_counter())
for i, data in tqdm(train.iterrows()):

    data = data.to_frame().T
    daily_data_date = data['date'].iloc[0]
    season = int(str(daily_data_date)[:4])
    daily_data_nested_df_names = data.drop('date', axis = 1).columns.values.tolist()

    eng, games, rosters, p_box_scores, t_box_scores, transactions, standings, awards, events, p_twitter, t_twitter = get_unnested_data(data, train_colnames)
    eng_shape = eng.shape

    print(time.perf_counter())

    t_tmp = eng.copy()

    # If rosters data is missing then use rosters from the most recent day available
    if rosters.empty:
        rosters = prior_day_rosters
    prior_day_rosters = rosters.copy()


    if not games.empty and not games.loc[games['gameType'].isin(["R", "F","D","L","W","C","P"])].empty:

        schedule_day = pd.concat([games.loc[games['gameType'].isin(["R", "F","D","L","W","C","P"]) & ~games['detailedGameState'].isin(["Postponed"]),['dailyDataDate', 'gamePk','homeId', 'gameDate', 'gameTimeUTC', 'homeWinner']].rename({'homeId': 'teamId', 'homeWinner': 'winner'}, axis=1),
                          games.loc[games['gameType'].isin(["R", "F","D","L","W","C","P"]) & ~games['detailedGameState'].isin(["Postponed"]), ['dailyDataDate', 'gamePk','awayId', 'gameDate', 'gameTimeUTC','awayWinner']].rename({'awayId': 'teamId', 'awayWinner': 'winner'}, axis=1)])

        schedule_day = schedule_day.sort_values('gameTimeUTC')
        team_win_dict = schedule_day.groupby("teamId")['winner'].apply(list).to_dict()
        for k,v in team_win_dict.items():
            if not k in team_win_history:
                team_win_history[k] = v
            else:
                team_win_history[k].extend(v)

        win_streaks = {k: v[::-1].index(0) if 0 in v else len(v) for k, v in team_win_history.items()}


        print(time.perf_counter())
        if not schedule_day.empty:
            game_rosters = schedule_day.merge(rosters, how='left', on=['gameDate','teamId'])
            game_rosters = game_rosters[game_rosters['playerId'].notnull()] #missing roster for Nationals 20200910
            game_rosters['playerId'] = game_rosters['playerId'].astype(int)
            p_box_scores = p_box_scores.sort_values("gameTimeUTC")
            p_box_scores['gameDate'] = pd.to_datetime(p_box_scores['gameDate'])
            p_box_scores['season'] = p_box_scores['gameDate'].dt.year
            player_history_daily = game_rosters.merge(p_box_scores, how='left', on=['gamePk', 'playerId'])
            player_history_daily['gameTimeUTC_y'] = player_history_daily['gameTimeUTC_y'].fillna(player_history_daily['gameTimeUTC_x'])
            # NOTE: dailyDataDate==2020918 gamePk==631122 Start time of 2020-09-18T03:33:00Z is not accurate; that would imply the game started the day before at ~11:30PM local time
            player_history_daily = player_history_daily.sort_values(['playerId','gameTimeUTC_y']) # SORT BY gameTimeUTC from p_box_scores. `gameTimeUTC` is not accurate from the `games` data
            player_history_daily[hitter_history_feats] = player_history_daily[hitter_history_feats].fillna(0)


            print(time.perf_counter())

            hitter_history_tmp = {n: grp.to_dict('list') for n, grp in player_history_daily[hitter_history_feats + ['season', 'playerId']].groupby('playerId')}
            for k,v in hitter_history_tmp.items():
                if not k in hitter_history_dict:
                    hitter_history_dict[k] = v
                else:
                    for feat in hitter_history_feats + ['season']:
                        hitter_history_dict[k][feat].extend(v[feat])
            # For hitters, only use games they played in. Pitchers need off days filled in because it's important to account for rest/off days
            # Fill in days with 0 if hitter isn't in daily box scores
            # for k,v in hitter_history_dict.items():
            #     if not k in hitter_history_tmp:
            #         for feat in hitter_history_feats + ['season']:
            #             hitter_history_dict[k][feat].append(season if feat=='season' else 0.0)
            fielder_history_tmp = {n: grp.to_dict('list') for n, grp in player_history_daily[fielder_history_feats + ['season', 'playerId']].groupby('playerId')}
            for k,v in fielder_history_tmp.items():
                if not k in fielder_history_dict:
                    fielder_history_dict[k] = v
                else:
                    for feat in fielder_history_feats + ['season']:
                        fielder_history_dict[k][feat].extend(v[feat])

            pitcher_history_tmp = {n: grp.to_dict('list') for n, grp in p_box_scores.loc[p_box_scores['positionName']=='Pitcher', pitcher_history_feats + ['season', 'playerId']].groupby('playerId')}
            for k,v in pitcher_history_tmp.items():
                if not k in pitcher_history_dict:
                    pitcher_history_dict[k] = v
                else:
                    for feat in pitcher_history_feats + ['season']:
                        pitcher_history_dict[k][feat].extend(v[feat])
            # Fill in days with 0 if pitcher isn't in daily box scores
            for k,v in pitcher_history_dict.items():
                if not k in pitcher_history_tmp:
                    for feat in pitcher_history_feats + ['season']:
                        pitcher_history_dict[k][feat].append(season if feat=='season' else 0.0)


    print(time.perf_counter())
    # History features
    days_of_history = list(range(2,21)) #also could be games_of_history depending how its used
    max_days_of_history = np.max(days_of_history)
    hitting_history_features = {}
    pitching_history_features = {}
    fielding_history_features = {}

    for k, v in hitter_history_dict.items():
        # only need to include players in the current eng
        hitting_history_features[k] = {}
        hitting_history_features[k]['hit_streak'] =  v['hits'][::-1].index(0) if 0 in v['hits'] else len(v['hits'])
        for feat in hitter_history_feats:
            d = hitter_history_dict[k][feat]
            hitting_history_features[k][f'{feat}_season'] = sum([f for seas, f in zip(hitter_history_dict[k]['season'], d) if seas==season])
            if feat not in ['sacFlies','atBats']:
                d_padded = np.zeros(max_days_of_history)
                d_padded[:np.minimum(max_days_of_history, len(d))] = d[-np.minimum(max_days_of_history, len(d)):][::-1]
                # d_padded = np.pad(d[-days_of_history:], (np.maximum(0, days_of_history-len(d)+1), 0))[::-1]
                d_cumsum = nb_cumsum(d_padded)
                for day in days_of_history:
                    hitting_history_features[k][f'{feat}_last{day}'] = d_cumsum[day-1]
    #                 hitting_history_features[k][f'{feat}_{day-1}_games_ago'] = d_padded[day-1]

    hitting_history_df = pd.DataFrame.from_dict(hitting_history_features, orient='index').reset_index().rename({'index': 'playerId'}, axis=1)
    if 'homeRuns_season' in hitting_history_df.columns:
        hitting_history_df['homeRuns_rank'] = hitting_history_df['homeRuns_season'].rank(method='min', ascending=False)
        hitting_history_df['BA'] = hitting_history_df['hits_season'] / hitting_history_df['atBats_season']
        hitting_history_df['OBP'] = hitting_history_df[['hits_season','baseOnBalls_season', 'hitByPitch_season']].sum(axis=1) / hitting_history_df[['atBats_season','baseOnBalls_season', 'hitByPitch_season', 'sacFlies_season']].sum(axis=1)
        hitting_history_df['SLG'] = ((hitting_history_df['hits_season'] - hitting_history_df[['doubles_season','triples_season','homeRuns_season']].sum(axis=1)) + 2*hitting_history_df['doubles_season'] + 3*hitting_history_df['triples_season'] + 4*hitting_history_df['homeRuns_season'])/ hitting_history_df['atBats_season']

    for k, v in fielder_history_dict.items():
        # only need to include players in the current eng
        fielding_history_features[k] = {}
        for feat in fielder_history_feats:
            d = fielder_history_dict[k][feat]
            d_padded = np.zeros(max_days_of_history)
            d_padded[:np.minimum(max_days_of_history, len(d))] = d[-np.minimum(max_days_of_history, len(d)):][::-1]
            # d_padded = np.pad(d[-days_of_history:], (np.maximum(0, days_of_history-len(d)+1), 0))[::-1]
            d_cumsum = nb_cumsum(d_padded)
            for day in days_of_history:
#                 fielding_history_features[k][f'{feat}_last{day}'] = d_cumsum[day-1]
                fielding_history_features[k][f'{feat}_{day-1}_games_ago'] = d_padded[day-1]

    fielding_history_df = pd.DataFrame.from_dict(fielding_history_features, orient='index').reset_index().rename({'index': 'playerId'}, axis=1)

    print(time.perf_counter())

    for k, v in pitcher_history_dict.items():
        # only need to include players in the current eng
        pitching_history_features[k] = {}
        season_starts = [starts  for seas, starts in zip(pitcher_history_dict[k]['season'], pitcher_history_dict[k]['gamesStartedPitching']) if seas==season]
        season_played = [played  for seas, played in zip(pitcher_history_dict[k]['season'], pitcher_history_dict[k]['gamesPlayedPitching']) if seas==season]
        pitching_history_features[k]['days_since_last_start'] = season_starts[::-1].index(1.0) if 1 in season_starts else len(season_starts)
        pitching_history_features[k]['days_since_last_played'] = season_played[::-1].index(1.0) if 1 in season_played else len(season_played)
        for feat in pitcher_history_feats:
            d = pitcher_history_dict[k][feat]
            pitching_history_features[k][f'{feat}_season'] = sum([f for seas, f in zip(pitcher_history_dict[k]['season'], d) if seas==season])
            d_padded = np.pad(d, (np.maximum(0, max_days_of_history-len(d)), 0))[::-1]
            d_cumsum = nb_cumsum(d_padded)
            for day in days_of_history:
                pitching_history_features[k][f'{feat}_last{day}'] = d_cumsum[day-1]
                pitching_history_features[k][f'{feat}_{day-1}_games_ago'] = d_padded[day-1]

    pitching_history_df = pd.DataFrame.from_dict(pitching_history_features, orient='index').reset_index().rename({'index': 'playerId'}, axis=1)


    if not p_twitter.empty:
        p_twitter_recent = p_twitter

    # How to handle doubleheaders? Taking stats from first game for now
    if not t_box_scores.empty:
        # THIS IS STILL NOT VERY SMART. IF A PLAYER ONLY PLAYS IN 1 OF THE GAMES, THE GAME THEY PLAYED IN MIGHT NOT GET PICKED
        # There is a double header on 2018-04-17 but the `doubleHeader` flag is only True for one of them and `gameNumber` is 1 for both (should be 1 and 2).
        # There might be more double headers that aren't correctly indicated in the `games` data
        t_tmp = t_tmp.merge(p_box_scores, how='left', on=['dailyDataDate', 'playerId'])
        dh_games = t_tmp[t_tmp[['dailyDataDate','playerId']].duplicated(keep=False)].sort_values('gameTimeUTC')[['dailyDataDate','playerId','gamePk']].reset_index(drop=True)
        dh_last_game = dh_games[dh_games[['dailyDataDate','playerId']].duplicated(keep='first')] #games to remove
        t_tmp = t_tmp[~(t_tmp['playerId'].isin(dh_last_game['playerId']) & t_tmp['gamePk'].isin(dh_last_game['gamePk']))]

        t_tmp['game_score_james'] = game_score_james(p_box_scores)
        t_tmp['game_score_tango'] = game_score_tango(p_box_scores)
        t_tmp['position_player_pitching'] = ((t_tmp['positionCode']>1) & (t_tmp['pitchesThrown']>0)).astype(int)
        t_tmp['pitcher_hit_home_run'] = ((t_tmp['positionCode']==1) & (t_tmp['homeRuns'] > 0)).astype(int)
        # t_tmp['pos_player_pitched_prior_day'] = 0
        # if not prior_day_pos_player_pitching.empty:
        #     t_tmp['pos_player_pitched_prior_day'] = t_tmp['playerId'].map(dict(zip(prior_day_pos_player_pitching.playerId, prior_day_pos_player_pitching.position_player_pitching)))
        # prior_day_pos_player_pitching = t_tmp.loc[t_tmp['position_player_pitching']==1, ['playerId','position_player_pitching']].fillna(0)
        t_tmp['no_hitter'] = ((t_tmp['inningsPitched']>=9) & (t_tmp['hitsPitching']==0)).astype(int)
        t_tmp['no_hitter_league'] = t_tmp['no_hitter'].max()

        t_tmp['position_player_pitching_league'] = t_tmp['position_player_pitching'].max()
        t_tmp['game_hour'] = (pd.to_datetime(t_tmp['gameTimeUTC']) + pd.Timedelta(hours=-5)).dt.hour
        t_tmp = t_tmp.merge(t_box_scores, how='left', on=['gamePk', 'teamId'], suffixes=['','_team_box_score'])

        t_tmp['positionType_freq'] = t_tmp['positionType'].fillna(-999).map(position_freq)
        if 'positionType' in t_tmp.columns:
            t_tmp = t_tmp.merge(position_target_agg, how='left', on='positionType')
        assert t_tmp.shape[0]==eng_shape[0], "t_tmp length does not match engagement frame length, check for duplicated data"


    print(time.perf_counter())
    # Did player have a walk-off hit/home run?
    if not events.empty:
        events  = events.sort_values(['gamePk','inning','halfInning', 'atBatIndex', 'eventId'], ascending=[True, True, False, True, True])
        last_play = events.groupby('gamePk').tail(1)
        # filter out top of inning because one game was ended after the top of the inning
        walk_offs = last_play[(last_play['halfInning']=='bottom') & (last_play['rbi']>0)][['dailyDataDate','hitterId', 'pitcherId','rbi', 'event']]
        walk_offs.columns = ['dailyDataDate','hitterId', 'pitcherId','walk_off_rbi', 'walk_off_hr']
        walk_offs['walk_off_hr']  =  (walk_offs['walk_off_hr'].isin(['Home Run'])).astype(int)
        t_tmp = t_tmp.merge(walk_offs[['dailyDataDate','hitterId', 'walk_off_hr', 'walk_off_rbi']].rename({'hitterId': 'playerId'}, axis=1), how='left', on=['dailyDataDate','playerId'])
        t_tmp = t_tmp.merge(walk_offs[['dailyDataDate','pitcherId', 'walk_off_hr', 'walk_off_rbi']].rename({'pitcherId': 'playerId'}, axis=1), how='left', on=['dailyDataDate','playerId'], suffixes=["","_pitcher"])
        t_tmp[['walk_off_rbi', 'walk_off_hr', 'walk_off_hr_pitcher','walk_off_rbi_pitcher']] = t_tmp[['walk_off_rbi', 'walk_off_hr', 'walk_off_hr_pitcher','walk_off_rbi_pitcher']].fillna(0)
        t_tmp['walk_off_league'] = t_tmp['walk_off_rbi'].max()

        hr_dist = events[events['event']=='Home Run'].groupby('hitterId')['totalDistance'].max().reset_index()
        hr_launchSpeed = events[events['event']=='Home Run'].groupby('hitterId')['launchSpeed'].max().reset_index()
        t_tmp = t_tmp.merge(hr_dist.rename({'hitterId': 'playerId'}, axis=1), how='left', on='playerId')
        t_tmp = t_tmp.merge(hr_launchSpeed.rename({'hitterId': 'playerId'}, axis=1), how='left', on='playerId')

        # How long did a starting pitcher go without a hit? (Did they start picking up potential no-hitter hype?)
        starters = events[events['isStarter']==1].reset_index(drop=True)
        starters['hit'] = starters['event'].isin(['Single','Double', 'Triple', 'Home Run']).astype(int)
        starters['hits_cumsum'] = starters.groupby('pitcherId')['hit'].cumsum()
        starters_first_hit_inning = starters[starters['hits_cumsum']==1].groupby('pitcherId').first()[['inning','outs']].reset_index()
        starters_first_hit_inning['inning'] = starters_first_hit_inning['inning'] + starters_first_hit_inning['outs']/10
        starters_first_hit_inning = starters_first_hit_inning.rename({'inning': 'pitcher_first_hit_inning'}, axis=1)
        t_tmp = t_tmp.merge(starters_first_hit_inning[['pitcherId', 'pitcher_first_hit_inning']], how='left', left_on='playerId', right_on='pitcherId')

        # How long did a starting pitcher have a perfect game going?
        starters_first_mob_inning = starters[~starters['menOnBase'].isin([None,"Empty"])]
        starters_first_mob_inning = starters_first_mob_inning.groupby('pitcherId').first()[['inning','outs']].reset_index()
        starters_first_mob_inning['inning'] = starters_first_mob_inning['inning'] + starters_first_mob_inning['outs']/10
        starters_first_mob_inning = starters_first_mob_inning.rename({'inning': 'pitcher_first_mob_inning'}, axis=1)
        t_tmp = t_tmp.merge(starters_first_mob_inning[['pitcherId', 'pitcher_first_mob_inning']], how='left', left_on='playerId', right_on='pitcherId')

        # Pitch features
        nastyFactor_features = events[events['type']=='pitch'].groupby("pitcherId")['nastyFactor'].agg(['mean','median','min','max']).reset_index().rename(columns={f: f'nastyFactor_{f}' for f in ['mean','median','max','min']}).rename(columns={'pitcherId': 'playerId'})
        t_tmp = t_tmp.merge(nastyFactor_features, how='left', on='playerId')

        # Possible robbed HRs
        # NEED TO IMPROVE NAME EXTRACTION
        # robbed = events[events['event'].isin(['Pop Out','Lineout','Sac Fly','Flyout']) & (events['totalDistance']>400)]
        # robbed['value'] = 1
        # robbed['fielderName'] = robbed['description'].apply(lambda x: re.search('[left|center|right] fielder (.+?)\.', x).group(1)).tolist()
        # robbed['fielderId'] = robbed['fielderName'].map(dict(zip(player_mapping.playerName, player_mapping.playerId)))
        # t_tmp['robbed_hitter'] = t_tmp['hitterId'].map(dict(zip(robbed.hitterId, robbed.value))).fillna(0)
        # t_tmp['robbed_fielder'] = t_tmp['hitterId'].map(dict(zip(robbed[robbed['fielderId'].notnull()].fielderId, robbed[robbed['fielderId'].notnull()].value))).fillna(0)

        # Calculate player Win Probability Added
        #need to get assign 100% WPA to winning team to assign WPA scores to correct player/team

        player_wpa = pd.Series(dtype=float)
        for gamePk, game in events.groupby('gamePk'):
            game = game.reset_index(drop=True)
            game['run_diff'] = game['homeScore'] - game['awayScore']
            game['halfInning_index'] = game['halfInning'].map({'top':1, 'bottom': 2})
            game['base_state'] = game['menOnBase'].map({None: np.nan, 'Empty': 1, 'Men_On': 2, 'RISP': 3, 'Loaded': 8})
            game['base_state'] = game['base_state'].ffill().fillna(1).astype(int)
            game['outs_beg'] = np.maximum(game['outs'] - 1, 0)
            game['win_exp'] = game.apply(winnexp_feature, axis=1)
            game['win_exp_lag'] = game['win_exp'].shift(-1)
            game.loc[game.shape[0]-1, 'win_exp_lag'] = 1 if game.loc[game.shape[0]-1, 'homeScore']>game.loc[game.shape[0]-1, 'awayScore'] else 0
            game['win_exp_delta'] = game['win_exp_lag'] - game['win_exp']
            # Increases in the top of the inning are assigned to the pitcher
            # Increases in the bottom of the inning are assigned to the hitter
            pitcher_wpa_top = game.loc[(game['halfInning']=='top') & (game['win_exp_delta']>0),['pitcherId','win_exp_delta']].groupby('pitcherId')['win_exp_delta'].sum()
            hitter_wpa_top = game.loc[(game['halfInning']=='top') & (game['win_exp_delta']>0),['hitterId','win_exp_delta']].groupby('hitterId')['win_exp_delta'].sum()
            hitter_wpa_top = -hitter_wpa_top

            pitcher_wpa_bot = game.loc[(game['halfInning']=='bottom') & (game['win_exp_delta']>0),['pitcherId','win_exp_delta']].groupby('pitcherId')['win_exp_delta'].sum()
            hitter_wpa_bot = game.loc[(game['halfInning']=='bottom') & (game['win_exp_delta']>0),['hitterId','win_exp_delta']].groupby('hitterId')['win_exp_delta'].sum()
            pitcher_wpa_bot = -pitcher_wpa_bot

            player_wpa = player_wpa.add(pitcher_wpa_top, fill_value=0)
            player_wpa = player_wpa.add(hitter_wpa_top, fill_value=0)
            player_wpa = player_wpa.add(pitcher_wpa_bot, fill_value=0)
            player_wpa = player_wpa.add(hitter_wpa_bot, fill_value=0)

        player_wpa = player_wpa.reset_index()
        player_wpa = player_wpa.rename({"index": "playerId", 0: "wpa"}, axis=1)

        t_tmp = t_tmp.merge(player_wpa, how='left', on='playerId')
        t_tmp['wpa_daily_max'] = t_tmp['wpa'].max()

        print(time.perf_counter())

        # get ejections
        ejections = events.loc[events['event']=="Ejection", ['dailyDataDate','description']].reset_index(drop=True)
        if not ejections.empty:
            ejections['description'] = [x.split(" ejected by")[0] for x in ejections['description']]
            # Get team; needed for coach_ejected feature
            ejections['teamName'] = [team_regex.findall(x)[0] if team_regex.findall(x) else None for x in ejections['description']] # else None to account for names not spelled in a way that matches the regex
            ejections['teamId'] = ejections['teamName'].map(team_mapping)
            ejections['coach_ejected'] = [1 if coaching_regex.search(x) else 0 for x in ejections['description']]
            ejections['player_ejected'] = 1 - ejections['coach_ejected']
            # Get player name
            ejections['playerName'] = [team_full_regex.sub("", ' '.join(x.split())) for x in ejections['description']]
            ejections['playerName'] = [coaching_regex.sub("", ' '.join(x.split())) for x in ejections['playerName']]
            ejections['playerName'] = [pos_regex.sub("", ' '.join(x.split())).strip() for x in ejections['playerName']]
            # If there is no match for a player use fuzzywuzzy to find the closest match
            ejections.loc[(ejections['player_ejected']==1), 'playerName'] = ejections.loc[(ejections['player_ejected']==1), 'playerName'].apply(lambda x: find_closest_playerName(x, players))
            ejections.loc[(ejections['player_ejected']==1), 'playerId'] = ejections.loc[(ejections['player_ejected']==1)].apply(lambda x: find_playerId(x, players, rosters), axis=1)

            t_tmp = t_tmp.merge(ejections.groupby('teamId')['coach_ejected'].sum().reset_index(), how='left', on='teamId')
            t_tmp['coach_ejected'] = t_tmp['coach_ejected'].fillna(0)
            t_tmp = t_tmp.merge(ejections.loc[ejections['player_ejected']==1,['playerId','player_ejected']], how='left', on='playerId')
            t_tmp['player_ejected'] = t_tmp['player_ejected'].fillna(0)
        else:
            t_tmp['coach_ejected'] = 0
            t_tmp['player_ejected'] = 0

        print(time.perf_counter())
        assert t_tmp.shape[0]==eng_shape[0], "t_tmp length does not match engagement frame length, check for duplicated data"

    # if 'teamId' not in t_tmp.columns:
    #     t_tmp = t_tmp.merge(rosters[['playerId','teamId']], how='left', on='playerId')
    # t_tmp = t_tmp.merge(all_dates[['dailyDataDate_lead','teamId','nextDayGame']], how='left', left_on=['dailyDataDate', 'teamId'], right_on=['dailyDataDate_lead','teamId'])
    # t_tmp['nextDayGame'] = t_tmp['nextDayGame'].fillna(0)
    roster_dummies = pd.concat([rosters[['dailyDataDate','playerId']], pd.get_dummies(rosters['statusCode'])], axis=1)
    roster_dummies = roster_dummies.groupby(["dailyDataDate", "playerId"]).sum().reset_index()
    for col in ['A', 'BRV', 'D10', 'D60', 'D7', 'DEC','FME', 'PL', 'RES', 'RM', 'SU']:
        if col not in roster_dummies.columns:
            roster_dummies[col] = 0

    t_tmp = t_tmp.merge(roster_dummies, how='left', on=['dailyDataDate','playerId'])
    assert t_tmp.shape[0]==eng_shape[0], "rosters: t_tmp length does not match engagement frame length, check for duplicated data"

    if not transactions.empty:
        transactions_dummies = pd.concat([transactions[['dailyDataDate','playerId']], pd.get_dummies(transactions['typeCode'])], axis=1)
        transactions_dummies = transactions_dummies.groupby(["dailyDataDate", "playerId"]).sum().reset_index()
        for col in ['ASG', 'CLW', 'CU', 'DES', 'DFA', 'NUM','OPT', 'OUT', 'REL', 'RET', 'RTN', 'SC', 'SE', 'SFA', 'SGN', 'TR']:
            if col not in transactions_dummies.columns:
                transactions_dummies[col] = 0
        t_tmp = t_tmp.merge(transactions_dummies, how='left', on=['dailyDataDate','playerId'])
    else:
        t_tmp[['ASG', 'CLW', 'CU', 'DES', 'DFA', 'NUM','OPT', 'OUT', 'REL', 'RET', 'RTN', 'SC', 'SE', 'SFA', 'SGN', 'TR']] = 0

    assert t_tmp.shape[0]==eng_shape[0], "transactions: t_tmp length does not match engagement frame length, check for duplicated data"

    keep_awards = ['NLPOW', 'ALPOW', 'NLROM', 'ALROM','NLPOM','ALPOM','NLRRELMON','ALRRELMON','ALPITOM','NLPITOM','MLBPLAYOW']
    if not awards.empty:
        awards_filtered = awards[awards['awardId'].isin(keep_awards)].reset_index(drop=True)
        if not awards_filtered.empty:
            awards_dummies = pd.concat([awards_filtered[['dailyDataDate','playerId']], pd.get_dummies(awards_filtered['awardId'])], axis=1)
            awards_dummies = awards_dummies.groupby(["dailyDataDate", "playerId"]).sum().reset_index()
            for col in  keep_awards:
                if col not in awards_dummies.columns:
                    awards_dummies[col] = 0
            t_tmp = t_tmp.merge(awards_dummies, how='left', on=['dailyDataDate','playerId'])
        else:
            t_tmp[keep_awards] = 0
    else:
        t_tmp[keep_awards] = 0

    assert t_tmp.shape[0]==eng_shape[0], "awards: t_tmp length does not match engagement frame length, check for duplicated data"

    if 'teamId' not in t_tmp.columns:
        t_tmp = t_tmp.merge(rosters[['playerId','teamId']], how='left', on='playerId')
    assert t_tmp.shape[0]==eng_shape[0], "teamId: t_tmp length does not match engagement frame length, check for duplicated data"

    if not standings.empty:
        standings = standings.replace("-",0.0)
        object_cols = standings.select_dtypes(exclude=['float']).columns
        standings[object_cols] = standings[object_cols].apply(pd.to_numeric, downcast='float', errors='coerce')
        bool_cols  = standings.select_dtypes(include=['boolean']).columns
        standings[bool_cols] = standings[bool_cols].astype(int)
        t_tmp = t_tmp.merge(standings, how='left', on=['teamId'], suffixes=['','_team_standings'])
        t_tmp['team_games_played'] = t_tmp['wins'] + t_tmp['losses']
        assert t_tmp.shape[0]==eng_shape[0], "standings: t_tmp length does not match engagement frame length, check for duplicated data"

    if len(win_streaks) > 0:
        t_tmp['team_win_streak'] = t_tmp['teamId'].map(win_streaks)
    if not hitting_history_df.empty:
        t_tmp = t_tmp.merge(hitting_history_df, how='left', on='playerId')
        t_tmp['hr_rank'] = t_tmp['homeRuns_season'].rank(ascending=False)
        assert t_tmp.shape[0]==eng_shape[0], "hitting_history_df: t_tmp length does not match engagement frame length, check for duplicated data"
    if not pitching_history_df.empty:
        t_tmp = t_tmp.merge(pitching_history_df, how='left', on='playerId')
        # Calculate ERA
        # there are no more standings after season end so team_games_played is no longer known
        if 'team_games_played' in t_tmp.columns:
            t_tmp['era'] = 9 * (t_tmp['earnedRuns_season']/ t_tmp['inningsPitched_season'])
            t_tmp['era_rank'] = t_tmp.loc[t_tmp['inningsPitched_season'] >= t_tmp['team_games_played'], 'era'].rank(method='min')
        assert t_tmp.shape[0]==eng_shape[0], "pitching_history_df: t_tmp length does not match engagement frame length, check for duplicated data"
    if not fielding_history_df.empty:
        t_tmp = t_tmp.merge(fielding_history_df, how='left', on='playerId')
        assert t_tmp.shape[0]==eng_shape[0], "fielding_history_df: t_tmp length does not match engagement frame length, check for duplicated data"



    t_tmp = t_tmp.merge(player_country_dummies, how='left', on='playerId')
    assert t_tmp.shape[0]==eng_shape[0], "player_country_dummies: t_tmp length does not match engagement frame length, check for duplicated data"
    # Add games features
    # if not games.empty and not p_box_scores.empty:
    #     games['dayNight'] = games['dayNight'].map({'day': 0, 'night': 1})
    #     games['homeWinner'] = games['homeWinner'].fillna(-1).astype(float)
    #     t_tmp = t_tmp.merge(games[['gamePk', 'dayNight','homeWinPct','awayWinPct','homeScore','awayScore','homeWinner']], how='left', on='gamePk')
    #     assert t_tmp.shape[0]==eng_shape[0], "games: t_tmp length does not match engagement frame length, check for duplicated data"
    # t_tmp = t_tmp.merge(players[['playerId','playerForTestSetAndFuturePreds']], how='left')


    # Add Twitter features
    t_tmp = t_tmp.merge(p_twitter_recent[['playerId','numberOfFollowers']], how='left', on=['playerId'])

    # Add median player engagement
    # t_tmp = t_tmp.merge(player_medians, how='left', on='playerId')
    t_tmp = t_tmp.merge(player_aggs, how='left', on='playerId')
    # t_tmp = t_tmp.merge(player_variances, how='left', on='playerId')
    # if 'gamePk' in t_tmp.columns:
    #     t_tmp['game_played'] = t_tmp['gamePk'].notnull().astype(int)
    # else:
    #     t_tmp['game_played'] = 0
    t_tmp = t_tmp.merge(game_day_player_means, how='left', on='playerId')
    # t_tmp = t_tmp.merge(off_day_player_means, how='left', on='playerId')
    t_tmp = t_tmp.merge(game_day_player_vars, how='left', on='playerId')
    # t_tmp = t_tmp.merge(recent_player_means, how='left', on='playerId')
    # Does the player have a game the next day?

    t_tmp['monthday'] = t_tmp['dailyDataDate'].astype(str).str[4:].astype(int)
    t_tmp['dayofweek'] = pd.to_datetime(t_tmp['engagementMetricsDate']).dt.dayofweek
    t_tmp['data_dayofmonth'] = t_tmp['dailyDataDate'].astype(str).str[6:].astype(int)
    t_tmp['eng_dayofmonth'] = pd.to_datetime(t_tmp['engagementMetricsDate']).dt.day


    assert t_tmp.shape[0]==eng_shape[0], "final check: t_tmp length does not match engagement frame length, check for duplicated data"
    # Some features to indicate if something important/exciting happened around the league
    # Something important in one game might decrease the engagement in other games

    t.append(t_tmp)

tf = pd.concat(t)

# tf.to_csv("./tf_saved.csv", index=False)
# tf.to_parquet("./tf_saved.parq")
# tf = pd.read_csv("./tf_saved.csv")
# tf = pd.read_parquet("./tf_saved.parq")


tf.groupby('dailyDataDate')['playerId'].size().max()


tf.season.isna().sum()  # too many blanks
# tf['season'] = tf.dailyDataDate//10000


#  Add more features
# fantasy rankings
rank_cols=['season', 'playerId', 'rank', 'adp']
ranks = pd.read_csv('data/fantasy_rankings.csv', usecols=rank_cols)
tf = tf.merge(ranks, how='left', on=['season', 'playerId'])


ranks[ranks.duplicated(['season', 'playerId'], keep=False)]


tf.groupby('dailyDataDate')['playerId'].size().max()

tf = tf.drop_duplicates(['season', 'playerId'])
                # bug in the rankings file: 6 duplicated player_seasons; no idea yet which one is right



# team static stats
stat_cols = ['season', 'teamId', 'Attn_avg', 'TV_revenue', '26-MAN PAYROLL']
team_stats = pd.read_csv('data/team_season_stats.csv', usecols=stat_cols, na_values="na")
team_stats['Attn_avg'] = pd.to_numeric(team_stats.Attn_avg.str.replace(",", "", regex=False))
team_stats['26-MAN PAYROLL'] = pd.to_numeric(team_stats['26-MAN PAYROLL'].str.replace("\$|\,", "", regex=True))
team_stats.dtypes

tf = tf.merge(team_stats, how='left', on=['season', 'teamId'])


# twitter
tweet_cols = ['playerId', 'f_actual', 'f_bin1',
       'f_bin2', 'f_bin3', 'f_bin4', 'f_bin5', 'tweet_count', 'mean_retes',
       'mean_likes']
tweet = pd.read_csv('data/twitter_stats_current.csv', usecols=tweet_cols)
tf = tf.merge(tweet, how='left', on=['playerId'])




# Frequency Encodings


#tf[['started_next_game_pred', 'relieved_next_game_pred']] = tf[['started_next_game_pred', 'relieved_next_game_pred']].fillna(0)
tr = tf[(tf['dailyDataDate']<20210401)]
val = tf[(tf['dailyDataDate']>=20210401)]


params = {
    'num_leaves': 73,
    'min_data_in_leaf': 10,
    'objective': 'regression_l1',
    'max_depth': -1,
    'learning_rate': 0.05,
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
    "nthread": 32,
    "random_state": 12345,
    'boost_from_average': False,
    'reg_sqrt': False, # Change to false for regression_l2
    "device_type": 'gpu'
}

excl_cols = ['hitterId','pitcherId','playerId','gamePk','dailyDataDate','engagementMetricsDate','target1','target2','target3','target4',
             "gameDate", "gameTimeUTC", "teamName", "playerName", "jerseyNum",
             "positionName", "positionType", "target1_p_mean", "target2_p_mean","target3_p_mean","target4_p_mean",
             "gameTimeUTC_team_box_score", "gameDate_team_box_score", "dailyDataDate_team_box_score","dailyDataDate_team_standings",
             "dailyDataDate_lead", 'playerForTestSetAndFuturePreds','hr_rank','team_games_played', 'dayNight','homeWinPct','awayWinPct',
             'homeScore','awayScore','homeWinner', 'pitcherId_x', 'pitcherId_y']
excl_cols_new = ['rank', 'adp', 'Attn_avg', 'TV_revenue', '26-MAN PAYROLL', 'f_actual',
       'f_bin1', 'f_bin2', 'f_bin3', 'f_bin4', 'f_bin5', 'tweet_count',
       'mean_retes', 'mean_likes']

use_cols = [col for col in tr.columns if col not in excl_cols+excl_cols_new]
use_cols_plus = [col for col in tr.columns if col not in excl_cols]

tf.columns[tf.columns.str.contains('_x')]

# excl_cols = [f"{feat}_last{n}" for feat in hitter_history_feats + pitcher_history_feats + fielder_history_feats for n in [6,8,9,11,12,13,14,15,16,17,18,19]]
# use_cols = [col for col in use_cols if col not in excl_cols]
# excl_cols = [f'{feat}_{n-1}_games_ago' for feat in hitter_history_feats + pitcher_history_feats + fielder_history_feats for n in [6,8,9,11,12,13,14,15,16,17,18,19]]
# use_cols = [col for col in use_cols if col not in excl_cols]
# use_cols = [col for col in use_cols if "chances_" not in col]
# use_cols = [col for col in use_cols if "assists_" not in col]
# use_cols = [col for col in use_cols if "putOuts_" not in col]
# use_cols = [col for col in use_cols if "triples_" not in col]
# use_cols = [col for col in use_cols if "doubles_" not in col]
# splits = split_generator(X['tfrecord'])
# val_meta = train.loc[splits[self.hparams.fold][1], meta_columns]
# train_meta = train.loc[splits[self.hparams.fold][0], meta_columns]

import matplotlib.pyplot as plt

model_dict = {}
for target in [
    # 'target1',
    'target2',
    # 'target3',
    # 'target4'
    ]:
    params['learning_rate'] = 0.001 #if target in ['target1', 'target4'] else 0.001

    if target in ['target2', 'target4']:
        use_cols = use_cols_plus

    lgb_dtrain = lgb.Dataset(tr[use_cols], tr[target], free_raw_data=False)
    lgb_dval = lgb.Dataset(val[use_cols], val[target], free_raw_data=False)

    np.random.seed(123456)
    lgb_model = lgb.train(params,
                    train_set = lgb_dtrain,
                    # folds=splits,
                    valid_sets = [lgb_dtrain, lgb_dval],
                    num_boost_round = 200000,
                    early_stopping_rounds = 2000,
                    verbose_eval=1000,
                    #eval_train_metric=True,
                    # show_stdv=True,
                    # return_cvbooster=True
                    )
    if not os.path.exists("./saved_new"):
        os.makedirs("./saved")
    lgb_model.save_model(f"./saved_new/lgb_{target}_dubs_trips_plus.txt")
    lgb.plot_importance(lgb_model, height=0.1, xlim=None, ylim=None,
        title=target, xlabel='Feature importance', ylabel='feature_builder',
        importance_type='split', max_num_features=40, figsize=(8,8))
    plt.savefig(f'./saved_new/features_{target}_plus.jpg', bbox_inches="tight")


    model_dict[target] = {}
    model_dict[target]['num_trees'] = lgb_model.num_trees()
    model_dict[target]['imp'] = pd.DataFrame({'features': lgb_model.feature_name(), 'gain': lgb_model.feature_importance()})
    model_dict[target]['best_score'] = lgb_model.best_score




for k,v in model_dict.items():
    # print(v['imp'].sort_values('gain')[-40:])
    print(v['best_score']['valid_1']['l1'])

lgb_preds_model = lgb.Booster(model_file = 'saved_new/lgb_target2_dubs_trips_plus.txt')
preds = lgb_preds_model.predict(val[use_cols])

val['preds_t2'] = preds
val.to_parquet('./saved_new/valpreds_plus.parq')


for target in ['target2',
    # 'target3','target4'
    ]:
    params['learning_rate'] = 0.001 #if target=='target1' else 0.001
    if target in ['target2', 'target4']:
        use_cols = use_cols_plus

    lgb_dtrain = lgb.Dataset(tf[use_cols], tf[target], free_raw_data=False)

    # lgb_dval = lgb.Dataset(val[use_cols], val[target], free_raw_data=False)

    np.random.seed(123456)
    lgb_model = lgb.train(params,
                    train_set = lgb_dtrain,
                    # folds=splits,
                    # valid_sets = [lgb_dtrain, lgb_dval],
                    num_boost_round = model_dict[target]['num_trees'],
                    # early_stopping_rounds = 1000,
                    # verbose_eval=100,
                    #eval_train_metric=True,
                    # show_stdv=True,
                    # return_cvbooster=True
                    )

    lgb_model.save_model(f"./saved_new/lgb_{target}_dubs_trips_plus_full.txt")

