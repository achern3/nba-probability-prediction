import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

TOTAL_TIME = 48 * 60 # 48 min * 60 s/min

def is_nan(value):
    # checks if dataframe entry is nan
    try:
        return np.isnan(value)
    except:
        return False

def get_time_remaining(quarter, times):
    # converts time to time remaining as integer (seconds)
    time_list = times.split(':')
    quarter_time = (int(quarter) - 1) * 12 * 60 # 12 min/quarter * 60 s/min
    elapsed = int(time_list[0]) * 60 + int(time_list[1]) + quarter_time
    return (TOTAL_TIME - elapsed)

def get_score_diff(scores):
    # converts score to integer difference
    score_list = scores.split(' - ')
    return (int(score_list[0]) - int(score_list[1]))

def get_classifier(df):
    # retain only the scoring plays
    df_scores = df[df['score'].notnull()].reset_index(drop=True)

    label_dict = {}
    for i in range(len(df_scores)):
        # check which team wins based on score in last non-empty entry of each game
        if ((i == len(df_scores) - 1) or
            (df_scores['uid'][i] != df_scores['uid'][i + 1])):
            score_list = df_scores['score'][i].split(' - ')
            # if first score greater than second, assign label 0, otherwise 1
            if (int(score_list[0]) > int(score_list[1])): # away team wins
                label_dict[df_scores['uid'][i]] = 0
            else: # home team wins
                label_dict[df_scores['uid'][i]] = 1

    # train logistic regression classifier
    X = np.zeros((len(df_scores), 2)) # design matrix
    y = np.zeros(len(df_scores)) # labels (0: away team wins, 1: home team wins)
    for i in range(len(df_scores)):
        # first feature is time remaining in seconds
        X[i, 0] = get_time_remaining(df['quarter'][i], df_scores['time'][i])
        # second feature is score difference
        X[i, 1] = get_score_diff(df_scores['score'][i])
        y[i] = label_dict[df_scores['uid'][i]]

    clf = LogisticRegression(n_jobs=-1) # instantiate logistic regression classifier
    clf.fit(X, y) # train on the data
    return clf

def main():
    # read in data and build classifier
    df = pd.read_csv('combined_output.csv')
    df = df.drop(df[df['quarter'] > 4].index).reset_index(drop=True)
    clf = get_classifier(df)

    # plot probabilities for first game in training data
    time_remaining_list = [] # keep track of time remaining to plot
    event_probs = [] # list of probabilities for each event below occurring
    i = 0 # count tracker
    first_game_uid = df['uid'][0]
    while(df['uid'][i] == first_game_uid): # only want first game
        if (not is_nan(df['description'][i]) and
            'scored' in df['description'][i].lower()):
            # if scored, predict probability based on new scores
            time_remaining = get_time_remaining(df['quarter'][i], df['time'][i])
            time_remaining_list.append(time_remaining)

            current_score = df['score'][i]
            score_diff = get_score_diff(current_score)

            prob = clf.predict_proba([time_remaining, score_diff])
            event_probs.append(prob[0][1]) # retain home team winning probability

        i += 1

    plt.plot(time_remaining_list, event_probs, 'k-')
    plt.title('Home Team Winning Probability')
    plt.xlabel('Time Remaining (s)')
    plt.ylabel('Winning Probability')
    plt.gca().invert_xaxis()
    plt.show()

if __name__ == '__main__':
    main()
