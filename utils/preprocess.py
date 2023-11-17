import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def cluster_MarkovChain_states(df_amount_path, num_grps, gaussian_marginal=False, absolute_tail=False, segregate_samples=False, tail_samples_pct=None):

    # assume that first column is time index
    df_amount_raw = pd.read_csv(df_amount_path)
    df_time = df_amount_raw[['time']]
    df_amount_data = df_amount_raw[df_amount_raw.columns[1:]]

    # random seed
    random_state=54321

    if segregate_samples:
        assert gaussian_marginal == True, "marginals have to be normalized"

        # compute samples in tail and body
        if absolute_tail:
            df_amount_data['max_amount'] = df_amount_data.abs().max(axis=1)
        else:
            df_amount_data['max_amount'] = df_amount_data.max(axis=1)

        df_amount_data.insert(0, "idx", df_amount_data.index)
        df_body = df_amount_data.loc[df_amount_data.max_amount <= 1.75]
        df_body = df_body.drop(columns=['max_amount'])        
        df_tail = df_amount_data.loc[df_amount_data.max_amount > 1.75]
        df_tail = df_tail.drop(columns=['max_amount'])

        # compute number of samples in tail and body
        n_clusters_tail = round(tail_samples_pct * num_grps)
        n_clusters_body = num_grps - n_clusters_tail

        # apply k-means
        kmeans_body = KMeans(n_clusters=n_clusters_body, random_state=random_state, n_init=20, verbose=1).fit(df_body[df_body.columns[1:]])
        kmeans_tail = KMeans(n_clusters=n_clusters_tail, random_state=random_state, n_init=20, verbose=1).fit(df_tail[df_tail.columns[1:]])

        # merge data frames
        df_body['state'] = kmeans_body.labels_
        df_tail['state'] = kmeans_tail.labels_ + n_clusters_body
        df_state = pd.concat([df_body, df_tail], axis = 0, ignore_index=True)

        # sort by idx
        df_state.sort_values(by=['idx'], inplace=True)

        # add time
        df_state['time'] = df_time
        df_state = df_state[['time','state']]

    else:

        # apply k-means directly
        kmeans = KMeans(n_clusters=num_grps, random_state=random_state, n_init=20, verbose=1).fit(df_amount_data)
        df_time['state'] = kmeans.labels_
        df_state = df_time
        
    return df_state