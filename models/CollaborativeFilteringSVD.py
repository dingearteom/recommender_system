from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class CollaborativeFilteringSVD:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, articles_supplementary_information=None):
        self.articles_supplementary_information = articles_supplementary_information

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, interactions_train):
        """

        :param interactions_train: dataFrame with index named personId and two columns contentId and eventStrength
        :return:
        Sets self.cf_predictions_df - matrix for making recommendations,
             self.interactions_train
        """
        assert isinstance(interactions_train, pd.DataFrame), "interactions_train must be of type DataFrame"
        assert interactions_train.index.name == 'personId', "Index of interations_train must be named personId"
        assert len(interactions_train.columns) == 2 and 'contentId' in set(interactions_train.columns) \
               and 'eventStrength' in set(interactions_train.columns), "interactions_train must have two columns: " \
                                                                       "contentId and eventStrength"




        self.interactions_train = interactions_train

        users_items_pivot_matrix_df = interactions_train.reset_index().pivot(index='personId',
                                                                  columns='contentId',
                                                                  values='eventStrength').fillna(0)
        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_ids = list(users_items_pivot_matrix_df.index)
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

        # The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 15
        # Performs matrix factorization of the original user item matrix
        # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)

        user_factors = np.dot(U, sigma)
        item_factors = Vt
        all_user_predicted_ratings = np.dot(user_factors, item_factors)
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                    all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        self.cf_predictions_df = pd.DataFrame(all_user_predicted_ratings_norm,
                                              columns=users_items_pivot_matrix_df.columns,
                                              index=users_ids).transpose()

    def recommend_items(self, user_id, topn=10, verbose=False):
        # Get and sort the user's predictions
        items_to_ignore = set(self.interactions_train.loc[user_id]['contentId'])
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
            .sort_values('recStrength', ascending=False) \
            .head(topn)

        if verbose:
            if self.articles_supplementary_information is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.articles_supplementary_information, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')

        return recommendations_df