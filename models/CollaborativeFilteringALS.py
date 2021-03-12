from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pandas as pd


class CollaborativeFilteringALS:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, alpha_val=40, lambda_val=0.1, features=10, articles_supplementary_information=None):
        """

        :param alpha_val (int): The rate in which we'll increase our confidence
                in a preference with more interactions.
        :param lambda_val (float): Regularization value
        :param features (int): How many latent features we want to compute.
        """

        self.alpha_val = alpha_val
        self.lambda_val = lambda_val
        self.features = features
        self.articles_supplementary_information = articles_supplementary_information

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, interactions_train, iterations=10, verbose=False):
        """ Implementation of Alternating Least Squares with implicit data. We iteratively
            compute the user (x_u) and item (y_i) vectors using the following formulas:

            x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
            y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

            Args:
                interactions_train (DataFrame): Interactions, DataFrame with index personId
                 and a single column contentId.

                iterations (int): How many times we alternate between fixing and
                updating our user and item vectors

                verbose (bool): Whether you wish a progress to be printed

            Sets: self.cf_predictions_df - a matrix for recommendations
                  self.interactions_train
        """
        assert isinstance(interactions_train, pd.DataFrame), "interactions_train must be of type DataFrame"

        self.interactions_train = interactions_train

        ratings = interactions_train.reset_index().pivot(index='personId',
                                                         columns='contentId',
                                                         values='eventStrength').fillna(0)
        users_ids = list(ratings.index)
        items_ids = list(ratings.columns)
        ratings = ratings.values
        ratings = sparse.csr_matrix(ratings)

        tmp = interactions_train.reset_index()
        tmp['one'] = np.ones((interactions_train.shape[0]))
        preference = tmp.pivot(index='personId',
                               columns='contentId',
                               values='one').fillna(0)
        preference = preference.values
        preference = sparse.csr_matrix(preference)

        confidence = ratings * self.alpha_val
        user_size, item_size = preference.shape

        # We create the user vectors X of size users-by-features, the item vectors
        # Y of size items-by-features and randomly assign the values.
        X = sparse.csr_matrix(np.random.normal(size=(user_size, self.features)))
        Y = sparse.csr_matrix(np.random.normal(size=(item_size, self.features)))

        # Precompute I and lambda * I
        X_I = sparse.eye(user_size)
        Y_I = sparse.eye(item_size)

        I = sparse.eye(self.features)
        lI = self.lambda_val * I

        # Start main loop. For each iteration we first compute X and then Y
        for i in range(iterations):
            if verbose:
                print('iteration %d of %d' % (i + 1, iterations))

            # Precompute Y-transpose-Y and X-transpose-X
            yTy = Y.T.dot(Y)
            xTx = X.T.dot(X)

            # Loop through all users
            for u in range(user_size):
                # Get the user row.
                u_row = confidence[u, :].toarray()

                # Calculate the binary preference p(u)
                p_u = preference[u, :].toarray()

                # Calculate Cu and Cu - I
                CuI = sparse.diags(u_row, [0])
                Cu = CuI + Y_I

                # Put it all together and compute the final formula
                yT_CuI_y = Y.T.dot(CuI).dot(Y)
                yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
                X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

            for i in range(item_size):
                # Get the item column and transpose it.
                i_row = confidence[:, i].T.toarray()

                # Calculate the binary preference p(i)
                p_i = preference[:, i].T.toarray()

                # Calculate Ci and Ci - I
                CiI = sparse.diags(i_row, [0])
                Ci = CiI + X_I

                # Put it all together and compute the final formula
                xT_CiI_x = X.T.dot(CiI).dot(X)
                xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
                Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

        all_user_predicted_ratings = np.dot(X, Y.transpose()).todense()
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        self.cf_predictions_df = pd.DataFrame(all_user_predicted_ratings_norm,
                                              columns=items_ids,
                                              index=users_ids).transpose()

    def recommend_items(self, user_id, topn=10, verbose=False):
        # Get and sort the user's predictions
        items_to_ignore = set(self.interactions_train.loc[user_id]['contentId'])
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
            .reset_index().rename(columns={'index': 'contentId', user_id: 'recStrength'})

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