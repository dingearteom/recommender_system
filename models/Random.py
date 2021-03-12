import numpy as np


class Random:
    MODEL_NAME = 'Random'

    def __init__(self):
        pass

    def fit(self, interactions_train):
        self.interactions_train = interactions_train

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, topn=10, verbose=False):
        recommendations_df = self.interactions_train[['contentId']].reset_index(drop=True)
        recommendations_df['eventStrength'] = np.ones((recommendations_df.shape[0]))
        recommendations_df = recommendations_df.sample(frac=1).head(topn)

        return recommendations_df