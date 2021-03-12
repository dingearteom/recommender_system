import numpy as np


class Mixin:
    MODEL_NAME = 'Mixin'

    def __init__(self, articles_supplementary_information=None, **kwargs):
        self.articles_supplementary_information = articles_supplementary_information
        self.params = kwargs
        sum = 0
        for _, value in self.params.items():
            sum += value

        np.testing.assert_almost_equal(sum, 1)

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, **kwargs):
        self.models = kwargs

    def recommend_items(self, user_id, topn=10, verbose=False):
        recommendations_df = None
        for (_, param), (_, model) in zip(self.params.items(), self.models.items()):
            if recommendations_df is None:
                recommendations_df = model.recommend_items(user_id, topn=1000000000).set_index('contentId') * param
            else:
                recommendations_df += model.recommend_items(user_id, topn=1000000000).set_index('contentId') * param

        recommendations_df = recommendations_df.reset_index().sort_values('recStrength', ascending=False).head(topn)

        if verbose:
            if self.articles_supplementary_information is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.articles_supplementary_information, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')
        return recommendations_df

