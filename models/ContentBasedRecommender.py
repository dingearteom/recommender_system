import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import sklearn


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, articles_supplementary_information=None):
        self.articles_supplementary_information = articles_supplementary_information

    def fit(self, articles_df, interactions_train):
        """
        :param articles_df: dataFrame of shape (num_items, 1) with articles info.
                            Index is named contentId, a single column is named content.
        :param interactions_train: dataFrame of shape (num_persons, 2) with person-article interactions.
                            Index is named personId, two columns are contentId and eventStrength.
        :return:
        Sets tfidf_matrix, tfidf_feature_names, user_profiles, item_ids, interactions_train fields
        """
        assert isinstance(articles_df, pd.DataFrame), 'articles_df must be of type DataFrame'
        assert articles_df.index.name == 'contentId', 'Index of articles_df must be named contentId'
        assert len(articles_df.columns) == 1 and articles_df.columns[0] == 'content', 'articles_df must have a single' \
                                                                                      ' column named content'
        assert isinstance(interactions_train, pd.DataFrame), 'interactions_train must be of type DataFrame'
        assert interactions_train.index.name == 'personId', 'Index of interactions_train must be named personId'
        assert len(interactions_train.columns) == 2 and 'contentId' in interactions_train.columns and \
               'eventStrength' in interactions_train.columns, '' \
                                                              'interactions_train must have two columns: contentId and eventStrength'

        # Ignoring stopwords (words with no semantics) from English and Portuguese
        # (as we have a corpus with mixed languages)
        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

        # Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus,
        # ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)

        self.interactions_train = interactions_train
        self.item_ids = articles_df.index.tolist()
        self.tfidf_matrix = vectorizer.fit_transform(articles_df['content'])
        self.tfidf_feature_names = vectorizer.get_feature_names()

        def _get_item_profile(item_id):
            idx = self.item_ids.index(item_id)
            item_profile = self.tfidf_matrix[idx:idx + 1]
            return item_profile

        def _get_item_profiles(ids):
            item_profiles_list = [_get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def _build_users_profile(person_id):
            interactions_person_df = interactions_train.loc[person_id]
            user_item_profiles = _get_item_profiles(interactions_person_df['contentId'])

            user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
            # Weighted average of item profiles by the interactions strength

            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths),
                                                      axis=0) / np.sum(user_item_strengths)
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            return user_profile_norm

        def _build_users_profiles():
            user_profiles = {}
            for person_id in interactions_train.index.unique():
                user_profiles[person_id] = _build_users_profile(person_id)
            return user_profiles

        self.user_profiles = _build_users_profiles()

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.tfidf_matrix)[0]
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[i]) for i in similar_indices],
                               key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        items_to_ignore = self.interactions_train.loc[user_id]['contentId'].tolist()
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
            .head(topn)

        if verbose:
            if self.articles_supplementary_information is None:
                raise Exception('"articles_supplementary_information" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.articles_supplementary_information, how='left',
                                                          left_on='contentId',
                                                          right_on='contentId')
        return recommendations_df
