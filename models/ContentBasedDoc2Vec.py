import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.ContentBasedRecommender import ContentBasedRecommender
import models
import scipy
import sklearn
from gensim import utils
from gensim.models import Phrases, Word2Vec
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import models

nltk.download('punkt')


class ContentBasedDoc2Vec(ContentBasedRecommender):
    MODEL_NAME = 'Content-BasedDoc2Vec'

    def __init__(self, articles_supplementary_information=None, size_of_embedings=100):
        self.SIZE_OF_EMBEDINGS = size_of_embedings
        super(ContentBasedDoc2Vec, self).__init__(articles_supplementary_information)

    def fit(self, articles_df, interactions_train, verbose=False):
        """
        :param articles_df: dataFrame of shape (num_items, 1) with articles info.
                            Index is named contentId, a single column is named content.
        :param interactions_train: dataFrame of shape (num_persons, 2) with person-article interactions.
                            Index is named personId, two columns are contentId and eventStrength.

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

        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

        class MyCorpus:
            """An iterator that yields sentences (lists of str)."""

            def __iter__(self):
                for contentId in articles_df.index:
                    for sent in nltk.tokenize.sent_tokenize(articles_df.loc[contentId, 'content']):
                        yield utils.simple_preprocess(sent)

        if (verbose):
            print("Word2Vec is being trained...")
        bigram_transformer = Phrases(MyCorpus(), common_terms=stopwords_list, max_vocab_size=5000)
        model = Word2Vec(bigram_transformer[MyCorpus()], min_count=1, workers=4, max_vocab_size=5000,
                         size=self.SIZE_OF_EMBEDINGS)
        if (verbose):
            print("Word2Vec's training has been finished.")

        if (verbose):
            print("TF_IDF matrix is being built...")
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)
        self.tfidf_matrix = vectorizer.fit_transform(articles_df['content'])
        self.tfidf_feature_names = vectorizer.get_feature_names()
        self.tfidf_feature_names = {self.tfidf_feature_names[i]: i for i in range(len(self.tfidf_feature_names))}
        if (verbose):
            print("TF_IDF matrix's building has been finished.")

        doc2vec_dict = dict()

        if (verbose):
            print("Doc2Vec maxtrix is being built...")
        for i, contentId in enumerate(articles_df.index):
            tokenizer = RegexpTokenizer(r'\w+')
            doc2vec_dict[contentId] = np.zeros(self.SIZE_OF_EMBEDINGS) # size of embedings
            for sent in nltk.tokenize.sent_tokenize(articles_df.loc[contentId, 'content']):
                for phrase in bigram_transformer[tokenizer.tokenize(sent.lower())]:
                    if phrase in model.wv and phrase in self.tfidf_feature_names:
                        doc2vec_dict[contentId] += \
                            model.wv[phrase] * self.tfidf_matrix[i, self.tfidf_feature_names[phrase]]


        self.doc2vec_matrix = \
            csr_matrix(pd.DataFrame(np.vstack(tuple(list(doc2vec_dict.values()))),
                                    columns=range(self.SIZE_OF_EMBEDINGS)))

        if (verbose):
            print("Doc2Vec matrix building has been finished.")

        self.interactions_train = interactions_train
        self.item_ids = articles_df.index.tolist()

        def _get_item_profile(item_id):
            idx = self.item_ids.index(item_id)
            item_profile = self.doc2vec_matrix[idx:idx + 1]
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

    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.doc2vec_matrix)[0]
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[i]) for i in similar_indices],
                               key=lambda x: -x[1])
        return similar_items
