import pandas as pd
import numpy as np
import random


class ModelEvaluator:

    def __init__(self, EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS=100):
        self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS

    def fit(self, interactions_train, interactions_test):
        """
        :param interactions_train: dataframe with index personId and a single column contentId
        :param interactions_test: dataframe with index personId and a single column contentId
        :param EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS: number of non interacted items used in evaluation
        """
        assert isinstance(interactions_train, pd.DataFrame), 'interactions_train must be of type DateFrame'
        assert interactions_train.index.name == 'personId', 'interactions_train index must have name personId'
        assert len(interactions_train.columns) == 1 and interactions_train.columns[0] == 'contentId', \
            'interactions_train should have a single column contentId'
        assert isinstance(interactions_test, pd.DataFrame), 'interactions_test must be of type DateFrame'
        assert interactions_test.index.name == 'personId', 'interactions_test index must have name personId'
        assert len(interactions_test.columns) == 1 and interactions_test.columns[0] == 'contentId', \
            'interactions_test should have a single column contentId'

        self.interactions_test = interactions_test
        self.interactions_train = interactions_train
        self.all_items = set(self.interactions_train['contentId']).union(self.interactions_test['contentId'])

    @staticmethod
    def get_items_interacted(person_id, interactions_df):
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = ModelEvaluator.get_items_interacted(person_id, self.interactions_train).union(
            ModelEvaluator.get_items_interacted(person_id, self.interactions_test)
        )
        non_interacted_items = self.all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def _calculate_hits(self, person_interacted_items_testset, person_id, person_recs_df):
        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            # Getting a random sample (100) items the user has not interacted
            # (to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample \
                = self.get_not_interacted_items_sample(person_id,
                                                       sample_size=self.EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS,
                                                       seed=item_id % (2 ** 32))

            # Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            # Filtering only recommendations that are either the interacted item
            # or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]
            valid_recs = valid_recs_df['contentId'].values
            # Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        return hits_at_5_count, hits_at_10_count

    def evaluate_model_for_user(self, model, person_id):
        # Getting the items in test set
        interacted_values_testset = self.interactions_test.loc[[person_id]]
        person_interacted_items_testset = set(interacted_values_testset['contentId'])
        interacted_items_count_testset = len(person_interacted_items_testset)

        # Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, topn=10000000000)

        hits_at_5_count, hits_at_10_count = self._calculate_hits(person_interacted_items_testset, person_id,
                                                                 person_recs_df)
        # Recall is the rate of the interacted items that are ranked among the Top-N recommended items,
        # when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        # print(interacted_items_count_testset)
        precision_at_3 = person_recs_df.iloc[:3]['contentId'].isin(person_interacted_items_testset).mean()

        relevancy = np.array(person_recs_df['contentId'].isin(person_interacted_items_testset))
        num_relevent_elements = np.sum(relevancy)
        if num_relevent_elements != 0:
            precisions = relevancy.copy()
            precisions = np.cumsum(precisions) * np.array([1 / i for i in range(1, precisions.shape[0] + 1)])
            average_precision = np.sum(precisions * relevancy) / num_relevent_elements
        else:
            average_precision = 0

        person_metrics = {'hits@5_count': hits_at_5_count,
                          'hits@10_count': hits_at_10_count,
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'precision@3': precision_at_3,
                          'average_precision': average_precision}
        return person_metrics

    def evaluate_model(self, model, verbose=False):
        if verbose:
            print('Running evaluation for users')
        people_metrics = []
        num_users_to_process = self.interactions_test.index.nunique()
        for idx, person_id in enumerate(list(self.interactions_test.index.unique().values)):
            if verbose:
                if idx % 100 == 0 and idx > 0:
                    print('%d of %d users processed' % (idx, num_users_to_process))
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)

        detailed_results_df = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_precision_at_3 = detailed_results_df['precision@3'].mean()
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(
            detailed_results_df['interacted_count'].sum())
        global_average_precision = detailed_results_df['average_precision'].mean()

        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'precision@3': global_precision_at_3,
                          'mean_average_precision': global_average_precision}
        return global_metrics, detailed_results_df

