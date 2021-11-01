# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import utils.rankings as rnk
from models.linearmodel import LinearModel
from algorithms.basiconlineranker import BasicOnlineRanker
from multileaving.TeamDraftMultileave import TeamDraftMultileave
from utils.fair_util import *


# Dueling Bandit Gradient Descent
class TD_DBGD(BasicOnlineRanker):

    def __init__(self, learning_rate, learning_rate_decay, fair_alpha, decay_mode, unfairness, *args, **kargs):
        super(TD_DBGD, self).__init__(*args, **kargs)
        self.learning_rate = learning_rate
        self.model = LinearModel(n_features=self.n_features, learning_rate=learning_rate, n_candidates=1,
                                 learning_rate_decay=learning_rate_decay)
        self.multileaving = TeamDraftMultileave(n_results=self.n_results)

        self.fair_alpha = fair_alpha
        self.decay_mode = decay_mode
        self.unfairness = unfairness
        self.current_unfairness = 0
        self.group_candidates = {self.n_results: generate_all_combination(self.n_results)}
        self.uf_dict = {}
        self.uf = {}
        self.create_unfairness_table(self.n_results)

    @staticmethod
    def default_parameters():
        parent_parameters = BasicOnlineRanker.default_parameters()
        parent_parameters.update({'learning_rate': 0.01, 'learning_rate_decay': 1.0, })
        return parent_parameters

    def create_unfairness_table(self, k):
        self.uf_dict[k] = {}
        position_bias = position_probability(k, self.decay_mode)
        uf_table = self.group_candidates[k] * (-1 - self.fair_alpha) + 1
        self.uf[k] = np.dot(uf_table, position_bias)
        for i in range(len(self.group_candidates[k])):
            group_sequence = self.group_candidates[k][i]
            self.uf_dict[k]["".join(str(v) for v in group_sequence)] = self.uf[k][i]

    def calculate_unfairness(self, ranking):

        # get group information
        query_group = self.get_query_groups(self._last_query_id, self._train_groups, self._train_query_ranges)
        if self.unfairness == 'projected':
            k = min(len(ranking), self.n_results)
            if k not in self.uf_dict:
                self.group_candidates[k] = generate_all_combination(k)
                self.create_unfairness_table(k)
            ranking_group = query_group[ranking]
            self.current_unfairness += self.uf_dict[k]["".join(str(v) for v in ranking_group)]
        else:
            print("unfairness calculation is not included")
            exit()

    def get_test_rankings(self, features, query_ranges, inverted=True):
        scores = self.model.score(features)
        return rnk.rank_multiple_queries(scores, query_ranges, inverted=inverted, n_results=self.n_results)

    def _create_train_ranking(self, query_id, query_feat, inverted):
        assert inverted == False
        self.model.sample_candidates()
        scores = self.model.candidate_score(query_feat)
        rankings = rnk.rank_single_query(scores, inverted=False, n_results=self.n_results)
        multileaved_list = self.multileaving.make_multileaving(rankings)
        self.calculate_unfairness(multileaved_list[:self.n_results])
        return multileaved_list

    def update_to_interaction(self, clicks):
        winners = self.multileaving.winning_rankers(clicks)
        self.model.update_to_mean_winners(winners)
