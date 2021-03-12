import os
from typing import Optional

import numpy as np
import tensorflow as tf

from surprise.surprise_adequacy import DSA, SurpriseAdequacyConfig, LSA


class DSAbyLSA(DSA):

    def __init__(self,
                 model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 select_share: float,
                 dsa_batch_size=500,
                 precomputed_likelihoods: np.ndarray = None,
                 max_workers: Optional[int] = None) -> None:
        super().__init__(model, train_data, config, dsa_batch_size, max_workers)
        self.select_share = select_share
        self.precomputed_likelihoods = precomputed_likelihoods

    def _load_or_calc_train_ats(self, use_cache=False) -> None:
        saved_train_path = self._get_saved_path("train")

        if os.path.exists(saved_train_path[0]) and use_cache:
            print("Found saved {} ATs, skip serving".format("train"))
            # In case train_ats is stored in a disk
            self.train_ats, self.train_pred = np.load(saved_train_path[0]), np.load(saved_train_path[1])

        else:
            self.train_ats, self.train_pred = self._load_or_calculate_ats(dataset=self.train_data, ds_type="train",
                                                                          use_cache=use_cache)

        self._select_smart_ats()

    def prep(self, use_cache: bool = False) -> None:
        self._load_or_calc_train_ats(use_cache=use_cache)

    def _select_smart_ats(self):

        all_train_ats = self.train_ats
        all_train_pred = self.train_pred

        # Note: We're not passing model and train_data as ats are already cached
        if self.precomputed_likelihoods is not None:
            lsa_values = self.precomputed_likelihoods
        else:
            inner_lsa = LSA(model=None, train_data=None, config=self.config)
            inner_lsa.prep(use_cache=True)
            lsa_values = inner_lsa._calc_lsa(target_ats=all_train_ats, target_pred=all_train_pred)

        new_class_matrix_norms_vec = {}
        new_ats = []  # Will be concatenated to get new self.train_ats
        new_pred = []  # Will be concatenated to get new self.train_pred
        selected_so_far = 0
        for label in range(10):
            available_indices = np.where(all_train_pred == label)[0]
            for_label_lsa = lsa_values[available_indices]
            for_label_ats = all_train_ats[available_indices]

            for_label_indexes_sorted_by_lsa = np.argsort(for_label_lsa)
            num_chosen_samples = int(np.floor(available_indices.shape[0] * self.select_share))
            selected_for_label_indexes = for_label_indexes_sorted_by_lsa[:num_chosen_samples]
            chosen_ats = for_label_ats[selected_for_label_indexes]

            new_ats.append(np.copy(chosen_ats))
            new_pred.append(np.full(shape=num_chosen_samples, fill_value=label))
            index_list = np.arange(selected_so_far, selected_so_far + num_chosen_samples)
            new_class_matrix_norms_vec[label] = index_list
            selected_so_far += num_chosen_samples

        self.train_ats = np.concatenate(new_ats)
        self.train_pred = np.concatenate(new_pred)
        self.number_of_samples = sum(len(lst) for lst in new_class_matrix_norms_vec.values())
        self.class_matrix = new_class_matrix_norms_vec
