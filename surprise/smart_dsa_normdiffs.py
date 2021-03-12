import numpy as np
import os
import tensorflow as tf
from typing import Optional

from surprise.surprise_adequacy import DSA, SurpriseAdequacyConfig


class NormOfDiffsSelectiveDSA(DSA):

    def __init__(self,
                 model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 threshold: float = 0.05,
                 dsa_batch_size: int = 500,
                 max_workers: Optional[int] = None) -> None:
        super().__init__(model, train_data, config, dsa_batch_size, max_workers)
        self.threshold = threshold

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

        new_class_matrix_norms_vec = {}
        new_ats = []  # Will be concatenated to get new self.train_ats
        new_pred = []  # Will be concatenated to get new self.train_pred
        selected_so_far = 0
        for label in range(10):

            # This index used in the loop indicates the latest element selected to be added to chosen items
            ats = all_train_ats[all_train_pred == label]
            is_available_mask = np.ones(dtype=bool, shape=ats.shape[0])

            chosen_per_label_indexes = []

            i = 0
            while True:
                # TODO Move this to a unit test
                # if len(chosen_per_label_indexes) > 0:
                #     assert np.min(np.linalg.norm(ats[i] - ats[chosen_per_label_indexes], axis=1)) >= self.threshold

                # Select with (all_train_ats) index i in the selected list of ats
                # and put its new index in the new matrix
                current_ats = ats[i]
                chosen_per_label_indexes.append(i)

                # Current ats is selected and becomes unavailable
                is_available_mask[i] = False

                # Calculate differences and update is_available_mask
                avail_ats = ats[is_available_mask]
                diffs = np.linalg.norm(avail_ats - current_ats, axis=1)
                is_available_indexes = np.where(is_available_mask)[0]
                drop_indeces = is_available_indexes[np.where(diffs < self.threshold)]
                is_available_mask[drop_indeces] = False

                i = np.argmax(is_available_mask)
                if i == 0:
                    break

            new_ats.append(ats[chosen_per_label_indexes])
            new_pred.append(np.full(shape=len(chosen_per_label_indexes), fill_value=label))
            index_list = np.arange(selected_so_far, selected_so_far + len(chosen_per_label_indexes))
            new_class_matrix_norms_vec[label] = index_list
            selected_so_far += len(chosen_per_label_indexes)

        self.train_ats = np.concatenate(new_ats)
        self.train_pred = np.concatenate(new_pred)
        self.number_of_samples = sum(len(lst) for lst in new_class_matrix_norms_vec.values())
        self.class_matrix = new_class_matrix_norms_vec

        # TODO Move this to a unit test
        # for label in range(10):
        #     selected_ats = self.train_ats[self.train_pred == label]
        #     for i in range(selected_ats.shape[0] - 1):
        #         # Note: This completely ignores labels
        #         min_dist = np.min(np.linalg.norm(selected_ats[1 + i:] - selected_ats[i], axis=1))
        #         assert min_dist >= self.threshold, f"Found difference {min_dist} < {self.threshold}"

    def sample_diff_distributions(self, x_subarray: np.ndarray, num_samples=100) -> np.ndarray:
        """
        Calculates all differences between the samples passed in the subarray.
        This can be used to guess thresholds for the algorithm.
        The threshold passed when creating this DSA instance is ignored.
        :param x_subarray: the subset of the train data (or any other data) for which to calc the differences
        :param num_samples: The number of samples to use (i.e., the subset of the subarray to consider)
        :return: Sorted one-dimensional array of differences
        """
        ats, pred = self._calculate_ats(x_subarray)
        differences = np.empty(shape=num_samples)
        for i in range(num_samples):
            # Note: This completely ignores labels
            min_dist = np.min(np.linalg.norm(ats[1 + i:] - ats[i], axis=1))
            differences[i] = min_dist
        return np.sort(differences)
