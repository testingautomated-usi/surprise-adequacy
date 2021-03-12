import abc
import os
import pickle
from abc import ABC
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Tuple, List, Union, Dict

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from scipy.stats import gaussian_kde
from tensorflow.keras.models import Model
from tqdm import tqdm


@dataclass
class SurpriseAdequacyConfig:
    """Stores basic immutable surprise adequacy configuration.
    Instances of this class are reusable amongst different instances of surprise adequacy.

    Note: Jetbrains 'unresolved reference' is wrong: https://youtrack.jetbrains.com/issue/PY-28549

    Args:
        is_classification (bool): A boolean indicating if the NN under test solves a classification problem.
        num_classes (None, int): The number of classes (for classification problems)
        or None (for regression problems). Default: None

        layer_names (List(str)): List of layer names whose ATs are to be extracted. Code takes last layer.
        saved_path (str): Path to store and load ATs
        dataset_name (str): Dataset to be used. Currently supports mnist and cifar-10.
        num_classes (int): No. of classes in classification. Default is 10.
        min_var_threshold (float): Threshold value to check variance of ATs
        batch_size (int): Batch size to use while predicting.

     Raises:
        ValueError: If any of the config parameters takes an illegal value.
    """

    saved_path: str
    is_classification: bool
    layer_names: List[str]
    ds_name: str
    num_classes: Union[int, None]
    min_var_threshold: float = 1e-5
    batch_size: int = 128

    def __post_init__(self):
        if self.is_classification and not self.num_classes:
            raise ValueError("num_classes is a mandatory parameter "
                             "in SurpriseAdequacyConfig for classification problems")
        elif not self.is_classification and self.num_classes:
            raise ValueError(f"num_classes must be None (but was {self.num_classes}) "
                             "in SurpriseAdequacyConfig for classification problems")
        elif self.num_classes < 0 and self.is_classification:
            raise ValueError(f"num_classes must be positive but was {self.num_classes}) ")
        elif self.min_var_threshold < 0:
            raise ValueError(f"Variance threshold cannot be negative, but was {self.min_var_threshold}")

        elif self.ds_name is None or self.ds_name == "":
            raise ValueError(f"dataset name must not be None or empty")

        elif len(self.layer_names) == 0:
            raise ValueError(f"Layer list cannot be empty")
        elif len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError(f"Layer list cannot contain duplicates")


class SurpriseAdequacy(ABC):

    def __init__(self, model: tf.keras.Model, train_data: np.ndarray, config: SurpriseAdequacyConfig) -> None:
        self.model = model
        self.train_data = train_data
        self.train_ats = None
        self.train_pred = None
        self.class_matrix = {}
        self.config = config

    def _get_saved_path(self, ds_type: str) -> Tuple[str, str]:
        """Determine saved path of ats and pred

        Args:
            ds_type: Type of dataset: Typically one of {Train, Test, Target}.

        Returns:
            ats_path: File path of ats.
            pred_path: File path of pred (independent of layers)
        """

        joined_layer_names = "_".join(self.config.layer_names)

        return (
            os.path.join(
                self.config.saved_path,
                self.config.ds_name + "_" + ds_type + "_" + joined_layer_names + "_ats" + ".npy",
            ),
            os.path.join(self.config.saved_path, self.config.ds_name + "_" + ds_type + "_pred" + ".npy"),
        )

    # Returns ats and returns predictions
    def _load_or_calculate_ats(self, dataset: np.ndarray, ds_type: str, use_cache: bool) -> Tuple[
        np.ndarray, np.ndarray]:

        """Determine activation traces train, target, and test datasets

        Args:
            dataset (ndarray): x_train or x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            ats (ndarray): Activation traces (Shape of num_examples * num_nodes).
            pred (ndarray): 1-D Array of predictions

        """
        print(f"Calculating the ats for {ds_type} dataset")

        saved_target_path = self._get_saved_path(ds_type)
        if saved_target_path is not None and os.path.exists(saved_target_path[0]) and use_cache:
            print(f"Found saved {ds_type} ATs, skip at collection from model")
            return self._load_ats(ds_type)
        else:
            ats, pred = self._calculate_ats(dataset)

            if saved_target_path is not None:
                np.save(saved_target_path[0], ats)
                np.save(saved_target_path[1], pred)
                print(f"[{ds_type}] Saved the ats and predictions to {saved_target_path[0]} and {saved_target_path[1]}")

            return ats, pred

    @classmethod
    def _output_dim_reduction(cls, layer_output):
        return np.mean(layer_output, axis=tuple(range(1, layer_output.ndim - 1)))

    def _calculate_ats(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        output_layers = [self.model.get_layer(layer_name).output for layer_name in self.config.layer_names]
        output_layers.append(self.model.output)
        temp_model = Model(
            inputs=self.model.input,
            outputs=output_layers
        )

        # Get the activation traces of the inner layers and the output of the final layer
        layer_outputs: List[np.ndarray] = temp_model.predict(dataset, batch_size=self.config.batch_size, verbose=1)
        # Remove the (output layer) dnn outputs from the list and store them as separate result
        dnn_output = layer_outputs.pop()

        if self.config.is_classification:
            pred = np.argmax(dnn_output, axis=1)

            ats = None
            for layer_name, layer_output in zip(self.config.layer_names, layer_outputs):
                print("Layer: " + layer_name)
                if layer_output[0].ndim >= 3:
                    # (primarily for convolutional layers - note that kim et al used ndim==3)
                    layer_matrix = self._output_dim_reduction(layer_output)
                else:
                    layer_matrix = np.array(layer_output)

                if ats is None:
                    # Shape of ats will be num_inputs x num_nodes_in_layer
                    ats = layer_matrix
                else:
                    ats = np.append(ats, layer_matrix, axis=1)

        return ats, pred

    def _load_ats(self, ds_type: str) -> Tuple[np.ndarray, np.ndarray]:
        # In case train_ats is stored in a disk
        saved_target_path = self._get_saved_path(ds_type)
        ats: np.ndarray = np.load(saved_target_path[0])
        pred: np.ndarray = np.load(saved_target_path[1])
        return ats, pred

    def _load_or_calc_train_ats(self, use_cache=False) -> None:
        """Load or get actviation traces of training inputs

        Args:
            use_cache: To load stored files or not

        Returns:
            None. train_ats and train_pred are init() variables in super class NoveltyScore.

        """

        saved_train_path = self._get_saved_path("train")

        if os.path.exists(saved_train_path[0]) and use_cache:
            print("Found saved {} ATs, skip serving".format("train"))
            # In case train_ats is stored in a disk
            self.train_ats, self.train_pred = np.load(saved_train_path[0]), np.load(saved_train_path[1])

        else:
            self.train_ats, self.train_pred = self._load_or_calculate_ats(dataset=self.train_data, ds_type="train",
                                                                          use_cache=use_cache)

    def prep(self, use_cache: bool = False) -> None:
        """
        Prepare class matrix from training activation traces. Class matrix is a dictionary
        with keys as labels and values as lists of positions as predicted by model

        Args:
            use_cache: bool If true, prepared values (activation traces, ...) will be
            stored on the file system for later use.

        Returns:
            None.

        """
        self._load_or_calc_train_ats(use_cache=use_cache)
        if self.config.is_classification:
            # TODO Check if we can vectorize this loop
            for i, label in enumerate(self.train_pred):
                if label not in self.class_matrix:
                    self.class_matrix[label] = []
                self.class_matrix[label].append(i)

    def clear_cache(self, saved_path: str) -> None:
        """

        Delete files of activation traces.

        Args:
            saved_path(str): Base directory path

        """
        to_remove = ['train', 'test', 'target']
        for f in to_remove:
            path = self._get_saved_path(f)
            os.remove(os.path.join(saved_path, path[0]))
            os.remove(os.path.join(saved_path, path[1]))

        # files = [f for f in os.listdir(saved_path) if f.endswith('.npy')]
        # for f in files:
        #     os.remove(os.path.join(saved_path, f))

    @abc.abstractmethod
    def calc(self, target_data: np.ndarray, use_cache: bool, ds_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates prediction and novelty scores
        :param target_data: a numpy array consisting of the data to be tested
        :param use_cache: whether or not to use caching, i.e., re-use ats from previous cals to calc on *any* SA
        :param ds_type: string, 'train' or 'test'
        :return: A tuple of two one-dimensional arrays: surprises and predictions
        """
        pass


class LSA(SurpriseAdequacy):

    def __init__(self, model: tf.keras.Model, train_data: np.ndarray, config: SurpriseAdequacyConfig) -> None:
        super().__init__(model, train_data, config)
        self.kdes = None
        self.removed_rows = None

    def prep(self, use_cache: bool = False) -> None:
        super().prep(use_cache=use_cache)
        self._load_or_create_likelyhood_estimator(use_cache=use_cache)

    def _load_or_create_likelyhood_estimator(self, use_cache=False) -> None:
        """Load or get actviation traces of training inputs

        Args:
            use_cache: To load stored files or not

        Returns:
            None. train_ats and train_pred are init() variables in super class NoveltyScore.

        """

        kdes_path = os.path.join(self.config.saved_path, self.config.ds_name + "kdes.npy")
        rem_row_path = os.path.join(self.config.saved_path, self.config.ds_name + "remrows.npy")

        if os.path.exists(kdes_path) and os.path.exists(rem_row_path) and use_cache:
            with open(kdes_path, 'rb') as file:
                self.kdes = pickle.load(file)
            with open(rem_row_path, 'rb') as file:
                self.removed_rows = pickle.load(file)
        else:
            self.kdes, self.removed_rows = self._calc_kdes()
            with open(kdes_path, 'wb') as file:
                pickle.dump(self.kdes, file=file)
            with open(rem_row_path, 'wb') as file:
                pickle.dump(self.removed_rows, file=file)

    def calc(self, target_data: np.ndarray, ds_type: str, use_cache=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return LSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            lsa (float): List of scalar LSA values

        """
        assert self.kdes is not None and self.removed_rows is not None, \
            "LSA has not yet been prepared. Run lsa.prep()"

        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_type=ds_type, use_cache=use_cache)

        print(f"[{ds_type}] Calculating LSA")
        lsa_as_list = self._calc_lsa(target_ats, target_pred)
        return np.array(lsa_as_list), target_pred

    def _calc_kdes(self) -> Tuple[dict, List[int]]:
        """
        Determine Gaussian KDE for each label and list of removed rows based on variance threshold, if any.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            kdes: Dict - labels are keys, values are scipy kde objects
            removed_rows: Array of positions of removed rows

        """

        if self.config.is_classification:
            kdes, removed_rows = self._classification_kdes()
        else:
            kdes, removed_rows = self._regression_kdes()

        print((f"Ignoring the activations of {len(removed_rows)} traces "
               f"as their variance is not high enough."))

        return kdes, removed_rows

    def _regression_kdes(self) -> Tuple[List[gaussian_kde], List[int]]:
        removed_rows = []
        row_vectors = np.transpose(self.train_ats)
        for activation_node in range(row_vectors.shape[0]):
            if np.var(row_vectors[activation_node]) < self.config.min_var_threshold:
                removed_rows.append(activation_node)
        refined_ats = np.transpose(self.train_ats)
        refined_ats = np.delete(refined_ats, removed_rows, axis=0)
        if refined_ats.shape[0] != 0:

            kdes = [self._create_gaussian_kde(refined_ats)]
            return kdes, removed_rows

        else:
            raise ValueError(f"All ats were removed by threshold: ", self.config.min_var_threshold)

    def _classification_kdes(self) -> Tuple[Dict[int, gaussian_kde], List[int]]:
        removed_rows = []
        for label in range(self.config.num_classes):
            # Shape of (num_activation nodes x num_examples_by_label)
            row_vectors: np.ndarray = np.transpose(self.train_ats[self.class_matrix[label]])
            positions: np.ndarray = np.where(np.var(row_vectors) < self.config.min_var_threshold)[0]

            for p in positions:
                removed_rows.append(p)
        removed_rows = list(set(removed_rows))

        kdes = {}
        for label in tqdm(range(self.config.num_classes), desc="kde"):

            refined_ats = np.transpose(self.train_ats[self.class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_rows, axis=0)

            if refined_ats.shape[0] == 0:
                print(f"Ats for label {label} were removed by threshold {self.config.min_var_threshold}")
                break

            kdes[label] = self._create_gaussian_kde(refined_ats)

        return kdes, removed_rows

    @staticmethod
    def _create_gaussian_kde(refined_ats):
        return gaussian_kde(refined_ats)

    def _calc_lsa(self,
                  target_ats: np.ndarray,
                  target_pred: np.ndarray) -> np.ndarray:
        """
        Calculate scalar LSA value of target activation traces

        Args:
            target_ats (ndarray): Activation traces of target_data.
            target_pred(ndarray): 1-D Array of predicted labels
            ds_type (str): Type of dataset: Test or Target.
            removed_rows (list): Positions to skip
            kdes: Dict of scipy kde objects

        Returns:
            lsa (float): List of scalar LSA values

        """

        if self.config.is_classification:
            lsa: np.ndarray = self._calc_classification_lsa(target_ats, target_pred)
        else:
            lsa: np.ndarray = self._calc_regression_lsa(target_ats)
        return lsa

    def _calc_regression_lsa(self, target_ats: np.ndarray) -> np.ndarray:
        kde = self.kdes[0]
        refined_at: np.ndarray = np.delete(target_ats, self.removed_rows, axis=1)
        return -kde.logpdf(np.transpose(refined_at))

    def _calc_classification_lsa(self,
                                 target_ats: np.ndarray,
                                 target_pred: np.ndarray) -> np.ndarray:
        result = np.empty(shape=target_pred.shape, dtype=float)
        refined_ats = np.delete(target_ats, self.removed_rows, axis=1)
        for label in self.class_matrix.keys():
            for_label_indexes = target_pred == label
            kde = self.kdes[label]
            selected_ats = refined_ats[for_label_indexes]
            result[for_label_indexes] = -kde.logpdf(np.transpose(selected_ats))
        return result


class DSA(SurpriseAdequacy):

    def __init__(self, model: tf.keras.Model,
                 train_data: np.ndarray,
                 config: SurpriseAdequacyConfig,
                 dsa_batch_size=500,
                 max_workers=None) -> None:
        super().__init__(model, train_data, config)
        self.dsa_batch_size = dsa_batch_size
        self.max_workers = max_workers

    def calc(self, target_data: np.ndarray, ds_type: str, use_cache=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return DSA values for target. Note that target_data here means both test and adversarial data. Separate calls in main.

        Args:
            target_data (ndarray): x_test or x_target.
            ds_type (str): Type of dataset: Train, Test, or Target.
            use_cache (bool): Use stored files to load activation traces or not

        Returns:
            dsa (float): List of scalar DSA values

        """
        target_ats, target_pred = self._load_or_calculate_ats(dataset=target_data, ds_type=ds_type, use_cache=use_cache)
        return self._calc_dsa(target_ats, target_pred, ds_type), target_pred

    def _calc_dsa(self, target_ats: np.ndarray, target_pred: np.ndarray, ds_type: str) -> np.ndarray:

        """
        Calculate scalar DSA value of target activation traces

        Args:
            target_ats (ndarray): Activation traces of target_data.
            ds_type (str): Type of dataset: Test or Target.
            target_pred (ndarray): 1-D Array of predicted labels

        Returns:
            dsa (float): List of scalar DSA values

        """

        start = 0

        print(f"[{ds_type}] Calculating DSA")

        num_targets = target_pred.shape[0]
        futures = []
        dsa = np.empty(shape=target_pred.shape[0])

        print(f"[{self.__class__}] Using {self.train_ats.shape[0]} train samples")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while start < num_targets:

                # Select batch
                diff = num_targets - start
                if diff < self.dsa_batch_size:
                    batch = target_pred[start:start + diff]
                else:
                    batch = target_pred[start: start + self.dsa_batch_size]

                # Calculate DSA per label
                for label in range(self.config.num_classes):

                    def task(t_batch, t_label, t_start):
                        matches = np.where(t_batch == t_label)
                        if len(matches) > 0:
                            a_min_dist, b_min_dist = self._dsa_distances(t_label, matches, t_start, target_ats)
                            t_task_dsa = a_min_dist / b_min_dist
                            return matches[0], t_start, t_task_dsa
                        else:
                            return None, None, None

                    futures.append(executor.submit(task, np.copy(batch), label, start))

                start += self.dsa_batch_size

        for future in futures:
            f_idxs, f_start, f_task_dsa = future.result()
            if f_idxs is not None:
                dsa[f_idxs + f_start] = f_task_dsa

        return dsa

    def _dsa_distances(self, label: int, matches: np.ndarray, start: int, target_ats: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:

        target_matches = target_ats[matches[0] + start]
        train_matches_same_class = self.train_ats[self.class_matrix[label]]
        a_dist = target_matches[:, None] - train_matches_same_class
        a_dist_norms = np.linalg.norm(a_dist, axis=2)
        a_min_dist = np.min(a_dist_norms, axis=1)
        closest_position = np.argmin(a_dist_norms, axis=1)
        closest_ats = train_matches_same_class[closest_position]
        other_classes_indexes = np.ones(shape=self.train_ats.shape[0], dtype=bool)
        other_classes_indexes[self.class_matrix[label]] = 0
        train_matches_other_classes = self.train_ats[other_classes_indexes]
        b_dist = closest_ats[:, None] - train_matches_other_classes
        b_dist_norms = np.linalg.norm(b_dist, axis=2)
        b_min_dist = np.min(b_dist_norms, axis=1)

        return a_min_dist, b_min_dist
