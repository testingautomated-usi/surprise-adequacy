import os
import pickle
import time
from typing import Dict, Tuple, List

import numpy as np
from dataclasses import dataclass
from sklearn import metrics

from surprise.smart_dsa_by_lsa import DSAbyLSA
from surprise.smart_dsa_diffnorms import DiffOfNormsSelectiveDSA
from surprise.smart_dsa_normdiffs import NormOfDiffsSelectiveDSA
from surprise.surprise_adequacy import DSA, SurpriseAdequacyConfig, SurpriseAdequacy, LSA
from case_study import config

# Note
USE_CACHE = False


class Result:
    def __init__(self,
                 name: str,
                 prepare_time: float,
                 approach_custom_info: Dict):
        self.name = name
        self.prepare_time = prepare_time
        self.approach_custom_info = approach_custom_info
        self.evals: Dict[str, 'TestSetEval'] = dict()


@dataclass
class TestSetEval:
    eval_time: float
    # avg_pr_score: float
    accuracy: float
    ood_auc_roc: float
    num_nominal_samples: int
    num_outlier_samples: int


def _get_thresholds(smart_class, model, train_x, sa_config):
    temp_dsa = smart_class(model=model,
                           train_data=train_x,
                           config=sa_config,
                           dsa_batch_size=config.DSA_BATCH_SIZE,
                           threshold=0.1  # Threshold does not matter here
                           )
    num_samples = train_x.shape[0]  # use subset to estimate thresholds
    num_sampled_thresholds = 10  # The number of thresholds collected from the samples
    sample_diffs = temp_dsa.sample_diff_distributions(train_x[:num_samples], num_samples=1000)
    # Take samples uniformly distributed over indexes
    indexes = np.floor(np.arange(0, num_sampled_thresholds) * (sample_diffs.shape[0] / num_sampled_thresholds))
    indexes = list(np.floor(indexes).astype(int))
    indexes.append(sample_diffs.shape[0] - 1)
    thresholds = list(sample_diffs[indexes])
    # Prepend a very small threshold
    thresholds.insert(0, thresholds[0] / 2)
    # Append a very large threshold
    thresholds.append(thresholds[len(thresholds) - 1] * 2)
    return thresholds


def time_experiments(model,
                     sa_config: SurpriseAdequacyConfig,
                     train_x: np.ndarray,
                     test_x: np.ndarray) -> None:
    results = []
    for train_share in (0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1):
        num_samples = int(train_x.shape[0] * train_share)
        train_subset = train_x[:num_samples]
        lsa = LSA(model=model, train_data=train_subset, config=sa_config)
        results.append(_time_result(f"lsa_{train_share}", lsa, x_test=test_x))

        dsa_st = DSA(model=model, train_data=train_subset, config=sa_config, dsa_batch_size=500, max_workers=1)
        results.append(_time_result("dsa_st_"+str(train_share), dsa_st, x_test=test_x))
    for line in results:
        print(line)


def _time_result(sa_name, sa, x_test):
    sa.prep(use_cache=False)
    start = time.time()
    sa.calc(x_test, ds_type='test', use_cache=False)
    total_time = time.time() - start
    return f"{sa_name} : {total_time}"


def run_experiments(model,
                    sa_config: SurpriseAdequacyConfig,
                    train_x: np.ndarray,
                    test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Result]:
    results = []

    nominal_data = test_data.pop("nominal")

    # Make sure inner lsa is cached for the smart dsa approach afterwards
    inner_lsa = LSA(model=model, train_data=train_x, config=sa_config)
    inner_lsa.prep(use_cache=False)
    lsa_values = inner_lsa._calc_lsa(target_ats=np.copy(inner_lsa.train_ats),
                                     target_pred=np.copy(inner_lsa.train_pred))
    for thresh_count, select_share in enumerate(range(10, 101, 10)):
        select_share /= 100
        dsa_by_lsa = DSAbyLSA(model=model, train_data=train_x, config=sa_config,
                              dsa_batch_size=config.DSA_BATCH_SIZE, select_share=select_share,
                              precomputed_likelihoods=lsa_values)
        custom_info = {
            "select_share": select_share,
            "dsa_batch_size": config.DSA_BATCH_SIZE
        }
        results.append(eval_for_sa(f"dsa_by_lsa_t{thresh_count}", dsa_by_lsa, custom_info, nominal_data, test_data))

    for train_percent in range(5, 101, 5):
        num_samples = int(train_x.shape[0] * train_percent / 100)
        train_subset = train_x[:num_samples]
        # DSA
        dsa = DSA(model=model, train_data=train_subset, config=sa_config, dsa_batch_size=config.DSA_BATCH_SIZE)
        dsa_custom_info = {"num_samples": num_samples, "dsa_batch_size": config.DSA_BATCH_SIZE}
        results.append(eval_for_sa(f"dsa_rand{train_percent}_perc", dsa, dsa_custom_info, nominal_data, test_data, ))
        # LSA
        lsa = LSA(model=model, train_data=train_subset, config=sa_config)
        lsa_custom_info = {"num_samples": num_samples}
        results.append(eval_for_sa(f"lsa_rand{train_percent}_perc", lsa, lsa_custom_info, nominal_data, test_data))

    thresholds = _get_thresholds(NormOfDiffsSelectiveDSA, model, train_x, sa_config)
    for thresh_count, diff_threshold in enumerate(thresholds):
        dsa = NormOfDiffsSelectiveDSA(model=model,
                                      train_data=train_x,
                                      config=sa_config,
                                      dsa_batch_size=config.DSA_BATCH_SIZE,
                                      threshold=diff_threshold)
        dsa_custom_info = {
            "diff_threshold": diff_threshold,
            "dsa_batch_size": config.DSA_BATCH_SIZE
        }
        results.append(eval_for_sa(f"dsa_nod_t{thresh_count}", dsa, dsa_custom_info, nominal_data, test_data))

    return results


def eval_for_sa(sa_name,
                sa: SurpriseAdequacy,
                approach_custom_info: Dict,
                nominal_data: Tuple[np.ndarray, np.ndarray],
                test_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Result:
    # Prepare SA (offline)
    prep_start_time = time.time()
    sa.prep(use_cache=USE_CACHE)
    prep_time = time.time() - prep_start_time
    if isinstance(sa, DiffOfNormsSelectiveDSA) or isinstance(sa, NormOfDiffsSelectiveDSA) or isinstance(sa, DSAbyLSA):
        approach_custom_info['num_samples'] = sa.number_of_samples

    # Create result object
    result = Result(name=sa_name, prepare_time=prep_time, approach_custom_info=approach_custom_info)

    nom_surp, nom_pred = sa.calc(target_data=nominal_data[0], use_cache=USE_CACHE, ds_type='test')

    for test_set_name, test_set in test_data.items():
        print(f"Evaluating {sa_name} with test set {test_set_name}")
        x_test = test_set[0]

        calc_start = time.time()
        surp, pred = sa.calc(target_data=x_test, use_cache=USE_CACHE, ds_type='test')
        calc_time = time.time() - calc_start

        # Used for (outlier-only) misclassification prediction
        is_misclassified = test_set[1] != pred
        accuracy = (x_test.shape[0] - np.sum(is_misclassified)) / x_test.shape[0]

        is_outlier = np.ones(shape=(nom_surp.shape[0] + surp.shape[0]), dtype=bool)
        is_outlier[:nom_surp.shape[0]] = 0
        combined_surp = np.concatenate((nom_surp, surp))
        ood_auc_roc = metrics.roc_auc_score(is_outlier, combined_surp)

        print(f"Result Preview ({sa_name}): {approach_custom_info['num_samples']} => auc roc {ood_auc_roc}")
        result.evals[test_set_name] = TestSetEval(eval_time=calc_time,
                                                  ood_auc_roc=ood_auc_roc,
                                                  accuracy=accuracy,
                                                  num_nominal_samples=nominal_data[0].shape[0],
                                                  num_outlier_samples=x_test.shape[0])

    return result


def save_results_to_fs(case_study: str, results: List[Result], model_id=int) -> None:
    for res in results:
        os.makedirs(f"/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/mnist_19012021/results/{case_study}/{res.name}", exist_ok=True)
        with open(f"/Users/rwiddhichakraborty/PycharmProjects/Thesis/apotoma/mnist_19012021/results/{case_study}/{res.name}/model_{model_id}.pickle", "wb+") as f:
            pickle.dump(res, file=f)
