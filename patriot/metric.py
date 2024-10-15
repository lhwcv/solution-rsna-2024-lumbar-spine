

#V 10 https://www.kaggle.com/code/metric/rsna-lumbar-metric-71549

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def get_condition(full_location: str) -> str:
    # Given an input like spinal_canal_stenosis_l1_l2 extracts 'spinal'
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')


def CALC_score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
    ) -> float:
    '''
    Pseudocode:
    1. Calculate the sample weighted log loss for each medical condition:
    2. Derive a new any_severe label.
    3. Calculate the sample weighted log loss for the new any_severe label.
    4. Return the average of all of the label group log losses as the final score, normalized for the number of columns in each group.
       This mitigates the impact of spinal stenosis having only half as many columns as the other two conditions.
    '''

    target_levels = ['normal_mild', 'moderate', 'severe']

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    solution['study_id'] = solution['row_id'].apply(lambda x: x.split('_')[0])
    solution['location'] = solution['row_id'].apply(lambda x: '_'.join(x.split('_')[1:]))
    solution['condition'] = solution['row_id'].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission['study_id'] = solution['study_id']
    submission['location'] = solution['location']
    submission['condition'] = solution['condition']

    condition_losses = []
    condition_weights = []
    for condition in ['spinal', 'foraminal', 'subarticular']:
        condition_indices = solution.loc[solution['condition'] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, 'sample_weight'].values
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    any_severe_spinal_labels = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_weights = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['sample_weight'].max())
    any_severe_spinal_predictions = pd.Series(submission.loc[submission['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_loss = sklearn.metrics.log_loss(
        y_true=any_severe_spinal_labels,
        y_pred=any_severe_spinal_predictions,
        sample_weight=any_severe_spinal_weights
    )
    condition_losses.append(any_severe_spinal_loss)
    condition_weights.append(1)
    return np.average(condition_losses, weights=condition_weights)

def get_level_target(valid_df, level=0):
    names = ["spinal_canal_stenosis",
             "left_neural_foraminal_narrowing",
             "right_neural_foraminal_narrowing",
             "left_subarticular_stenosis",
             "right_subarticular_stenosis"
             ]
    _idx_to_level_name = {
        1: 'l1_l2',
        2: 'l2_l3',
        3: 'l3_l4',
        4: 'l4_l5',
        5: 'l5_s1',
    }
    for i in range(len(names)):
        names[i] = '{}_{}'.format(names[i], _idx_to_level_name[level])
    return valid_df[names].values


def make_calc(data_root, test_stusy, l5, l4, l3, l2, l1, folds_tmp):
    # ref from patriot
    new_df = pd.DataFrame()
    tra_df = list(pd.read_csv(f"{data_root}/train.csv").columns[1:])
    col = []
    c_ = []
    level = []
    for i in test_stusy:
        for j in tra_df:
            col.append(f"{i}_{j}")
            c_.append(f"{i}")
            level.append(j.split("_")[-2])

    # print(level[:10])

    new_df["row_id"] = col
    new_df["study_id"] = c_
    new_df["level"] = level

    new_df["level"] = new_df["level"].astype("str")
    new_df["row_id"] = new_df["row_id"].astype("str")
    new_df["normal_mild"] = 0
    new_df["moderate"] = 0
    new_df["severe"] = 0
    new_df___ = []
    name__2 = {"l5": 0, "l4": 1, "l3": 2, "l2": 3, "l1": 4}
    for pred, level in zip([l5, l4, l3, l2, l1], [5, 4, 3, 2, 1]):
        name_ = f'l{level}'
        new_df_ = new_df[new_df["level"] == name_]
        # fold_tmp_ = folds_tmp[folds_tmp["level"] == name__2[name_]][
        #     ["spinal_canal_stenosis", "left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing",
        #      "left_subarticular_stenosis", "right_subarticular_stenosis"]].values

        fold_tmp_ = get_level_target(folds_tmp, level)
        new_df_[["normal_mild", "moderate", "severe"]] = pred.reshape(-1, 3)
        new_df_["GT"] = fold_tmp_.reshape(-1, )
        new_df___.append(new_df_)

    new_df = pd.concat(new_df___).sort_values("row_id").reset_index(drop=True)

    new_df = new_df[new_df["GT"] != -100].reset_index(drop=True)
    # この時点でauc計算でも良い？
    GT = new_df.iloc[:, [0, -1]].copy()
    GT[["normal_mild", "moderate", "severe"]] = np.eye(3)[GT["GT"].to_numpy().astype(np.uint8)]
    GT["sample_weight"] = 2 ** GT["GT"].to_numpy()

    GT = GT.iloc[:, [0, 2, 3, 4, 5]]
    metirc_ = CALC_score(GT, new_df.iloc[:, [0, 3, 4, 5]], row_id_column_name="row_id")
    return metirc_, new_df
