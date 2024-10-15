# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

conditions = ['Spinal Canal Stenosis',
              'Left Neural Foraminal Narrowing',
              'Right Neural Foraminal Narrowing',
              'Left Subarticular Stenosis',
              'Right Subarticular Stenosis']


def extract_info(study_id,
                 label_coordinates_df,
                 train_series,
                 ):
    study_coordinates = label_coordinates_df[
        label_coordinates_df['study_id'] == study_id]
    study_series = train_series[train_series['study_id'] == study_id]
    labeled_series_ids = study_coordinates['series_id'].unique().tolist()

    # if study_id == 3294654272:
    #     print(study_coordinates)

    # assert len(study_coordinates) == 25, print(study_coordinates)

    # TODO when use label_coordinates we should care this?
    # if len(study_coordinates) != 25:
    #     print('[WARN] not full labeled study_id {}, len_coord: {}'.format(study_id,
    #                                                                       len(study_coordinates)))
    des_to_sid = {}
    sid_to_des = {}
    series_ids = set()
    for sid, des in zip(study_series['series_id'], study_series['series_description']):

        if sid not in labeled_series_ids:
            print(f'[WARN] {study_id} / {sid} not in labeled coordinates')
        if study_id == 3637444890 and sid == 3892989905:
            print('skip 3892989905,  not lumbar image')
            continue
        sid_to_des[sid] = des
        if des in des_to_sid.keys():
            des_to_sid[des].append(sid)
        else:
            des_to_sid[des] = [sid]
        series_ids.add(sid)
    series_ids = list(series_ids)
    cond_to_sid = {}
    for cond in conditions:
        ids = study_coordinates[
            study_coordinates['condition'] == cond
            ].series_id.unique().tolist()
        cond_to_sid[cond] = ids

    infos = {}
    infos['study_id'] = study_id
    infos['sids'] = series_ids
    infos['cond_to_sid'] = cond_to_sid
    infos['des_to_sid'] = des_to_sid
    infos['sid_to_des'] = sid_to_des
    aux_infos = {}
    for cond in conditions:
        c = study_coordinates[
            study_coordinates['condition'] == cond
            ]
        coord_info = []
        for _, row in c.iterrows():
            coord_info.append(dict(row))
        coord_info = sorted(coord_info,
                            key=lambda k: int(k['level'].split('/')[0][1:]))

        aux_infos[cond] = coord_info

    infos['aux_infos'] = aux_infos
    return infos


def get_train_study_aux_info(data_root):
    label_coordinates_df = pd.read_csv(
        f'{data_root}/train_label_coordinates.csv')
    train_series = pd.read_csv(
        f'{data_root}/train_series_descriptions.csv')

    train_df = pd.read_csv(
        f'{data_root}/train.csv')

    n = 0
    for study_id in train_series['study_id'].unique():
        g = train_series[train_series['study_id'] == study_id]
        if len(g) > 3:
            n += 1
    print('samples have multi a_t2: ', n)

    infos = {}
    for study_id in train_df['study_id'].unique().tolist():
        infos[study_id] = extract_info(study_id, label_coordinates_df, train_series)
    return infos
