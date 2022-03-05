# -*- coding: utf-8 -*-
from __future__ import division, print_function

import pickle
import json
import glob

import numpy as np
from demos import cmd


def run_target_1_summary():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    results = []
    for filename in params['dataset_files']:
        for fea in params['features']:
            for trec in params['trecs']:
                results.append(run_summary(filename=str(filename).split(".")[0], fea=str(fea), trec=trec))

    for line in results:
        print("{}, {}, {}, {} ({})".format(line['feature'], line['dataset'], line['trec'], line['median'], line['iqr']))

    with open('../dump/simulation_results.pickle', 'w') as handle:
        pickle.dump(results, handle)


def run_summary(filename=None, fea=None, trec=None):
    if filename is None or fea is None or trec is None:
        raise Exception("invalid params run_summary")

    strec = '@' + str(int(trec * 100))

    files = glob.glob(
        "/home/vitor-apolinario/Desktop/harmless/Vuls_HPC/dump/features_hpcc_{}_{}_{}*.pickle".format(filename, fea, strec))

    costs = []

    for f in files:
        try:
            with open(f, "r") as handle:
                execution_result = pickle.load(handle)
                cost = execution_result['stats']['unique'] / execution_result['stats']['files']

                if not isinstance(cost, float) or not cost > 0:
                    raise Exception('invalid cost {} @{}'.format(f, str(trec)))

                costs.append(cost)
        except:
            raise Exception("unable to summaryze {}".format(f))
            pass

    median = int(np.median(costs) * 100)
    iqr = int((np.percentile(costs, 75) - np.percentile(costs, 25)) * 100)

    return {"dataset": filename, "feature": fea, "trec": trec, "median": median, "iqr": iqr}


def export_results():
    try:
        with open('../dump/simulation_results.pickle') as handle:
            results = pickle.load(handle)
    except:
        raise Exception('problem reading results file')

    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    trecs_df = { str(trec): None for trec in params['trecs'] }
    ds_df    = { str(ds).replace('.csv', ''): None for ds in params['dataset_files'] }

    dictdf = {}

    for fea in params['features']:
        dictdf[str(fea)] = ds_df.copy()

        for ds in dictdf[str(fea)]:
            dictdf[str(fea)][ds] = trecs_df.copy()

    for result in results:
        fea = result['feature']
        ds  = result['dataset']
        trec = result['trec']
        dictdf[fea][ds][str(trec)] = "{} ({})".format(result['median'], result['iqr'])

    # todo: need to transform in csv
    return dictdf


def check_missing_results():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    for ds_filename in params['dataset_files']:
        ds_rerun_params = []
        ds_filename = str(ds_filename).replace('.csv', '')
        for fea in params['features']:
            for trec in params['trecs']:
                for seed in range(30):
                    strec = '@' + str(int(trec * 100))
                    try:
                        result_file_path = "/home/vitor-apolinario/Desktop/harmless/Vuls_HPC/dump/" \
                                           "features_hpcc_{}_{}_{}_{}.pickle".format(ds_filename, str(fea), strec,seed)

                        with open(result_file_path, "r") as handle:
                            execution_result = pickle.load(handle)
                            cost = execution_result['stats']['unique'] / execution_result['stats']['files']

                            if not isinstance(cost, float) or not cost > 0:
                                raise Exception('invalid cost')
                    except:
                        try:
                            raw_filename = result_file_path.split('/')[-1].replace('.pickle', '')
                            seed = raw_filename.split('_')[-1]
                            run = {'fea': str(fea), 'seed': seed, 'filename': "{}.csv".format(ds_filename), 'trec': trec}
                            ds_rerun_params.append(run)
                        except:
                            raise Exception("Unable to check {} {}".format(result_file_path, trec))

        ds_rerun_params = sorted(ds_rerun_params, key = lambda r: (r['filename'], r['fea'], r['trec'], r['seed']))

        for x in ds_rerun_params:
            print(x)

        print("{} missing targets for {}".format(len(ds_rerun_params), ds_filename))

        with open("../memory/rerun_params_{}.pickle".format(ds_filename), "w") as handle:
            pickle.dump(ds_rerun_params, handle)


if __name__ == "__main__":
    eval(cmd())
