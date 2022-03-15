# -*- coding: utf-8 -*-
from __future__ import division, print_function

import pickle
import json
import glob

import numpy as np
import pandas as pd

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


def get_recall_curve():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    for filename in params['dataset_files']:
        filename=str(filename).split(".")[0]
        for fea in params['features']:
            results = {}
            num_pos = None
            num_files = None
            strec = '@95'
            files = glob.glob("../dump/features_hpcc_{}_{}_{}*.pickle".format(str(filename), str(fea), strec))

            for f in files:
                with open(f, "r") as handle:
                    execution_result = pickle.load(handle)
                    if not num_pos:
                        num_pos = execution_result['stats']['truepos'] + execution_result['stats']['unknownyes']
                        num_files = execution_result['stats']['files']

                    for i, step in enumerate(execution_result['loops']['x']):
                        reviewed_perc = np.round(step/num_files, 2)
                        if not reviewed_perc in results:
                            results[reviewed_perc] = []

                        results[reviewed_perc].append(execution_result['loops']['pos'][i])

            for step in results:
                results[step] = [np.round(np.min(results[step]) / num_pos, 2), np.round(np.max(results[step]) / num_pos, 2)]

            print(str(filename), str(fea), results)

            with open("../dump/recall_curves/{}_{}.json".format(str(filename), str(fea)), 'w') as handle:
                handle.write(json.dumps(results))


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

    cols = params['trecs']  + ['feature', 'dataset']
    dictdf = { col: [] for col in cols }

    for fea in params['features']:
        for ds in params['dataset_files']:
            fea, ds = str(fea), str(ds)
            dictdf['feature'].append(fea)
            dictdf['dataset'].append(ds.split('_')[0])
            for result in results:
                if result['feature'] == fea and result['dataset'] == ds.replace('.csv', ''):
                    dictdf[result['trec']].append("{} ({})".format(result['median'], result['iqr']))


    pd.DataFrame.from_dict(data=dictdf).to_csv('out.csv')


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
