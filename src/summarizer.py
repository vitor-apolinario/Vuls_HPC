from __future__ import division, print_function

import pickle
import json

import numpy as np
from demos import cmd


def run_target_1_summary():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    res = []
    for filename in params['dataset_files']:
        for fea in params['features']:
            for trec in params['trecs']:
                res.append(run_summary(filename=str(filename).split(".")[0], fea=str(fea), trec=trec))

    for line in res:
        print("{}, {}, {}, {} ({})".format(line['feature'], line['dataset'], line['trec'], line['median'], line['iqr']))


def run_summary(filename=None, fea=None, trec=None):
    if filename is None or fea is None or trec is None:
        raise Exception("invalid params run_summary")

    import glob
    files = glob.glob(
        "/home/vitor-apolinario/Desktop/harmless/Vuls_HPC/dump/features_hpcc_{}_{}*.pickle".format(filename, fea))

    costs = []

    for f in files:
        try:
            with open(f, "r") as handle:
                results = pickle.load(handle)
                costs.append(results[str(trec)]['stats']['unique'] / results[str(trec)]['stats']['files'])
                # print(results[str(trec)]['stats'])
        except:
            raise Exception("unable to summaryze {}".format(f))
            pass

    median = int(np.median(costs) * 100)
    iqr = int((np.percentile(costs, 75) - np.percentile(costs, 25)) * 100)

    return {"dataset": filename.split("_")[0], "feature": fea, "trec": trec, "median": median, "iqr": iqr}


def check_missing_results():
    dataset_files = ['mozilla_cla']
    features = ['combine', 'text', 'random']
    trecs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

    rerun_params = []

    for ds_filename in dataset_files:
        for fea in features:
            for trec in trecs:
                for seed in range(30):
                    try:
                        result_file_path = "/home/vitor-apolinario/Desktop/harmless/Vuls_HPC" \
                                           "/dump/features_hpcc_{}_{}_{}.pickle".format(ds_filename, fea, seed)

                        with open(result_file_path, "r") as handle:
                            results = pickle.load(handle)
                            cost = (float(results[str(trec)]['stats']['unique']) / float(
                                results[str(trec)]['stats']['files']))
                    except:
                        try:
                            raw_filename = result_file_path.split('/')[-1].replace('.pickle', '')
                            seed = raw_filename.split('_')[-1]
                            run = {'fea': fea, 'seed': seed, 'filename': "{}.csv".format(ds_filename), 'trec': trec}
                            rerun_params.append(run)
                        except:
                            raise Exception("Unable to check {} {}".format(result_file_path, trec))

        rerun_params = sorted(rerun_params, key = lambda r: (r['filename'], r['fea'], r['trec'], r['seed']))

        for x in rerun_params:
            print(x)
        print(len(rerun_params))

        with open("../memory/rerun_params_{}.pickle".format(ds_filename), "w") as handle:
            pickle.dump(rerun_params, handle)


if __name__ == "__main__":
    eval(cmd())
