import glob
import pickle

import numpy as np
from demos import cmd


def run_target_1():
    dataset_files = ['drupal_combine.csv']
    features = ['combine', 'text', 'random']
    trecs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
    res = []

    for filename in dataset_files:
        for fea in features:
            for trec in trecs:
                res.append(run_summary(filename=filename.split(".")[0], fea=fea, trec=trec))

    for line in res:
        print("{}, {}, {}, {} ({})".format(line['feature'], line['dataset'], line['trec'], line['median'], line['iqr']))


def run_summary(filename='drupal_combine', fea='text', trec=0.95):
    import glob
    files = glob.glob(
        "/home/vitor-apolinario/Desktop/harmless/Vuls_HPC/dump/features_hpcc_{}_{}*.pickle".format(filename, fea))
    print(filename, fea, trec)

    costs = []

    for f in files:
        try:
            with open(f, "r") as handle:
                results = pickle.load(handle)
                costs.append(float(results[str(trec)]['stats']['unique']) / float(results[str(trec)]['stats']['files']))
                # print(results[str(trec)]['stats'])
        except:
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
                            print(run)
                            rerun_params.append(run)
                        except:
                            raise Exception("Unable to check {} {}".format(result_file_path, trec))

        with open("../memory/rerun_params_{}.pickle".format(ds_filename), "w") as handle:
            pickle.dump(rerun_params, handle)


if __name__ == "__main__":
    eval(cmd())
