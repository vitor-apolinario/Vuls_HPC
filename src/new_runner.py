# -*- coding: utf-8 -*-
from __future__ import division, print_function

import multiprocessing
import pickle
import json
from collections import Counter

import numpy as np

from mar import MAR
from demos import cmd


def run_target_1():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    arglist = []
    for filename in params['dataset_files']:
        for fea in params['features']:
            for trec in params['trecs']:
                for i in range(30):
                    arglist.append((str(fea), i, str(filename), trec))

    pool = multiprocessing.Pool()
    pool.map(error_hpcc_feature_ds_wrapper, arglist)
    pool.close()


def run_missing_target_1():
    try:
        with open('./params.json') as json_file:
            params = json.load(json_file)
    except:
        raise Exception('wrong params file path')

    arglist = []
    for ds in params['dataset_files']:
        ds = str(ds).replace('.csv', '')

        with open("../memory/rerun_params_{}.pickle".format(ds), "r") as handle:
            rerun = pickle.load(handle)
            # print(rerun)

            for execution in rerun:
                arglist.append((execution['fea'], int(execution['seed']), execution['filename'],
                                float(execution['trec'])))

    pool = multiprocessing.Pool()
    pool.map(error_hpcc_feature_ds_wrapper, arglist)
    pool.close()


def error_hpcc_feature_ds_wrapper(args):
    error_hpcc_feature_ds(*args)


def error_hpcc_feature_ds(fea, seed=1, filename='drupal_combine.csv', trec=0.95):
    np.random.seed(int(seed))

    strec = '@' + str(int(trec * 100))
    round_id = "hpcc_{}_{}_{}_{}".format(filename.split(".")[0], fea, strec, seed)

    if fea == 'combine':
        read = Combine(filename=filename, trec=trec, seed=seed, round_id=round_id)
    elif fea == 'text':
        read = Text(filename=filename, trec=trec, seed=seed, round_id=round_id)
    elif fea == 'crash':
        read = CRASH(filename=filename, trec=trec, seed=seed, round_id=round_id)
    elif fea == 'random':
        read = Rand(filename=filename, trec=trec, seed=seed, round_id=round_id)
    else:
        raise Exception('wrong feature provided')

    execution_results = {'loops': read.record, 'stats': read.results}
    read.results["reached"] = True

    target = int((read.results["truepos"] + read.results["unknownyes"]) * trec)
    vul_found = read.results["truepos"]

    if vul_found < target:
        read.results["reached"] = False

    with open("../dump/features_" + round_id + ".pickle", "w") as handle:
        pickle.dump(execution_results, handle)


def Combine(filename='vuls_data_new.csv', trec=0.95, seed=0, round_id='@unknow'):
    thres = 0
    starting = 1
    np.random.seed(seed)

    read = MAR()
    read.step = 10
    read.roundname = round_id
    read.correction = 'no'
    read.crash = 'append'
    read = read.create(filename, 'all')

    read.interval = 100000

    num2 = read.get_allpos()
    target = int(num2 * trec)

    read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        # print(pos, pos+neg)

        if pos + neg >= total:
            break

        if pos < starting or pos + neg < thres:
            for id in read.BM25_get():
                read.code_error(id, error='none')
        else:
            a, b, c, d = read.train(weighting=True, pne=True)

            if pos >= target:
                break

            if pos < 10:
                for id in a:
                    read.code_error(id, error='none')
            else:
                for id in c:
                    read.code_error(id, error='none')
    read.results = analyze(read)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    return read


def Text(filename='vuls_data_new.csv', seed=0, trec=0.95, round_id='@unknow'):
    thres = 0
    starting = 1
    np.random.seed(seed)
    read = MAR()
    read.roundname = round_id
    read.correction = 'no'
    read = read.create(filename, 'all')
    read.interval = 100000

    num2 = read.get_allpos()
    target = int(num2 * trec)

    read.enable_est = False
    read.step = 10

    while True:
        pos, neg, total = read.get_numbers()

        if pos + neg >= total:
            break

        if pos < starting or pos + neg < thres:
            for id in read.random():
                read.code_error(id, error='none')
        else:
            a, b, c, d = read.train(weighting=True, pne=True)

            if pos >= target:
                break

            if pos < 10:
                for id in a:
                    read.code_error(id, error='none')
            else:
                for id in c:
                    read.code_error(id, error='none')

    read.results = analyze(read)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    return read


def Rand(filename='vuls_data_new.csv', seed=0, trec=0.95, round_id='@unknow'):
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename, 'all')
    read.interval = 100000
    read.step = 10
    read.roundname = round_id
    read.enable_est = False

    num2 = read.get_allpos()
    target = int(num2 * trec)

    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total or pos >= target:
            break

        for id in read.random():
            read.code_error(id, error='none')

    read.results = analyze(read)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    return read


def CRASH(filename='vuls_data_new.csv', trec=0.95, seed=0, round_id='@unknow'):
    starting = 1
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename, 'all')
    thres = Counter(read.body.crashes > 0)[True]
    read.interval = 100000
    read.roundname = round_id

    num2 = read.get_allpos()
    target = int(num2 * trec)

    read.enable_est = False
    read.step = 10

    while True:
        pos, neg, total = read.get_numbers()
        # print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        # todo: confirm condition
        if (pos < starting or pos + neg < thres) and pos < target:
            for id in read.BM25_get():
                read.code_error(id, error='none')
        else:
            break
            # if pos >= target:
            # break

    read.results = analyze(read)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))

    # todo: understand why
    result = {'est': read.record_est, 'pos': read.record}

    return read


def analyze(read):
    unknown = np.where(np.array(read.body['code']) == "undetermined")[0]
    pos = np.where(np.array(read.body['code']) == "yes")[0]
    neg = np.where(np.array(read.body['code']) == "no")[0]
    yes = np.where(np.array(read.body['label']) == "yes")[0]
    no = np.where(np.array(read.body['label']) == "no")[0]
    falsepos = len(set(pos) & set(no))
    truepos = len(set(pos) & set(yes))
    falseneg = len(set(neg) & set(yes))
    unknownyes = len(set(unknown) & set(yes))
    unique = len(read.body['code']) - len(unknown)
    count = sum(read.body['count'])
    correction = read.correction
    return {"falsepos": falsepos, "truepos": truepos, "falseneg": falseneg, "unknownyes": unknownyes, "unique": unique,
            "count": count, "correction": correction, "files": len(read.body['code'])}


if __name__ == "__main__":
    eval(cmd())