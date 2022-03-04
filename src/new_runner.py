import multiprocessing
import pickle
from collections import Counter

import numpy as np

from mar import MAR
from demos import cmd


def run_target_1():
    dataset_files = ['drupal_combine.csv']
    features = ['combine', 'text', 'random']
    trecs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

    pool = multiprocessing.Pool()

    arglist = []
    for filename in dataset_files:
        for fea in features:
            for trec in trecs:
                for i in range(30):
                    arglist.append((fea, i, filename, trec))

    pool.map(error_hpcc_feature_ds_wrapper, arglist)
    pool.close()


def run_missing_target1(ds='mozilla_cla'):
    with open("../memory/rerun_params_{}.pickle".format(ds), "r") as handle:
        rerun = pickle.load(handle)
        print(rerun)

    pool = multiprocessing.Pool()

    arglist = []
    for execution in rerun:
        arglist.append((execution['fea'], int(execution['seed']), execution['filename'], float(execution['trec'])))

    pool.map(error_hpcc_feature_ds_wrapper, arglist)
    pool.close()


def error_hpcc_feature_ds_wrapper(args):
    error_hpcc_feature_ds(*args)


def error_hpcc_feature_ds(fea, seed=1, filename='drupal_combine.csv', trec=0.95):
    np.random.seed(int(seed))
    vul_type = 'all'
    results = {}

    round = "hpcc_{}_{}_{}".format(filename.split(".")[0], fea, seed)

    try:
        with open("../dump/features_" + round + ".pickle", "r") as handle:
            results = pickle.load(handle)
    except:
        pass

    strec = ' @' + str(int(trec * 100))
    # print(round+strec)

    # todo: set round in text, crash and random
    if fea == 'combine':
        read = Combine(vul_type, stop='true', seed=seed, filename=filename, trec=trec, round_id=round + strec)
    elif fea == 'text':
        read = Text(vul_type, stop='true', seed=seed, filename=filename, trec=trec, round_id=round + strec)
    elif fea == 'crash':
        read = CRASH(vul_type, stop='true', seed=seed, filename=filename, trec=trec, round_id=round + strec)
    elif fea == 'random':
        read = Rand(vul_type, stop='true', seed=seed, filename=filename, trec=trec, round_id=round + strec)
    else:
        raise Exception('wrong feature provided')

    results[str(trec)] = {'loops': read.record, 'stats': read.results}

    with open("../dump/features_" + round + ".pickle", "w") as handle:
        pickle.dump(results, handle)


def Combine(vul_type, stop='true', error='none', correct='no', interval=100000, seed=0, filename='vuls_data_new.csv',
            trec=0.95, round_id='@unknow'):
    stopat = trec
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read.step = 10
    read.roundname = round_id
    read.correction = correct
    read.crash = 'append'
    read = read.create(filename, vul_type)

    read.interval = interval

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        # print(pos, pos+neg)
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if (stop == 'knee') and error == 'random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.record['x'][read.kneepoint]]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                # part2 = set(seq[read.record['x'][read.kneepoint]:]) & set(
                #     np.where(np.array(read.body['code']) == "yes")[0])
                # for id in part1 | part2:
                for id in part1:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos + neg < thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos > 0 and pos_last == pos:
                    counter = counter + 1
                else:
                    counter = 0
                pos_last = pos
                if counter >= 5:
                    break
            elif stop == 'knee':
                if pos > 0:
                    if read.knee():
                        if error == 'random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1 | part2:
                                read.code_error(id, error=error)
                        break
            elif stop == 'true':
                if pos >= target:
                    break
            elif stop == 'mix':
                if pos >= target and stopat * read.est_num <= pos:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    read.results = analyze(read)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    # print(results)
    return read


def Text(vul_type, stop='true', error='none', error_rate=0.5, correct='no', interval=100000, seed=0, neg_len=0.5,
         filename='vuls_data_new.csv', trec=0.95, round_id='@unknow'):
    stopat = trec
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)
    read = MAR()
    read.roundname = round_id

    read.false_neg = float(error_rate)
    read.correction = correct
    read.neg_len = float(neg_len)
    read = read.create(filename, vul_type)

    read.interval = interval

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = False
    read.step = 10

    while True:
        pos, neg, total = read.get_numbers()
        # print(pos)
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if (stop == 'knee') and error == 'random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.record['x'][read.kneepoint]]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                # part2 = set(seq[read.record['x'][read.kneepoint]:]) & set(
                #     np.where(np.array(read.body['code']) == "yes")[0])
                # for id in part1 | part2:
                for id in part1:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos + neg < thres:
            for id in read.random():
                read.code_error(id, error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos > 0 and pos_last == pos:
                    counter = counter + 1
                else:
                    counter = 0
                pos_last = pos
                if counter >= 5:
                    break
            elif stop == 'knee':
                if pos > 0:
                    if read.knee():
                        if error == 'random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            # part2 = set(seq[read.kneepoint * read.step:]) & set(
                            #     np.where(np.array(read.body['code']) == "yes")[0])
                            # for id in part1 | part2:
                            for id in part1:
                                read.code_error(id, error=error)
                        break
            elif stop == 'true':
                if pos >= target:
                    break
            elif stop == 'mix':
                if pos >= target and stopat * read.est_num <= pos:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    # read.export()
    read.results = analyze(read)
    # print(results)
    # print(read.record)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    return read


def Rand(vul_type, stop='true', error='none', interval=100000, seed=0, filename='vuls_data_new.csv', trec=0.95,
         round_id='@unknow'):
    stopat = trec

    np.random.seed(seed)

    read = MAR()
    read = read.create(filename, vul_type)

    read.interval = interval
    read.step = 10
    read.roundname = round_id

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = False

    result = {}
    # result['est'] = {'x':[],'semi':[]}
    result['est'] = {'x': []}
    while True:
        pos, neg, total = read.get_numbers()
        # print(pos)
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total or pos >= target:
            break

        for id in read.random():
            read.code_error(id, error=error)
        if pos + neg > 0:
            result['est']['x'].append(pos + neg)
            # result['est']['semi'].append(float(pos)/(pos+neg)*total)

    result['pos'] = read.record

    read.results = analyze(read)

    return read


def CRASH(vul_type, stop='true', error='none', interval=100000, seed=0, filename='vuls_data_new.csv', trec=0.95,
          round_id='@unknow'):
    stopat = trec
    starting = 1
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename, vul_type)
    thres = Counter(read.body.crashes > 0)[True]
    read.interval = interval
    read.roundname = round_id

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    read.enable_est = False
    read.step = 10

    while True:
        pos, neg, total = read.get_numbers()
        # try:
        #     print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        # except:
        #     print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop == 'knee' and error == 'random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos + neg < thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            break
            # a,b,c,d =read.train(weighting=True,pne=True)
            # if stop == 'est':
            #     if stopat * read.est_num <= pos:
            #         break
            # elif stop == 'soft':
            #     if pos>0 and pos_last==pos:
            #         counter = counter+1
            #     else:
            #         counter=0
            #     pos_last=pos
            #     if counter >=5:
            #         break
            # elif stop == 'knee':
            #     if pos>0:
            #         if read.knee():
            #             if error=='random':
            #                 coded = np.where(np.array(read.body['code']) != "undetermined")[0]
            #                 seq = coded[np.argsort(np.array(read.body['time'])[coded])]
            #                 part1 = set(seq[:read.kneepoint * read.step]) & set(
            #                     np.where(np.array(read.body['code']) == "no")[0])
            #                 part2 = set(seq[read.kneepoint * read.step:]) & set(
            #                     np.where(np.array(read.body['code']) == "yes")[0])
            #                 for id in part1|part2:
            #                     read.code_error(id, error=error)
            #             break
            # elif stop == 'true':
            #     if pos >= target:
            #         break
            # elif stop == 'mix':
            #     if pos >= target and stopat * read.est_num <= pos:
            #         break
            # if pos < read.enough:
            #     for id in a:
            #         read.code_error(id, error=error)
            # else:
            #     for id in c:
            #         read.code_error(id, error=error)
    # read.export()
    read.results = analyze(read)
    print(read.results)
    print(read.roundname, read.results['unique'] / len(read.body["code"]))
    result = {}
    result['est'] = read.record_est
    result['pos'] = read.record
    # with open("../dump/"+type+"_crash.pickle","wb") as handle:
    #     pickle.dump(result,handle)
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