import os, sys, time, random, string, json, numpy, glob, pandas as pd
from collections import OrderedDict
sys.path.extend(["./cair", "./cair/main"])
numpy.random.seed(7881)
from cair.main.recommender import run

ReQue = {
    'input': './output/supervised/test',
    'output': './output/supervised/test'
}


def generate_random_string(n=12):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def tsv2json(df, output, topn=1):
    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)

    for a in ['acg', 'seq2seq', 'hredqs']:
        if not os.path.isdir('{}{}'.format(output, a)):
            os.mkdir('{}{}'.format(output, a))

    with open('{}dataset.json'.format(output), 'w') as fds, \
            open('{}train.json'.format(output), 'w') as ftrain, \
            open('{}dev.json'.format(output), 'w') as fdev, \
            open('{}test.json'.format(output), 'w') as ftest:
        for idx, row in df.iterrows():
            if pd.isna(row.query):
                continue
            qObj = OrderedDict([
                ('id', generate_random_string(12)),
                ('text', row.query),
                ('tokens', row.query.split()),
                ('type', ''),
                ('candidates', [])
            ])
            for i in range(1, topn + 1):
                session_queries = []
                session_queries.append(qObj)
                qcol = 'query_'
                if (qcol not in df.columns) or pd.isna(row[qcol]):
                    break
                # check if the query string is a dict (for weighted expanders such as onfields)
                try:
                    row[qcol] = ' '.join(eval(row[qcol]).keys())
                except:
                    pass

                q_Obj = OrderedDict([
                    ('id', generate_random_string(12)),
                    ('text', row[qcol]),
                    ('tokens', row[qcol].split()),
                    ('type', ''),
                    ('candidates', [])
                ])
                session_queries.append(q_Obj)

                obj = OrderedDict([
                    ('session_id', generate_random_string()),
                    ('query', session_queries)
                ])
                print(f'{hex_to_ansi("#BB8FCE")}{row.qid}{hex_to_ansi(reset=True)}: {qObj["text"]} -- {str(i)} --> {hex_to_ansi("#52BE80")}{q_Obj["text"]}{hex_to_ansi(reset=True)}')

                fds.write(json.dumps(obj) + '\n')

                choice = numpy.random.choice(3, 1, p=[0.9, 0.05, 0.05])[0]
                if choice == 0:
                    ftrain.write(json.dumps(obj) + '\n')
                elif choice == 1:
                    fdev.write(json.dumps(obj) + '\n')
                else:
                    ftest.write(json.dumps(obj) + '\n')


def call_cair_run(data_dir, epochs):
    dataset_name = 'msmarco'  # it is hard code in the library. Do not touch! :))
    baseline_path = 'cair/'
    cli_cmd = ''  # 'python '
    cli_cmd += '{}main/recommender.py '.format(baseline_path)
    cli_cmd += '--dataset_name {} '.format(dataset_name)
    cli_cmd += '--test_file {} '.format('dataset.json')
    cli_cmd += '--data_dir {} '.format(data_dir)
    cli_cmd += '--only_test True '

    cli_cmd += '--max_query_len 1000 '
    cli_cmd += '--uncase True '
    cli_cmd += '--batch_size 8 '
    cli_cmd += '--test_batch_size 8 '
    cli_cmd += '--data_workers 40 '
    cli_cmd += '--emsize 300 '
    cli_cmd += '--embed_dir {}data/fasttext/ '.format(baseline_path)
    cli_cmd += '--embedding_file crawl-300d-2M-subword.vec '

    # the models config are in QueStion\qs\cair\neuroir\hyparam.py
    # only hredqs can be unidirectional! all other models are in bidirectional mode
    df = pd.DataFrame(columns=['model', 'epoch', 'rouge', 'bleu', 'bleu_list', 'exact_match', 'f1', 'elapsed_time'])
    for baseline in ['acg', 'seq2seq', 'hredqs']:
        for epoch in epochs:
            print(f'Running for {hex_to_ansi("#3498DB")}{epoch}{hex_to_ansi(reset=True)} epochs and model {hex_to_ansi("#3498DB")}{baseline}{hex_to_ansi(reset=True)}')
            start_time = time.time()
            test_resutls = run((cli_cmd + '--model_dir {}/{} --model_name {}.e{} --model_type {} --num_epochs {}'.format(data_dir, baseline, baseline, epoch, baseline, epoch)).split())
            elapsed_time = time.time() - start_time
            df.loc[len(df)] = [baseline, epoch, test_resutls['rouge'], test_resutls['bleu'], ','.join([str(b) for b in test_resutls['bleu_list']]), test_resutls['em'], test_resutls['f1'], elapsed_time]
            df.to_csv('{}/results.csv'.format(data_dir, baseline), index=False)


def hex_to_ansi(hex_color_code="", reset=False):
    #TODO: create a utlis for customizaion
    if reset: return "\033[0m"
    hex_color_code = hex_color_code.lstrip('#')
    red = int(hex_color_code[0:2], 16)
    green = int(hex_color_code[2:4], 16)
    blue = int(hex_color_code[4:6], 16)
    return f'\033[38;2;{red};{green};{blue}m'

def aggregate(path):
    fs = glob.glob(path + "/**/results.csv", recursive=True)
    print(fs)
    df = pd.DataFrame(columns=['topics', 'topn', 'ranker', 'model', 'epoch', 'rouge', 'bleu', 'bleu_list', 'exact_match', 'f1', 'elapsed_time'])
    for f in fs:
        df_f = pd.read_csv(f, header=0)
        f = f.replace(path, '').split(os.path.sep)
        ds = f[-3].split('.')[0]
        topn = 1
        ranker = f[-2].split('.')[0]
        for idx, row in df_f.iterrows():
            df_f.loc[idx, 'topics'] = ds
            df_f.loc[idx, 'topn'] = topn
            df_f.loc[idx, 'ranker'] = ranker
        df = pd.concat([df, df_f], ignore_index=True)

    df.to_csv(path + "agg_results.csv", index=False)


# {CUDA_VISIBLE_DEVICES={zero-base gpu indexes, comma seprated reverse to the system}} python -u main.py {topn=[1,2,...]} {topics=[robust04, gov2, clueweb09b, clueweb12b13, all]} 2>&1 | tee log &
# CUDA_VISIBLE_DEVICES=0,1 python -u main.py 1 robust04 2>&1 | tee robust04.topn1.log &


if __name__ == '__main__':
    topn = 1
    corpora = ['robust04', 'gov2', 'antique']
    if not corpora:
        corpora = ['aol-ia']
    if not topn:
        topn = 1
    refiner = 'bt'
    rankers = ['bm25', 'qld']
    metrics = ['map']
    for corpus in corpora:
        for ranker in rankers:
            ranker = ranker.replace(' ', '.')
            for metric in metrics:
                data_dir = f'{ReQue["input"]}/{corpus}/{ranker}.{metric}.{refiner}/'
                print(f'Calling cair for {hex_to_ansi("#3498DB")}{data_dir}{hex_to_ansi(reset=True)}')
                call_cair_run(data_dir, epochs=[100])

    aggregate(ReQue['output'] + '/')
