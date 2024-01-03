import os
import json
import argparse



def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default='/scratch365/yding4/ET_project/dataset',
        help="The input data directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='bbn',
        choices=['bbn', 'fewnerd', 'onto'],
    )
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    data_dir = args.data_dir
    dataset = args.dataset

    # input data dir
    dataset_path = os.path.join(data_dir, dataset)
    # output data dir
    processed_dataset_path = os.path.join(data_dir, dataset + '/processed_data')
    os.makedirs(processed_dataset_path, exist_ok=True)
    #  type vocab file
    type_vocab_file_path = os.path.join(dataset_path, f'{dataset}_types.txt')


    # Load and process typing vocabulary
    upper2lower = dict()
    typing_vocab = []
    with open(type_vocab_file_path) as fin:
        for lines in fin:
            lines = lines.rstrip().split(":")
            if len(lines) != 2:
                print(lines)
            assert len(lines) == 2
            upper = lines[0]
            lower = lines[1]
            upper2lower[upper] = lower
            typing_vocab.append(lower)

    typing_vocab = sorted(set(typing_vocab))
    processed_vocab_file_path = os.path.join(processed_dataset_path,'types.txt')
    with open(processed_vocab_file_path, 'w+') as fout:
        fout.write('\n'.join(typing_vocab))

    # process test, dev, train
    for file in ['test', 'dev', 'train']:
        in_file_name = os.path.join(dataset_path, f'{dataset}_{file}.json')
        out_file_name = os.path.join(processed_dataset_path, f'{file}_processed.json')
        # load and process data
        data_lst = []
        idx = 0
        with open(in_file_name) as fin:
            for lines in fin:
                raw_dat = json.loads(lines)
                processed_dat = {}

                entity = raw_dat['word']
                left_tokens = raw_dat['left_context_text']
                right_tokens = raw_dat['right_context_text']
                raw_annotations = raw_dat['y_category']
                annotations = []
                for raw_annotation in raw_annotations:
                    raw_annotation = raw_annotation.rstrip('/')
                    if raw_annotation not in upper2lower:
                        continue
                    cleaned_annotation = upper2lower[raw_annotation]
                    annotations.append(cleaned_annotation)
                annotations = sorted(set(annotations))
                if not annotations:
                    continue
                premise = left_tokens + ' ' + entity + ' ' + right_tokens

                processed_dat['premise'] = premise
                processed_dat['entity'] = entity
                processed_dat['annotation'] = annotations
                processed_dat['id'] = f'{file}{idx:04n}'
                data_lst.append(processed_dat)
                idx += 1

        # save processed path
        with open(out_file_name, 'w+') as fout:
            fout.write('\n'.join([json.dumps(items) for items in data_lst]))


if __name__ == '__main__':
    main()