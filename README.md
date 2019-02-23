# Research 2019

This repo contains an implementation of the [GLoMo paper](https://arxiv.org/pdf/1806.05662.pdf)
--------------------------------------------------------------------------------------------

### Language Model
1. Wikipedia data preprocessing, run:
```
python glomo/preprocess.py -i [input text files] -o [output_data_directory]
```

2. Train a language model, run:
```
python glomo/train.py glomo [data_directory] -c [config_file.json] -s [model_saving_directory]
```
If running on CPU, add `--cpu`
If resume from previous training, add `-r [model checkpoint path]`

### Use pretrained LM model for downstreaming task.
1. Train a classifier on imdb dataset.
- IMDB Data preprocess with GLoVe embedding:
 ```
 python glomo/imdb_preprocess.py -embd [data_directory]
```

- Train the classifier
 ```
 python glomo/downstream_train.py imdb [data_directory] -c [config_file.json] -s [model_saving_directory] -g [path to pretrained language model]
 ```

- Evaluate the classifier
```
python glomo/eval.py imdb [model_checkpoint_path] [data_directory]
```

2. Train a classifier on MNLI: More details refer to the submodule README

 ```
 python ESIM/scripts/train/training_mnli.py --config [config_file]
 ```
