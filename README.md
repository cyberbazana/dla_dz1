# Automatic Speech Recognition (ASR) with PyTorch

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```


   ```bash
   pip install pyctcdecode==0.5.0 --no-deps
   ```

2. Download all required files

   ```bash
   !python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='cyberbazana/dz1_dla', filename='model_best.pth', local_dir='./saved/finetune_360_longer')"
   ```

   File for LM is automatically dowloaded by `inference.py` with `inference.yaml`.


## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments. The structure of the project is quite complicated.

The original model was trained with `greedy+aug.yaml` + `finetune.yaml`. All logs can be checked in the wandb report and in this repo.

If you want to change augmentations, then open `src/configs/transforms/aug.yaml`. All possible and valid augmentations are there (PeakNormalize is not used), so you can play with params.

If you want to change training params (lr, optimizer, n_epochs), all of them are in `greedy+aug.yaml` and `finetune.yaml` (you can create your own one, this one is just an original one for best accuracy).

Unfortunately, you can't change a dataset for training. There are only librispeech type ones and Common voice.

If you want to use a different model, then create a `.yaml` file in `src/configs/model`, there are some examples already.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

If you want to run inference on your dataset, then check a demo, this point is thoroughly covered there.

To run inference on a custom dataset use this script.

```bash
!python inference.py -cn=inference inference.save_path="PATH" datasets=custom_dataset datasets.inference.data_dir=path/to/dataset"
```

Data will be stored in `/data/saved/TEST/inference`.

If you want to run metrics, then use this script.

```bash
!python calc_WER_CER.py --target path/to/dataset/transcriptions --predictions ./data/saved/PATH/inference
```

If the validation is too slow, change a config in `src/configs/dataloader`. The current way of organizing them is quite bad.

WARNING. If you want to use lm_beam, do not forget to change the path in inference.yaml and location of the model. By default, it is ./saved/greedy+aug_ver_article there.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
