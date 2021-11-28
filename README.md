# Face Age and Gender Training

Support for training age and gender predictions given feature vectors of faces.

## Setup

Set up a virtual environment or a conda env.

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Hyperparameter Tuning

```shell
$ python hp_tuning.py -c config/HP_TUNING_CONFIG_FILE.json
```

## Training

**Configuration**: Set the correct parameters in the apporpriate `config/TRAIN_CONFIG_FILE.json`

```shell
$ python train.py -c config/TRAIN_CONFIG_FILE.json -d CUDA_DEVICE
```

## Testing

```shell
$ python test.py -c config/TRAIN_CONFIG_FILE.json -r PATH_TO_MODE_PT_FILE -d CUDA_DEVICE
```
