import argparse
import collections

import torch
import numpy as np

from trainer import Trainer
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
import metric.metrics as module_metric
from utils import prepare_device, update_lr_scheduler


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(config: ConfigParser):
    logger = config.get_logger('train')

    config['data_loader']['args']['training'] = True

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    config = update_lr_scheduler(config, len(data_loader))

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch,
                            num_classes=config['data_loader']['args']['num_classes'])
    logger.info(model)
    # only use config["checkpoint"] to train from epoch 1 if config.resume is None
    if config['checkpoint'] and config.resume is None:
        logger.info(f'Starting training with base checkpoint: {config["checkpoint"]} ...')
        checkpoint = torch.load(config["checkpoint"])
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and train_metrics
    criterion = config.init_ftn('loss', module_loss)
    train_metrics = [getattr(module_metric, met)
                     for met in config['train_metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, train_metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      amp=config['amp'])

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-cfg', '--config', default="config/train_feat_clsf_fcn.json", type=str,
                      help='config file path (default: %(default)s)')
    args.add_argument('-rp', '--resume', default=None, type=str,
                      help='path to latest checkpoint. Takes precedence over config["checkpoint"] (default: %(default)s)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: %(default)s)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    train(config)
