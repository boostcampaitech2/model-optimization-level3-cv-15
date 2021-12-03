"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info

import shutil
import wandb

def get_optimizer(optimizer_option, model, learning_rate):
    optimizer = None

    if optimizer_option.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_option.lower() == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif optimizer_option.lower() == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_option.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_option.lower() == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimizer_option.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    elif optimizer_option.lower() == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)
    elif optimizer_option.lower() == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    return optimizer

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:

    parameters = model_config
    parameters.update(data_config)
    print(parameters)
    wandb.config = parameters
    
    wandb.init(project="lightweight", entity="falling90")
    wandb.run.name = "Final_lw_KJY"
    wandb.save()

    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    # optimizer = torch.optim.SGD(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    optimizer = get_optimizer(data_config["OPTIMIZER"], model_instance.model, learning_rate=data_config["INIT_LR"])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--optim", default="adamax", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    data_config["OPTIMIZER"] = args.optim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir):
        if os.path.exists(log_dir + '/best.pt'):
            modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
            new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
            os.rename(log_dir, new_log_dir)
        else:
            shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )