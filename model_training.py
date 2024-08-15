from super_gradients.training import models, Trainer, training_hyperparams
from super_gradients.training.metrics.classification_metrics import Accuracy
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
import os

def train_model(train_dataloader, valid_dataloader, class_names):
    efficientnet_training_params = training_hyperparams.get('training_hyperparams/imagenet_efficientnet_train_params')
    
    early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="Accuracy", mode="max", patience=20, verbose=False)
    early_stop_val_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="CrossEntropyLoss", mode="min", patience=20, verbose=False)

    efficientnet_training_params["train_metrics_list"] = [Accuracy()]
    efficientnet_training_params["valid_metrics_list"] = [Accuracy()]
    efficientnet_training_params["phase_callbacks"] = [early_stop_acc, early_stop_val_loss]
    efficientnet_training_params["silent_mode"] = True
    efficientnet_training_params['ema'] = False
    efficientnet_training_params['zero_weight_decay_on_bias_and_bn'] = False
    efficientnet_training_params["criterion_params"] = {'smooth_eps': 0.25}
    efficientnet_training_params["max_epochs"] = 300
    efficientnet_training_params["initial_lr"] = 0.0001

    efficientnet_full_model = models.get(model_name='efficientnet_b0', num_classes=len(class_names), pretrained_weights='imagenet')
    full_model_trainer = Trainer(experiment_name="0_Baseline_Experiment", ckpt_root_dir='checkpoints')

    full_model_trainer.train(
        model=efficientnet_full_model, 
        training_params=efficientnet_training_params, 
        train_loader=train_dataloader,
        valid_loader=valid_dataloader
    )

    best_full_model = models.get(
        'efficientnet_b0',
        num_classes=len(class_names),
        checkpoint_path=os.path.join(full_model_trainer.checkpoints_dir_path, "ckpt_best.pth")
    )

    return best_full_model, full_model_trainer
