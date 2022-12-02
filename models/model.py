import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics.functional as tmf

from models.gcn import MolGCN
from models.gat import MolGAT
from models.attentiveFP import MolAttentiveFP

NODE_INPUT_SIZE = 19


class Model(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 weight_decay: float,
                 loss: nn.Module,
                 model_name: str):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        if model_name == 'MolGCN':
            self.model = MolGCN(gcn_in_channels=NODE_INPUT_SIZE,
                                gcn_hidden_channels=32,
                                gcn_num_layers=6,
                                gcn_out_channels=64,
                                mlp_channel_list=[64, 48, 32, 16, 8, 1])
        elif model_name == 'MolGAT':
            self.model = MolGAT(gcn_in_channels=NODE_INPUT_SIZE,
                                gcn_hidden_channels=32,
                                gcn_num_layers=6,
                                gcn_out_channels=64,
                                mlp_channel_list=[64, 48, 32, 16, 8, 1])
        elif model_name == 'MolAttentiveFP':
            self.model = MolAttentiveFP(afp_in_channels=NODE_INPUT_SIZE,
                                        afp_hidden_channels=32,
                                        afp_num_layers=6,
                                        afp_out_channels=1,
                                        afp_num_timesteps=4)
        self.loss_funct = loss
        self.lr = lr
        self.weight_decay = weight_decay
        print(self.model)

    def get_nb_parameters(self, only_trainable: bool = False):
        nb_params = 0
        if only_trainable:
            nb_params += sum(p.numel()
                             for p in self.model.parameters() if p.requires_grad)
        else:
            nb_params += sum(p.numel() for p in self.model.parameters())
        return nb_params

    def forward(self, x, edge_index, batch=None):
        y = self.model(x, edge_index, batch)
        return y

    def _common_step(self, batch, batch_idx, stage):
        # batch_size = batch.ptr.size()[0]-1
        y_pred = self(batch.x, batch.edge_index, batch.batch)
        loss = self.loss_funct(y_pred.view(-1), batch.y)
        #self.log("step/{}_loss".format(stage), loss, batch_size=batch_size)
        return loss, y_pred.view(-1), batch.y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx, 'train')
        return {'loss': loss, 'train_preds': preds.detach(),
                'train_targets': targets.detach()}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx, 'val')
        return {'val_loss': loss, 'val_preds': preds,
                'val_targets': targets}

    def common_epoch_end(self, outputs, stage):
        loss_name = 'loss' if stage == 'train' else "{}_loss".format(stage)
        loss_batched = torch.stack([x[loss_name] for x in outputs])
        set_size = loss_batched.size()[0]
        avg_loss = loss_batched.mean()
        all_preds = torch.concat([x["{}_preds".format(stage)]
                                  for x in outputs])
        all_targets = torch.concat([x["{}_targets".format(stage)]
                                    for x in outputs])
        r2 = tmf.r2_score(all_preds, all_targets)
        pearson = tmf.pearson_corrcoef(all_preds, all_targets)
        log_dict = {
            "ep_end/{}_loss".format(stage): avg_loss,
            "ep_end/{}_r2_score".format(stage): r2,
            "ep_end/{}_pearson".format(stage): pearson
        }
        self.log_dict(log_dict, sync_dist=True)

    def validation_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'val')

    def training_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'train')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        return {"optimizer": optimizer}
