from typing import Union
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import pickle

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


class NotAvailable(object):
    @classmethod
    def it_is(cls, obj):
        return isinstance(obj, cls)


@dataclass
class TrainingState(object):
    step: int
    epoch: int
    model_state_dict: Union[dict, NotAvailable]
    losses_state_dict: Union[dict, NotAvailable]
    optims_state_dict: Union[dict, NotAvailable]
    schedulers_state_dict: Union[dict, NotAvailable]
    misc_attributes: dict = field(default_factory=dict)

    def serialize(self, path=None):
        states = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "losses_state_dict": self.losses_state_dict,
            "optims_state_dict": self.optims_state_dict,
            "schedulers_state_dict": self.schedulers_state_dict,
        }
        if not path:
            return pickle.dumps(states)
        if xm:
            xm.save(states, path)
        else:
            torch.save(states, path)
        return self

    @classmethod
    def deserialize(cls, path):
        states = torch.load(path,)
        return cls(
            step=states["step"],
            epoch=states["epoch"],
            model_state_dict=states["model_state_dict"],
            losses_state_dict=states["losses_state_dict"],
            optims_state_dict=states["optims_state_dict"],
            schedulers_state_dict=states["schedulers_state_dict"],
        )

    @classmethod
    def initial_state(cls, step=0, epoch=0):
        return cls(
            step=step,
            epoch=epoch,
            model_state_dict=NotAvailable(),
            losses_state_dict=NotAvailable(),
            optims_state_dict=NotAvailable(),
            schedulers_state_dict=NotAvailable(),
        )

    def load_in_model(self, model: nn.Module):
        if NotAvailable.it_is(self.model_state_dict):
            return model
        model.load_state_dict(self.model_state_dict)
        return model

    def load_in_losses(self, **losses):
        if NotAvailable.it_is(self.losses_state_dict):
            return losses
        for key, loss in losses.items():
            if loss is not None:
                loss.load_state_dict(self.losses_state_dict[key])
        return losses

    def load_in_optims(self, **optims):
        if NotAvailable.it_is(self.optims_state_dict):
            return optims
        for key, optim in optims.items():
            if optim is not None:
                optim.load_state_dict(self.optims_state_dict[key])
        return optims

    def load_in_schedulers(self, **schedulers):
        if NotAvailable.it_is(self.schedulers_state_dict):
            return schedulers
        for key, scheduler in schedulers.items():
            if scheduler is not None:
                scheduler.load_state_dict(self.schedulers_state_dict[key])
        return schedulers

    @classmethod
    def is_instance(cls, obj):
        return isinstance(obj, cls) or (obj.__class__.__name__ == cls.__name__)
