import os
import torch


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print, cycle=None):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, f'{model_name}.pth'))

        if cycle is not None:
            torch.save(obj=model, f=os.path.join(model_dir, f'{model_name}_cycle_{cycle}.pth'))
