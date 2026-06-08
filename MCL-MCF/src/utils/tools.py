import torch
import os
import io


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    # name = save_load_name(args, name)
    name = 'best_model'
    save_dir = 'outputs/pre_trained_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/{name}.pt')


def load_model(args, name=''):
    # name = save_load_name(args, name)
    name = 'best_model'
    save_dir = 'outputs/pre_trained_models'
    with open(f'{save_dir}/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t
