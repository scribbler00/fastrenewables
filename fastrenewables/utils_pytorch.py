# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00f_utils_pytorch.ipynb (unless otherwise specified).

__all__ = ['unfreeze_n_final_layer', 'freeze', 'unfreeze', 'print_requires_grad']

# Cell
from torch import nn
from torch.nn import Embedding

# Cell
def unfreeze_n_final_layer(model, n, include_embedding=False):
    """
    Remove all but the last 'n' layers from the gradient computation.

    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model whose layers are to be excluded from the gradient computation.
    n : interger
        the number of layers not to be included for gradient computation.
    include_embedding : bool
        if True, include all embedding layers to the gradient computation.

    Returns
    -------

    Notes
    -----
    Currently embedding layers are either included or excluded through 'include_embedding'.
    """
    # freeze all parameters by excluding them from gradient computation
    for param in model.parameters():
        param.requires_grad = False

    # Reinclude the parameters of the last n layers to gradient computation
    layers = list(model.children())

    new_layers = []
    for l in layers:
        if type(l) is nn.ModuleList:
            unfreeze_n_final_layer(l, n, include_embedding=include_embedding)
        elif type(l) is Embedding and include_embedding:
            for param in l.parameters():
                param.requires_grad = True
        elif type(l) is Embedding and not include_embedding:
            for param in l.parameters():
                param.requires_grad = False
        elif hasattr(l, "weight") or isinstance(l, nn.Sequential):
            new_layers.append(l)

    if len(new_layers) > 0:
        layers = new_layers

        if n > len(layers) or n == -1:
            n = len(layers)  # relearn the whole network

        for i in range(1, n + 1):
            for param in layers[-i].parameters():
                param.requires_grad = True

# Cell
def freeze(layer):
    """
    Exclude a layer from the gradient computation.
    Parameters
    ----------
    layer : torch.nn
        the layer which is to be excluded from the gradient computation.

    Returns
    -------

    """
    for p in layer.parameters():
        p.requires_grad = False

# Cell
def unfreeze(layer):
    """
    Include a layer to the gradient computation.
    Parameters
    ----------
    layer : torch.nn
        the layer which is to be included to the gradient computation.

    Returns
    -------

    """
    for p in layer.parameters():
        p.requires_grad = True


# Cell
def print_requires_grad(
    model, include_embedding=True, type_name="", rec_level=0, tabs=""
):
    """
    Print which layers of the model are included in the gradient computation.
    Parameters
    ----------
    model : pytorch.nn.ModuleList/pytorch.nn.Sequential/any
        the model that is to be analyzed.
    include_embedding : bool
        currently not used.
    type_name : string
        currently not used.
    rec_level : integer
        currently not used.
    tabs : string
        the amount of space before each print.

    Returns
    -------

    """
    layers = list(model.children())
    new_rec_level = rec_level + 1

    modules = model._modules
    if isinstance(model, nn.ModuleList):
        cur_type = "ModuleList"
    elif isinstance(model, nn.Sequential):
        cur_type = "Sequential"
    else:
        cur_type = ""
    for k, v in modules.items():
        if len(v._modules) > 0:
            print(f"{tabs}{cur_type} ({k}): (")
            new_tabs = tabs + "  "
            print_requires_grad(v, tabs=new_tabs)
            print(f"{tabs})")
        else:
            if hasattr(v, "weight"):
                print(f"{tabs}({v}) Requires grad: {v.weight.requires_grad}")
            else:
                print(f"{tabs}({v})")