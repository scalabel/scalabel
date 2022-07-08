import torch

kMinBlockSize = 512


def set_fullname(mod, fullname):
    mod.fullname = fullname
    if len(list(mod.children())) == 0:
        for index, p in enumerate(mod.parameters()):
            p.reserved_name = "%s->p%d" % (fullname, index)
    for child_name, child in mod.named_children():
        child_fullname = "%s->%s" % (fullname, child_name)
        set_fullname(child, child_fullname)


def group_to_shape(group):
    shape_list = []
    param_list = []
    buf_list = []
    mod_list = []

    def travel_layer(mod):
        if len(list(mod.children())) == 0:
            mod_list.append(mod)
        else:
            for child in mod.children():
                travel_layer(child)

    for mod in group:
        travel_layer(mod)

    for mod in mod_list:
        for p in mod.parameters():
            shape_list.append(p.shape)
            param_list.append(p)
    for mod in mod_list:
        for key, buf in mod._buffers.items():
            if buf is not None and buf.dtype is torch.float32:
                shape_list.append(buf.shape)
                buf_list.append((mod, key))
    return shape_list, param_list, buf_list, mod_list


def group_to_batch(group):
    mod_list = []

    def travel_layer(mod):
        if len(list(mod.children())) == 0:
            mod_list.append(mod)
        else:
            for child in mod.children():
                travel_layer(child)

    def pad(t, blockSize):
        length = t.numel()
        size = length * t.element_size()
        padded_size = blockSize * ((size + blockSize - 1) // blockSize)
        padded_length = padded_size // t.element_size()
        t_padded = torch.zeros(padded_length)
        t_padded[:length] = t
        return t_padded

    for mod in group:
        travel_layer(mod)
    param_tensor_list = [
        pad(p.view(-1), kMinBlockSize)
        for mod in mod_list
        for p in mod.parameters()
    ]
    buffer_tensor_list = [
        pad(buf.view(-1), kMinBlockSize)
        for mod in mod_list
        for _, buf in mod._buffers.items()
        if buf is not None and buf.dtype is torch.float32
    ]
    if (len(param_tensor_list) + len(buffer_tensor_list)) > 0:
        tensor_list = param_tensor_list + buffer_tensor_list
        batched_tensor = torch.cat(tensor_list)
    else:
        batched_tensor = None
    modname_list = [
        mod.fullname if hasattr(mod, "fullname") else None for mod in mod_list
    ]
    return batched_tensor, modname_list
