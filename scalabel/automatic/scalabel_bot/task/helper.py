import importlib


def get_data(model_name, batch_size):
    model_module = importlib.import_module("pipeswitch.task.%s" % (model_name))
    data, _ = model_module.import_data(batch_size)
    return data


def get_model(model_name):
    model_module = importlib.import_module(
        "pipeswitch.task.%s_inference" % (model_name)
    )
    model = model_module.import_model()
    func = model_module.import_func()
    return model, func
