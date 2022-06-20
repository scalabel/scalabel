# Task

This folder contains codes for models.
They are based on examples in https://pytorch.org/hub/research-models.

# Content

- ResNet152
    - resnet152.py: Functions for importing sample data and importing the model.
    - resnet152_inference.py: Functions for performing inference.
    - resnet152_training.py: Functions for performing training.
- Inception-v3
    - inception_v3.py: Functions for importing sample data and importing the model.
    - inception_v3_inference.py: Functions for performing inference.
    - inception_v3_training.py: Functions for performing training.
- BERT-base
    - bert_base.py: Functions for importing sample data and importing the model.
    - bert_base_inference.py: Functions for performing inference.
    - bert_base_training.py: Functions for performing training.

# Example
``` Python
from resnet152 import import_data
from resnet152_inference import import_model, import_function
data, images = import_data(8) # Argument is batch size
model = import_model()
model = model.cuda()
func = import_func()
output = func(model, data.numpy().tobytes()) # Output is the sum of the model result, which is used to check correctness.
```