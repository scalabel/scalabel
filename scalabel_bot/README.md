# PipeSwitch
This folder contains code for PipeSwitch.

# Content
- main.py: Part of the controller and part of the memory management. It accepts connections from the client. It request the entrie GPU memory for the memory management. Besides, it creates threads and processes for other components.
- frontend_tcp.py: After main.py accepts connections from the client, this thread receives requests and send replies.
- frontend_schedule.py: Part of the controller and part of the memory management. It instructs workers to activate or deactivate. It transfers model parameters to GPU.
- worker.py: The workers of PipeSwitch. It is responsible for executing requests. Besides, it has some functions to handle preemption and pipeline, and cooperate with the memory management.
- worker_common.py: An interface to manage models. It loads models structures into the CPU. It add hooks to layers to wait for the parameters for the pipeline. It also executes models.
- worker_terminate.py: Listen to the controller for signals for deactivate. Then it notifies the main thread of the worker to stop the current task.

# Environment
To support PipeSwitch, we add some plugins into PyTorch. Thus, before running this demo, you need to compile PyTorch as stated in [PyTorch plugin](https://github.com/netx-repo/PipeSwitch/tree/main/pytorch_plugin).
Besides, you need to add the path to the repo to `PYTHONPATH`.

# Usage
```
python main.py model_list.txt
```
After starting the program, you need to wait for several seconds to allow it load required models.