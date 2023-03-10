# Simple Logo Search

Simple Logo Search with faiss(facebook)


## Install Faiss
``` shell
# CPU-only version  
$ conda install -c pytorch faiss-cpu

# GPU(+CPU) version  
$ conda install -c pytorch faiss-gpu
```

TODO: use yolo onnx for inference.


## Models
- image search: resnet, fingerprinting
- text search: ibot, mocov3, CLIP

# TODO:
- for each feature/model, make it a service, so that the main application can access.
- enable multiple models being used simultaneously.
