feature data[download](https://paddlemodels.bj.bcebos.com/video_detection/CTCN_data.tar.gz)

```
data
  |
  |----senet152-201cls-flow-60.9-5seg-331data\_train
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_train
  |----senet152-201cls-flow-60.9-5seg-331data\_val
  |----senet152-201cls-rgb-70.3-5seg-331data\_331img\_val
```

Meanwhile, download Activity1.3\_train\_rgb.listformat, Activity1.3\_val\_rgb.listformat, labels.txt, val\_duration\_frame.list, put in dataset/ctcn/

train:

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export FLAGS_fast_eager_deletion_mode=1
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    python train.py --model_name=CTCN \
                    --config=./configs/ctcn.yaml \
                    --log_interval=10 \
                    --valid_interval=1 \
                    --use_gpu=True \
                    --save_dir=./data/checkpoints \
                    --fix_random_seed=False \
                    --pretrain=$PATH_TO_PRETRAIN_MODEL

    bash run.sh train CTCN ./configs/ctcn.yaml

eval:

    python eval.py --model_name=CTCN \
                   --config=./configs/ctcn.yaml \
                   --log_interval=1 \
                   --weights=$PATH_TO_WEIGHTS \
                   --use_gpu=True

    bash run.sh eval CTCN ./configs/ctcn.yaml

