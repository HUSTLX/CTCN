    
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

