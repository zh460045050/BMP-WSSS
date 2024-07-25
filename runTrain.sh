######Training
port=29552
crop_size=512
file=train_voc.py
config=configs/voc_attn_reg.yaml
save_dir=train_logs/debug

CUDA_VISIBLE_DEVICES=0,4 python -m torch.distributed.launch --nproc_per_node=2 \
                                                            --master_port=$port $file \
                                                            --config $config \
                                                            --pooling gmp \
                                                            --crop_size $crop_size \
                                                            --work_dir $save_dir \
                                                            --check_name "debug" \
                                                            --lambda_1 0.7 \
                                                            --lambda_2 0.1 \
                                                            --lambda_3 0.1 \
                                                            --warm_iter_s2c 2000 \
                                                            --propogation "BSP" \
                                                            --warm_iter_bsp 4000
                                            