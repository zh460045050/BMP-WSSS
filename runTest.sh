CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
                                                            --master_port=29520 "test_voc.py" \
                                                            --config "configs/voc_attn_reg.yaml" \
                                                            --work_dir "test_logs/debug" \
                                                            --model_path "bmp_checkpoint.pth" \
                                                            --check_name "debug" \
                                                            --eval_set "val"
