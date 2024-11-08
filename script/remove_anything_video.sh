python remove_anything_video.py \
    --input_video ./example/remove-anything-video/motorcycle/example.mp4 \
    --coords_type key_in \
    --point_coords 1250 550 --point_labels 1 \
    --dilate_kernel_size 30 \
    --output_dir ./results \
    --sam_model_type "vit_t" \
    --sam_ckpt ./weights/mobile_sam.pt \
    --lama_config lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt ./pretrained_models/sttn.pth \
    --mask_idx 2 \
    --fps 25
