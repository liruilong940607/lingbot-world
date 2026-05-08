# GB200
cd ../flash-attention/
uv pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
cd -

uv pip install omegaconf imageio[ffmpeg] easydict lmdb diffusers
uv run --no-project \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    generate_fast.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir lingbot-world-base-cam \
    --image examples/00/image.jpg \
    --action_path examples/00 \
    --ulysses_size 1 \
    --frame_num 237 \
    --base_seed 44 \
    --offload_model False \
    --prompt "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls."

# GB200 DiT
# > Time taken: 1.6597743034362793 seconds for chunk 16. Generated x0 shape: torch.Size([16, 3, 58, 104])
# > Time taken: 1.7009646892547607 seconds for chunk 17. Generated x0 shape: torch.Size([16, 3, 58, 104])
# > Time taken: 1.74271559715271 seconds for chunk 18. Generated x0 shape: torch.Size([16, 3, 58, 104])
# > Time taken: 1.784914493560791 seconds for chunk 19. Generated x0 shape: torch.Size([16, 3, 58, 104])
