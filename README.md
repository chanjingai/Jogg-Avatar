# Jogg-Avatar for 720p Video Generation

This repository contains the training and inference code for Jogg-Avatar. This model is a large-scale video generation model based on DIT, which uses audio injection to achieve audio-visual synchronization in the generated video. Our training code supports training with 720p high-definition data and supports inference in multiple languages.For more demonstrations, please visit our official website. https://www.chanjing.cc/ https://www.jogg.ai



https://github.com/user-attachments/assets/becb1b28-890a-4316-9103-1b98411c4f86



## 🚀 Features

- Training pipeline for 720p video generation
- Based on DiffSynth-Studio architecture
- Complete inference scripts and configurations

# 🗓️⏰ Project Timeline

### finished ✅
- 2025-10: training code release

- 2025-11: inference code release

### plan ⏳
- 2025-12: JoggAvatar based on wan2.2-5B release
  

## 📹 Video Generation Results

Our model generates high-fidelity 720p videos with realistic motion and expressions. Below we showcase several examples of our model's capabilities.

### Generation Examples

| Scene | Image | Video1 | Image | Video2 |
|------|------|----------|------|------|
| 人物 | [1.jpg](assets/image/1.jpg) | [📹 ours](assets/1.mp4) | [4.png](assets/image/4.png) | [📹 ours](assets/4.mp4) |
| 卡通 | [2.jpg](assets/image/2.jpg) | [📹 ours](assets/2.mp4) | [6.jpg](assets/image/6.jpg) | [📹 ours](assets/6.mp4) |
| 动物 | [金毛.png](assets/image/金毛.png) | [📹 ours](assets/金毛.mp4) | [5.jpg](assets/image/5.jpg) | [📹 ours](assets/5.mp4) |

## Installation

```bash
git clone https://github.com/chanjingai/Jogg-Avatar.git
cd Jogg-Avatar
pip install -e .
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

## 🖥️ Hardware Requirements

### For Training

| Component | Specification | Notes |
|----------|---------------|-------|
| GPU | 8× NVIDIA A100 80GB | Multi-node training |
| CPU | 16+ cores | AMD/Intel server grade |
| RAM | 128GB+ | DDR4/DDR5 ECC |
| Storage | 2TB NVMe SSD | Fast I/O for data loading |
| Network | InfiniBand | For distributed training |

### For Inference

| Component | Specification | Notes |
|----------|---------------|-------|
| GPU | RTX 3090 (24GB) | Minimum requirement |
| GPU | RTX 4090 (24GB) | Recommended |
| CPU | 8+ cores | Intel i7/i9 or Ryzen 7/9 |
| RAM | 64GB+ | |
| Storage | 500GB SSD | |

## Train

```bash
cd Jogg-Avatar
#data process
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_avatar.py   --task data_process   --dataset_path /mlp/data   --output_path ./debug  --num_frames 81   --height 512   --width 512
#train
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_avatar.py   --task train   --train_architecture lora   --dataset_path /datadisk1/data   --output_path ./models   --steps_per_epoch 60000   --max_epochs 100   --learning_rate 1e-4   --lora_rank 16   --lora_alpha 16   --lora_target_modules "q,k,v,o,ffn.0,ffn.2"   --accumulate_grad_batches 1   --use_gradient_checkpointing --use_gradient_checkpointing_offload --pretrained_lora_path pretrained_lora_path/
```

## 🔑 Inference
```bash
# 14B
torchrun --standalone --nproc_per_node=1 scripts/inference.py --config configs/inference.yaml --input_file examples/infer_samples.txt
```
## 💡Tips
  - You can control the character's behavior through the prompt in examples/infer_samples.txt, and its format is [prompt]@@[img_path]@@[audio_path]. The recommended range for prompt and audio cfg is [4-6]. You can increase the audio cfg to achieve more consistent lip-sync.

  - Control prompts guidance and audio guidance respectively, and use audio_scale=3 to control audio guidance separately. At this time, guidance_scale only controls prompts.

  - To speed up, the recommanded num_steps range is [20-50], more steps bring higher quality. To use multi-gpu inference, just set sp_size=$GPU_NUM. To use TeaCache, you can set tea_cache_l1_thresh=0.14 , and the recommanded range is [0.05-0.15].

  - To reduce GPU memory storage, you can set use_fsdp=True and num_persistent_param_in_dit. An example command is as follows:
     
```bash
torchrun --standalone --nproc_per_node=8 scripts/inference.py --config configs/inference.yaml --input_file examples/infer_samples.txt --hp=sp_size=8,max_tokens=30000,guidance_scale=4.5,overlap_frame=13,num_steps=25,use_fsdp=True,tea_cache_l1_thresh=0.14,num_persistent_param_in_dit=7000000000
```

## 🧱Model Download
| Models | Download Link | Notes |
|------|------|------|
| Wan2.1-T2V-14B | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | Base model for 14B |
| JoggAvatar model 14B(720p) | 🤗 [Huggingface](https://huggingface.co/chanjing-ai/Jogg-Avatar) | Our LoRA and audio condition weights |
| Wav2Vec | 🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h) | Audio encoder |

## 🧩 Community Works
We ❤️ contributions from the open-source community! If your work has improved Jogg-Avatar, please inform us. You can directly e-mail suqingchao@limayao.com or contact us with wechat . We are happy to reference your project for everyone's convenience. 🥸Have Fun!If you find this repository useful, please consider giving a star ⭐.

![resize](https://github.com/user-attachments/assets/5a185e16-2f54-470f-8b9d-740512d5e1b9)


## Acknowledgments
Thanks to Wan2.1, Omniavatar and DiffSynth-Studio for open-sourcing their models and code, which provided valuable references and support for this project. Their contributions to the open-source community are truly appreciated.
