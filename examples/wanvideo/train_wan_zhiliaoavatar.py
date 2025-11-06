import torch, os, imageio, argparse
from torchvision.transforms import v2
import torch.nn as nn
import sys
sys.path.append('./')
from einops import rearrange,repeat
import cv2
import lightning as pl
import pandas as pd
from Avatar.models.model_manager import ModelManager
from Avatar.utils.io_utils import load_state_dict 
from Avatar.wan_video import WanVideoPipeline
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from typing import Tuple, Optional
from PIL import Image
import numpy as np
from whisper.audio2feature import Audio2Feature
import json
import random
import csv
import time
import librosa
import math
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as TT
from Avatar.models.wav2vec import Wav2VecModel
from Avatar.models.vae2_2 import Wan2_2_VAE
import torch.nn.functional as F
from Avatar.models.audio_pack import AudioPack
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
      
        self.path=[]
        self.path_text=[]
        # self.text=[]
        # i=0
        # for file_name in metadata["file_name"]:
        for index,row in metadata.iterrows():
            file_name = row["file_name"]
            text = row["text"]
            if not os.path.exists(os.path.join(base_path,file_name)+ ".tensors.vae2.2.pth"):
               self.path.append(os.path.join(base_path,file_name))
               self.path_text.append(text)
                #  continue
            else:
                continue
          
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = 121
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        self.transform = TT.Compose(chained_trainsforms)
        self.image_sizes_720 = [[400, 720], [720, 720], [720, 400]]
        self.image_sizes_1280 = [[704, 1280],[1280, 704]]#, [528, 960],[960, 528],[720, 1280],[1280, 720]]
        self.image_sizes_1440 = [[800, 1440], [1440, 1440], [1440, 800]]
        self.frame_process = v2.Compose([
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def match_size(self,image_size, h, w):
        ratio_ = 9999
        size_ = 9999
        select_size = None
        for image_s in image_size:
            ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
            size_tmp = abs(max(image_s) - max(w, h))
            if ratio_tmp < ratio_:
                ratio_ = ratio_tmp
                size_ = size_tmp
                select_size = image_s
            if ratio_ == ratio_tmp:
                if size_ == size_tmp:
                    select_size = image_s
        return select_size

    def resize_pad(self,image, ori_size, tgt_size):
        h, w = ori_size
        scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
        scale_h = int(h * scale_ratio)
        scale_w = int(w * scale_ratio)

        image = TT.Resize(size=[scale_h, scale_w])(image)

        padding_h = tgt_size[0] - scale_h
        padding_w = tgt_size[1] - scale_w
        pad_top = padding_h // 2
        pad_bottom = padding_h - pad_top
        pad_left = padding_w // 2
        pad_right = padding_w - pad_left

        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return image
    def load_frames_using_imageio(self, file_path, interval):
        try:
            reader = imageio.get_reader(file_path) 
            count_frames=reader.count_frames()
            if count_frames<10:
               return None
        except:
            print(file_path)
            return None
        start_frame_id = torch.randint(0, 1, (1,))[0]#torch.randint(0, count_audio - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames=[]
        frame = reader.get_data(0)
        H,W,c=frame.shape
        for frame_id in range(self.num_frames):
            try:
                frame_id_new=min(count_frames,start_frame_id + frame_id * interval+1)
                frame_ori = reader.get_data(frame_id_new-1)   
                frame=frame_ori
            except:
                return None
         
            frame = Image.fromarray(frame)
            frame = self.transform(frame)

            select_size = self.match_size(self.image_sizes_1280, H, W)
            frame = self.resize_pad(frame, (H, W), select_size)
            # print("select_size")
            frame = frame * 2.0 - 1.0
            frames.append(frame)
           
        
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

       
        return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        # text = self.text[data_id]
        path = self.path[data_id]
        content=self.path_text[data_id]
        videos = self.load_frames_using_imageio(path,self.frame_interval)
        if videos is None:
            data = {"path": ""}
        else:
            frames=videos
            ref_id=random.randint(0,frames.shape[1]-1)
            ref_frame=frames[:,ref_id:ref_id+1]
            data = {"path": path,"text":content,"frames":frames,"ref_frame":ref_frame}
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.vae = Wan2_2_VAE(
            vae_pth="/mlp/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
            device=self.device)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "pretrained_models/wav2vec2-base-960h"
            )
        self.audio_encoder = Wav2VecModel.from_pretrained("pretrained_models/wav2vec2-base-960h", local_files_only=True)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.tiler_kwargs = {"tiled": False, "tile_size": (34,34), "tile_stride": (18,16)}
        
    def test_step(self, batch, batch_idx):
        if batch["path"][0]=="":
            return
        path = batch["path"][0]
        # print(count_frames)
        self.pipe.device = self.device
        #prompt
        self.vae.todevice(dtype=self.pipe.torch_dtype,device=self.pipe.device)

        text=batch["text"][0]
        prompt_emb = self.pipe.encode_prompt(text)
        video = batch["frames"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device).squeeze(0)
        ref_frame = batch["ref_frame"].to(dtype=self.pipe.torch_dtype, device=self.pipe.device).squeeze(0)
        ref_frame=[ref_frame]
        video=[video]
        latents = self.vae.encode(video)[0].to(device=self.device,dtype=torch.bfloat16)
      
        #image
        image_lat = self.vae.encode(ref_frame)[0].to(device=self.device,dtype=torch.bfloat16)
        audio_path = path[:-4]+".wav"
        if os.path.exists(audio_path):
            try:
               audio, sr = librosa.load(audio_path, sr=16000)
            except:
                return
            input_values = np.squeeze(
                    self.wav_feature_extractor(audio, sampling_rate=16000).input_values
                )
            input_values = torch.from_numpy(input_values)
       
            audio_len = math.ceil(len(input_values) / 16000 * 25)
            input_values = input_values.unsqueeze(0)
            # padding audio
            if audio_len < 121:
                input_values = F.pad(input_values, (0, 121 * int(16000 / 25) - input_values.shape[1]), mode='constant', value=0)
                audio_len=121
            input_values=input_values.to(self.device)
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
            
                audio_embeddings = hidden_states.last_hidden_state
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            audio_embeddings=audio_embeddings[:,0:0+121,:]
          
        else:
            print("no wav")
            return
        data = {"image_lat":image_lat,"latents":latents,"prompt_emb":prompt_emb,"audio_emb":audio_embeddings}
        torch.save(data, path + ".tensors.vae2.2.pth")


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
     
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        random.shuffle(self.path)
     
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.num_frames=121
        self.frame_interval=1
        self.first_fixed_frame=1
        self.fixed_frame=13
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        self.transform = TT.Compose(chained_trainsforms)
        self.image_sizes_720 = [[400, 720], [720, 720], [720, 400]]
        self.image_sizes_1280 = [[720, 720], [528, 960],[960, 528],[720, 1280],[1280, 720]]
        self.image_sizes_1440 = [[800, 1440], [1440, 1440], [1440, 800]]
        
    def match_size(self,image_size, h, w):
        ratio_ = 9999
        size_ = 9999
        select_size = None
        for image_s in image_size:
            ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
            size_tmp = abs(max(image_s) - max(w, h))
            if ratio_tmp < ratio_:
                ratio_ = ratio_tmp
                size_ = size_tmp
                select_size = image_s
            if ratio_ == ratio_tmp:
                if size_ == size_tmp:
                    select_size = image_s
        return select_size

    def resize_pad(self,image, ori_size, tgt_size):
        h, w = ori_size
        scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
        scale_h = int(h * scale_ratio)
        scale_w = int(w * scale_ratio)

        image = TT.Resize(size=[scale_h, scale_w])(image)

        padding_h = tgt_size[0] - scale_h
        padding_w = tgt_size[1] - scale_w
        pad_top = padding_h // 2
        pad_bottom = padding_h - pad_top
        pad_left = padding_w // 2
        pad_right = padding_w - pad_left

        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return image

    
    def load_frames_using_imageio(self, file_path, interval,safetensor_path):
        if not os.path.exists(safetensor_path) or not os.path.exists(file_path[:-4]+"_mouth_info_sm.json"):
            return None
        safetensor_data = torch.load(safetensor_path, weights_only=True, map_location="cpu") 
        audio_emb = safetensor_data["audio_emb"]
        _,count_audio,_=audio_emb.shape
        reader = imageio.get_reader(file_path) 
        count_frames=reader.count_frames()
        if count_audio<self.num_frames:
            return None
        else:
            start_frame_id = torch.randint(0, count_audio - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        with open(file_path[:-4]+"_mouth_info_sm.json", 'r') as file:
             data = json.load(file)
        frames=[]
        is_tran=random.random()
        frame = reader.get_data(0)
        H,W,c=frame.shape
        try:
            fXmin, fXmax, fYmin, fYmax=data[str(start_frame_id.numpy().astype(int) +1)]["face_bbox"]
        except:
            return None
        w=int(H*(400/720))
        x_center=(fXmin+fXmax)/2
        fXmin_new=max(0,int(x_center-w/2))
        fXmax_new=min(W,int(x_center+w/2))
        for frame_id in range(self.num_frames):
            try:
                frame_id_new=min(count_frames,start_frame_id + frame_id * interval+1)
                frame_ori = reader.get_data(frame_id_new-1)   
                if is_tran>0.5:
                    if W>H:
                         frame=frame_ori[:,fXmin_new:fXmax_new,:] 
                    else:
                        new_w=int(H*(720/400))
                        left=(new_w-W)//2+(new_w-W)%2
                        right=(new_w-W)//2
                        frame=np.zeros((H,new_w,3)).astype('uint8')
                        frame[:,left:-right,:]=frame_ori
                else:
                    frame=frame_ori

            except:
                return None
        
            H,W,c=frame.shape
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            select_size = self.match_size(self.image_sizes_720, H, W)
            frame = self.resize_pad(frame, (H, W), select_size)
        
            frame = frame * 2.0 - 1.0
            frames.append(frame)
           
        
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames,start_frame_id,audio_emb
    def __getitem__(self, index):
       
        video = None
        safetensor_path = ""
        while video==None:
            data_id = torch.randint(0, len(self.path), (1,))[0]   
            video_path = self.path[data_id]
            safetensor_path = video_path+".audio.tensors.wav2vec.pth"
            video = self.load_frames_using_imageio(video_path,self.frame_interval,safetensor_path)
 
        video,start_frame_id,audio_emb = video
        if "mead" in safetensor_path:
            prompt_id = 1
        elif "boyin" in safetensor_path or "ours" in safetensor_path:
            prompt_id = 2
        else:
            prompt_id = 0

        audio_emb=audio_emb[:,start_frame_id:start_frame_id+121,:]
        
        if random.random()>0.5:
           pre_fix_frames_num=1
        else:
            pre_fix_frames_num=13
        ref_id=random.randint(0,104)
        data={}
        data["ref_frame"]=video[:,ref_id:ref_id+1]
        data["video"]=video
        data["audio_emb"]=audio_emb
        data["prompt_id"]=prompt_id
        data["pre_fix_frames_num"]=pre_fix_frames_num
        return data
    

    def __len__(self):
        return len(self.path)
class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x
def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        csv_loss_path=None
    ):
        super().__init__()
        # Load models
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                args.dit_path.split(","),
                args.text_encoder_path,
                args.vae_path
            ],
            torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
            device='cpu',
        )
        self.csv_loss_path = csv_loss_path   
      
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        audio_input_dim = 10752
        audio_out_dim = 5120
        audio_hidden_size=32
        num_layers=40
        self.freq_dim = 256
        self.audio_proj = AudioPack(audio_input_dim, [4, 1, 1], audio_hidden_size, layernorm=True)
        self.audio_cond_projs = nn.ModuleList()
        for d in range(num_layers // 2 - 1):
            l = nn.Linear(audio_hidden_size, audio_out_dim)
            self.audio_cond_projs.append(l)   
        patch_size=[1, 2, 2]
        in_dim = 33
        dim = 5120
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
     
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
      
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.tiler_kwargs = {"tiled": False, "tile_size": (34,34), "tile_stride": (18,16)}
      
       
    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        return grid_size  # x, grid_size: (f, h, w)
    
    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=1, y=2, z=2
        )
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        self.audio_proj.train()
        self.audio_cond_projs.train()
        self.patch_embedding.train()

   
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            # 
            state_dict = load_state_dict(pretrained_lora_path)
            state_dict_new = {}
            state_dict_new_module = {}
            for key in state_dict.keys():
                print(key)
                if 'pipe.dit.' in key:
                    # print(key)
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
             
                if "audio_proj" in key or "audio_cond_projs" in key or "patch_embedding" in key:
                    state_dict_new_module[key] = state_dict[key]
            state_dict = state_dict_new
            
            state_dict_new = {}

            for key in state_dict_new_module:
                if "audio_proj" in key:
                    state_dict_new[key.split("audio_proj.")[1]] = state_dict_new_module[key]
            self.audio_proj.load_state_dict(state_dict_new, strict=True)
            print("audio_proj success")
            state_dict_new = {}
            for key in state_dict_new_module:
                if "audio_cond_projs" in key:
                    state_dict_new[key.split("audio_cond_projs.")[1]] = state_dict_new_module[key]
            self.audio_cond_projs.load_state_dict(state_dict_new, strict=True)

            state_dict_new = {}
            for key in state_dict_new_module:
                if "patch_embedding" in key:
                    state_dict_new[key.split("patch_embedding.")[1]] = state_dict_new_module[key]
            self.patch_embedding.load_state_dict(state_dict_new, strict=True)

            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    
    def training_step(self, batch, batch_idx):
        # Data
        video = batch["video"]
        ref_frame = batch["ref_frame"]
        audio_emb = batch["audio_emb"][0].to(self.device)
        # print("audio_emb:",audio_emb.shape)
        pre_fix_frames_num=batch["pre_fix_frames_num"][0].to(self.device)
        prompt_id=batch["prompt_id"][0].to(self.device)
        pre_fix_frames_num = (pre_fix_frames_num+3)//4
        self.pipe.device=self.device
        if video is not None:
       
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            ref_frame = ref_frame.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            # print("ref_frame:",ref_frame.shape)
            lat = self.pipe.encode_video(video, **self.tiler_kwargs)
            latents=lat.clone()
          
            image_lat = self.pipe.encode_video(ref_frame,  **self.tiler_kwargs)
        prompt_emb =  self.prompt_emb[prompt_id]
        prompt_emb["context"] = prompt_emb["context"].to(self.device)
        image_emb={}
        msk = torch.zeros_like(image_lat.repeat(1, 1, 27, 1, 1)[:,:1])
        image_cat = image_lat.repeat(1, 1, 27, 1, 1)
        msk[:, :, pre_fix_frames_num:] = 1
        image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        
        audio_emb = audio_emb.permute(0, 2, 1)[:, :, :, None, None]
        audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2) # 1, 768, 44, 1, 1
        audio_emb = self.audio_proj(audio_emb)
        
        audio_emb = torch.concat([audio_cond_proj(audio_emb) for audio_cond_proj in self.audio_cond_projs], 0)

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        noisy_latents[:,:,:pre_fix_frames_num]=lat[:,:,:pre_fix_frames_num]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        # print("training_target:",training_target.shape)
        training_target = training_target[:,:,pre_fix_frames_num:,:,:] 

        # Compute loss

        lat_h, lat_w = noisy_latents.shape[-2], noisy_latents.shape[-1]
        y=image_emb["y"]
        x = torch.cat([noisy_latents, y], dim=1)
     
        x = self.patch_embedding(x)
      
        noise_pred= self.pipe.denoising_model()(
            x, timestep=timestep, **prompt_emb,**extra_input, **image_emb,audio_emb=audio_emb,lat_h=lat_h, lat_w=lat_w
        )
     
        noise_pred=noise_pred[:,:,pre_fix_frames_num:,:,:] 
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
      
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        if "0" in str(self.device):
            with open(self.csv_loss_path, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['train_loss'])
                    writer.writerow({'train_loss': loss.item()})
        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
        


    def configure_optimizers(self):
        trainable_modules = [
            {'params': filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())},
            {'params': self.audio_proj.parameters()},
            {'params': self.audio_cond_projs.parameters()},
            {'params': self.patch_embedding.parameters()},
        ]
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters())) 
        
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()
        # state_dict.update()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="/mlp/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/mlp/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,/datadisk1/models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default="None",
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "hallo3_videos_clip_all.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "all.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    time_str = time.strftime("%Y/%m/%d %H:%M:%S", local_time).replace("/","_")
    csv_file = "loss/"+time_str+'_loss_values.csv'
    fieldnames = ['train_loss']

    # 打开CSV文件并写入表头
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        csv_loss_path=csv_file
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
