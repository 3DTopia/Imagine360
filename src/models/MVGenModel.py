import torch
import gc
import torch.nn as nn
from src.modules.attn_perspano import WarpAttn
from einops import rearrange
from src.utils.pano import pad_pano, unpad_pano
from fairscale.nn.checkpoint import checkpoint_wrapper
from torch.utils.checkpoint import checkpoint
from src.modules.utils import flush, check_cuda_memo

def add_noise_to_condition(condition, noise_level=0.1):
    noise = torch.randn_like(condition) * noise_level
    noisy_condition = condition + noise
    return noisy_condition

class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, pano_unet, pano_pad=True, device='cuda'):
        super().__init__()

        self.unet = unet
        self.pano_unet = pano_unet
        self.pano_pad = pano_pad
        if self.unet is not None:
            self.cp_blocks_encoder = nn.ModuleList()
            for downsample_block in self.unet.down_blocks:
                if downsample_block.downsamplers is not None:
                    self.cp_blocks_encoder.append(WarpAttn(
                        downsample_block.downsamplers[-1].out_channels))

            self.cp_blocks_mid = WarpAttn(
                self.unet.mid_block.resnets[-1].out_channels)

            self.cp_blocks_decoder = nn.ModuleList()
            for upsample_block in self.unet.up_blocks:
                if upsample_block.upsamplers is not None:
                    self.cp_blocks_decoder.append(WarpAttn(
                        upsample_block.upsamplers[0].channels))

            self.trainable_parameters = [(list(self.cp_blocks_mid.parameters()) + \
                list(self.cp_blocks_decoder.parameters()) + \
                list(self.cp_blocks_encoder.parameters()), 1.0)]


            if self.unet is not None:
                self.unet.up_blocks = nn.ModuleList([
                    checkpoint_wrapper(upsample_block)
                    for upsample_block in self.unet.up_blocks
                ])


            self.pano_unet.up_blocks = nn.ModuleList([
                checkpoint_wrapper(upsample_block)
                for upsample_block in self.pano_unet.up_blocks
            ])
        



    def forward(self, 
                latents, pano_latent, 
                timestep, 
                prompt_embd, pano_prompt_embd, 
                cameras,
                use_fps_condition,
                use_ip_plus_cross_attention,
                fps_tensor_pano, fps_tensor_pers,
                reference_images_clip_feat_pano, reference_images_clip_feat_pers,
                relative_position_tensor,
                pitchs_tensor):
        
        # bs*m, 4, 64, 64

        #latents:          [1, 20, 9, 64, 32, 32]           (b m c f h w)
        #pano_latent:     [1, 9, 64, 64, 128]              (b c f h w)
        #timestep:         tensor([389], device='cuda:0')
        #prompt_embd:      [20, 77, 1024]] pers_encoder_hidden_states
        #pano_prompt_embd: [1, 77, 1024]   pano_encoder_hidden_states
        #fps_tensor_pano:   shape/[2]   value/[8, 8]
        #fps_tensor_pers:   shape/[2,20]
        #reference_images_clip_feat_pers:   [2, 20, 64, 4096, 256]
        #reference_images_clip_feat_pano:   [2, 64, 4096, 256]
        #relative_position_tensor   :value| tensor([[1, 1, 255, 255, 512, 1024])
        
        #old latent pano/pers torch.Size([2, 20, 4, 32, 32]) torch.Size([2, 1, 4, 64, 128])
        
        """
        unet.conv_in (input: [b c f h w])       [2, 9, 64, 64, 128] ——> [2, 320, 64, 64, 128]
        
        """
                
        
        device = latents.device
        # check_cuda_memo("mv base model forward", device)
        torch.cuda.empty_cache()
        
        if latents is not None:
            b, m, c, f, h, w = latents.shape
            latents = rearrange(latents, 'b m c f h w -> (b m) c f h w').contiguous() 
            
        if cameras is not None:
            cameras = {k: rearrange(v, 'b m ... -> (b m) ...').contiguous()  for k, v in cameras.items()}
            
        # 1.1 process timesteps and fps_tensor
        timestep = timestep[:, None].repeat(b, m) #[1, 20]
        if self.unet is not None:
            pano_timestep = timestep[:, 0].clone()
            timestep = timestep.reshape(-1).clone()
            t_emb = self.unet.time_proj(timestep).to(self.unet.dtype)  # (bs*m, 320)
            emb = self.unet.time_embedding(t_emb)  # (bs*m, 1280)
            
        else:
            pano_timestep = timestep.clone()
            
        pano_t_emb = self.pano_unet.time_proj(pano_timestep).to(self.pano_unet.dtype)  # (bs, 320)
        pano_emb = self.pano_unet.time_embedding(pano_t_emb)  # (bs, 1280)
        
        if use_fps_condition:

            fps_tensor_pano = fps_tensor_pano.to(pano_latent.dtype)
            fps_tensor_pano = fps_tensor_pano.expand(pano_latent.shape[0]).to(pano_t_emb.device) #shape[1]
            fps_emb_pano = self.pano_unet.time_proj(fps_tensor_pano)
            fps_emb_pano = fps_emb_pano.to(dtype=self.pano_unet.dtype) #[1, 320]
            
            pano_emb += self.pano_unet.fps_embedding(fps_emb_pano) #self.pano_unet.fps_embedding(fps_emb_pano): [1, 1280]
            
            

            fps_tensor_pers = fps_tensor_pers.to(latents.dtype)   #[1, 20]
            fps_tensor_pers = rearrange(fps_tensor_pers, 'b m -> (b m)').contiguous()  #[20,]
            fps_tensor_pers = fps_tensor_pers.expand(latents.shape[0]).to(t_emb.device) #[20,]
            fps_emb_pers = self.unet.time_proj(fps_tensor_pers)
            fps_emb_pers = fps_emb_pers.to(dtype=self.unet.dtype)  #[20, 320]
            emb += self.unet.fps_embedding(fps_emb_pers)

        if self.unet is not None:
            hidden_states = self.unet.conv_in(latents)             #[20, 320, 64, 32, 32]
            
        if self.pano_pad:
            pano_latent = pad_pano(pano_latent, 1)                  #[1, 320, 64, 64, 130]

        pano_hidden_states = self.pano_unet.conv_in(pano_latent)    #[1, 320, 64, 64, 130]
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 1)  #[1, 320, 64, 64, 128]
        
        
        # check_cuda_memo("process timesteps and fps_tensor", device)
        torch.cuda.empty_cache()
        
        
        # 1.2 process ip_plus_cross_attention
        
        #reference_images_clip_feat_pano:   [2, 64, 4096, 256]
        #reference_images_clip_feat_pers:   [2, 20, 64, 4096, 256]
           
        if use_ip_plus_cross_attention:
            
            
            if self.pano_unet.ip_plus_condition == 'video':
                
                # perform temporal self-attention
                ### pano unet ####
                reference_images_clip_feat_pano = self.pano_unet.temporal_proj(reference_images_clip_feat_pano) #[1, 2, 256, 1024]

                batch_size, f, num_token, dim_feature = reference_images_clip_feat_pano.shape
                # batch size must be one -> validation classier free guidance use batch_size=2
                # batch_size = 1
                reference_images_clip_feat_pano = reference_images_clip_feat_pano.reshape(batch_size, f*num_token, dim_feature) #[1, 1024, 1024]

                ### pers unet ####
                reference_images_clip_feat_pers = rearrange(reference_images_clip_feat_pers, 'b m f (h w) c -> (b m) f (h w) c', h=64, w=64).contiguous() 
                reference_images_clip_feat_pers = self.unet.temporal_proj(reference_images_clip_feat_pers) #[20, 4, 256, 1024]
                                
                batch_size, f, num_token, dim_feature = reference_images_clip_feat_pers.shape
                # batch size must be one -> validation classier free guidance use batch_size=2
                # batch_size = 1
                reference_images_clip_feat_pers = reference_images_clip_feat_pers.reshape(batch_size, f*num_token, dim_feature) #[20, 1024, 1024]

            #reference_images_clip_feat_pano: [2, 256, 1024]
            #reference_images_clip_feat_pers: [40, 256, 1024]
            
            ###### relative_position is only on panounet_branch ######

            ip_tokens_pano = self.pano_unet.image_proj_model(reference_images_clip_feat_pano) #[2, 64, 1024]
            ip_tokens_pers = self.unet.image_proj_model(reference_images_clip_feat_pers)    #[40, 64, 1024]
            
            ip_tokens_pano = add_noise_to_condition(ip_tokens_pano)
            ip_tokens_pers = add_noise_to_condition(ip_tokens_pers)
            
            relative_position_tensors = relative_position_tensor.clone()
            pitchs_tensors = pitchs_tensor.unsqueeze(-1).clone()
            
            
            if relative_position_tensor is not None and self.pano_unet.use_relative_postions == 'WithAdapter':
                
                add_cond_emb_panos = []
                for i in range(relative_position_tensor.shape[1]):
                                        
                    relative_position_tensor = relative_position_tensors[:,i,:].flatten()
                    add_cond_emb_pano1 = self.pano_unet.add_cond_proj(relative_position_tensor) #[12, 320]
                    add_cond_emb_pano1 = add_cond_emb_pano1.reshape((pano_emb.shape[0], -1)) #[2, 1920]
                    add_cond_emb_pano1 = self.pano_unet.add_cond_embedding(add_cond_emb_pano1.to(self.unet.dtype))

                    add_cond_emb_pano1 = self.pano_unet.cond_rp_proj(add_cond_emb_pano1)  #[2, 786]

                    pitchs_tensor = pitchs_tensors[:,i,:].flatten()
                    add_cond_emb_pano2 = self.pano_unet.add_cond_proj(pitchs_tensor) #[12, 320]
                    add_cond_emb_pano2 = add_cond_emb_pano2.reshape((pano_emb.shape[0], -1)) #[2, 1920]
                    
                    add_cond_emb_pano2 = self.pano_unet.add_cond_embedding2(add_cond_emb_pano2.to(self.unet.dtype)) #[2, 256]
                    

                    # print("add_cond_emb_pano:", add_cond_emb_pano1.dtype, add_cond_emb_pano2.dtype, self.unet.dtype)

                    add_cond_emb_pano = torch.cat((add_cond_emb_pano1, add_cond_emb_pano2), dim=-1)
                    add_cond_emb_panos.append(add_cond_emb_pano)

       
                for _ in range(ip_tokens_pano.shape[1] - relative_position_tensors.shape[1]):
                    add_cond_emb_panos.append(add_cond_emb_pano)

                add_cond_emb_panos = torch.stack(add_cond_emb_panos, dim=1)
                ip_tokens_pano = ip_tokens_pano + add_cond_emb_panos    #[2, 64, 1024]
                
            ############################################################
            

            # ###### relative_position is only on panounet_branch ######

            #prompt_embd:      [20, 77, 1024]] pers_encoder_hidden_states
            #pano_prompt_embd: [1, 77, 1024]   pano_encoder_hidden_states
            
            if ip_tokens_pano.shape[2] != pano_prompt_embd.shape[2]:
                # pad the ip tokens
                ip_tokens_pano_pad = torch.zeros([ip_tokens_pano.shape[0], ip_tokens_pano.shape[1], pano_prompt_embd.shape[2]]).to(ip_tokens_pano.device)
                ip_tokens_pano_pad[:,:,:self.pano_unet.image_cross_attention_dim] = ip_tokens_pano
                ip_tokens_pano = ip_tokens_pano_pad

            if ip_tokens_pers.shape[2] != prompt_embd.shape[2]:
                # pad the ip tokens
                ip_tokens_pers_pad = torch.zeros([ip_tokens_pers.shape[0], ip_tokens_pers.shape[1], prompt_embd.shape[2]]).to(ip_tokens_pers.device)
                ip_tokens_pers_pad[:,:,:self.unet.image_cross_attention_dim] = ip_tokens_pers
                ip_tokens_pers = ip_tokens_pers_pad


            pano_encoder_hidden_states = torch.cat([pano_prompt_embd, ip_tokens_pano], dim=1)   #[1, 141, 1024]
            pers_encoder_hidden_states = torch.cat([prompt_embd, ip_tokens_pers], dim=1)        #[20, 141, 1024]]

    
        
        ###################################################################################################
        # unet

        #hidden_states:     [20, 320, 64, 32, 32]
        #pano_hidden_states: [1, 320, 64, 64, 128]
        # a. downsample
        if self.unet is not None:
            down_block_res_samples = (hidden_states,)
        pano_down_block_res_samples = (pano_hidden_states,)
        
        # print("======== down ========")
        for i, downsample_block in enumerate(self.pano_unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for j in range(len(downsample_block.resnets)):
                    if self.unet is not None:
                        hidden_states = self.unet.down_blocks[i].resnets[j](hidden_states.contiguous(), emb) #[b,c,f,h,w] [2, 320, 64, 64, 128]
                                                
                        hidden_states = self.unet.down_blocks[i].attentions[j](                 #[b,c,f,h,w] [2, 320, 64, 64, 128]
                            hidden_states.contiguous(), encoder_hidden_states=pers_encoder_hidden_states
                        ).sample
                        # add motion module
                        hidden_states = self.unet.down_blocks[i].motion_modules[j](hidden_states.contiguous(), emb, encoder_hidden_states=pers_encoder_hidden_states) if self.unet.down_blocks[i].motion_modules[j] is not None else hidden_states                            
                        down_block_res_samples += (hidden_states,)
                        

                    ####### pano branch #######
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states.contiguous(), 2)   #[1, 320, 16, 64, 128] -> [1, 320, 16, 64, 132]
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](pano_hidden_states.contiguous(), pano_emb)

                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states.contiguous(), 2) #[1, 320, 16, 64, 132] -> [1, 320, 16, 64, 128]
                        
                    pano_hidden_states = self.pano_unet.down_blocks[i].attentions[j](
                        pano_hidden_states.contiguous(), encoder_hidden_states=pano_encoder_hidden_states
                    ).sample
                    
                    # add motion module
                    pano_hidden_states = self.pano_unet.down_blocks[i].motion_modules[j](pano_hidden_states, pano_emb, encoder_hidden_states=pano_encoder_hidden_states) if self.pano_unet.down_blocks[i].motion_modules[j] is not None else pano_hidden_states
                   
                    pano_down_block_res_samples += (pano_hidden_states,)
            else:
                for j in range(len(downsample_block.resnets)):
                    if self.unet is not None:
                        hidden_states = self.unet.down_blocks[i].resnets[j](hidden_states, emb)
                        down_block_res_samples += (hidden_states,)

                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_down_block_res_samples += (pano_hidden_states,)

            if downsample_block.downsamplers is not None:
                for j in range(len(downsample_block.downsamplers)):
                    if self.unet is not None:
                        hidden_states = self.unet.down_blocks[i].downsamplers[j](hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].downsamplers[j](
                        pano_hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 1)

                if self.unet is not None:
                    down_block_res_samples += (hidden_states,)
                pano_down_block_res_samples += (pano_hidden_states,)
                
                # torch.cuda.empty_cache()
                
                if self.unet is not None:                    
                    #hidden_states:      [20, 320, 16, 16, 16]
                    #pano_hidden_states: [1,  320, 16, 32, 64]
                    hidden_states, pano_hidden_states = self.cp_blocks_encoder[i](
                        hidden_states, pano_hidden_states, cameras)
                    

        # check_cuda_memo("downsample done", device)
        flush()
        

        # print("======== mid ========")

        # b. mid
        if self.unet is not None:
            hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)
            
        if self.pano_pad:
            pano_hidden_states = pad_pano(pano_hidden_states, 2)
            
        pano_hidden_states = self.pano_unet.mid_block.resnets[0](pano_hidden_states, pano_emb)
        
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        for i in range(len(self.pano_unet.mid_block.attentions)):
            if self.unet is not None:
                hidden_states = self.unet.mid_block.attentions[i](
                    hidden_states, encoder_hidden_states=pers_encoder_hidden_states
                ).sample
                
                                      
                # add motion module
                hidden_states = self.unet.mid_block.motion_modules[i](hidden_states, emb, encoder_hidden_states=pers_encoder_hidden_states) if self.unet.mid_block.motion_modules[i] is not None else hidden_states                            

                hidden_states = self.unet.mid_block.resnets[i+1](hidden_states, emb)




            pano_hidden_states = self.pano_unet.mid_block.attentions[i](
                pano_hidden_states, encoder_hidden_states=pano_encoder_hidden_states
            ).sample
        
            # add motion module
            pano_hidden_states = self.pano_unet.mid_block.motion_modules[i](pano_hidden_states, pano_emb, encoder_hidden_states=pano_encoder_hidden_states) if self.pano_unet.mid_block.motion_modules[i] is not None else pano_hidden_states                            

            
            if self.pano_pad:
                pano_hidden_states = pad_pano(pano_hidden_states, 2)
            pano_hidden_states = self.pano_unet.mid_block.resnets[i+1](
                pano_hidden_states, pano_emb)
            if self.pano_pad:
                pano_hidden_states = unpad_pano(pano_hidden_states, 2)


        if self.unet is not None:
            hidden_states, pano_hidden_states = self.cp_blocks_mid(
                hidden_states, pano_hidden_states, cameras)

        flush()
        

        def resnet_forward(resnet, hidden_states, emb):
            return resnet(hidden_states, emb)

        def attention_forward(attention, hidden_states, encoder_hidden_states):
            return attention(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

        def motion_module_forward(motion_module, hidden_states, emb, encoder_hidden_states):
            return motion_module(hidden_states, emb, encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states

        # c. upsample
        for i, upsample_block in enumerate(self.pano_unet.up_blocks):
            if self.unet is not None:
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            pano_res_samples = pano_down_block_res_samples[-len(upsample_block.resnets):]
            pano_down_block_res_samples = pano_down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for j in range(len(upsample_block.resnets)):
                    if self.unet is not None:
                        res_hidden_states = res_samples[-1]
                        res_samples = res_samples[:-1]
                        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                        
                        hidden_states = checkpoint(resnet_forward, self.unet.up_blocks[i].resnets[j], hidden_states, emb)
                        hidden_states = checkpoint(attention_forward, self.unet.up_blocks[i].attentions[j], hidden_states, pers_encoder_hidden_states)
                        hidden_states = checkpoint(motion_module_forward, self.unet.up_blocks[i].motion_modules[j], hidden_states, emb, pers_encoder_hidden_states)

                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat([pano_hidden_states, pano_res_hidden_states], dim=1)

                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = checkpoint(resnet_forward, self.pano_unet.up_blocks[i].resnets[j], pano_hidden_states, pano_emb)

                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_hidden_states = checkpoint(attention_forward, self.pano_unet.up_blocks[i].attentions[j], pano_hidden_states, pano_encoder_hidden_states)
                    pano_hidden_states = checkpoint(motion_module_forward, self.pano_unet.up_blocks[i].motion_modules[j], pano_hidden_states, pano_emb, pano_encoder_hidden_states)

            else:
                for j in range(len(upsample_block.resnets)):
                    if self.unet is not None:
                        res_hidden_states = res_samples[-1]
                        res_samples = res_samples[:-1]
                        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                        
                        hidden_states = checkpoint(resnet_forward, self.unet.up_blocks[i].resnets[j], hidden_states, emb)

                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat([pano_hidden_states, pano_res_hidden_states], dim=1)

                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = checkpoint(resnet_forward, self.pano_unet.up_blocks[i].resnets[j], pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

            if upsample_block.upsamplers is not None:
                if self.unet is not None:
                    hidden_states, pano_hidden_states = self.cp_blocks_decoder[i](hidden_states, pano_hidden_states, cameras)

                for j in range(len(upsample_block.upsamplers)):
                    if self.unet is not None:
                        hidden_states = checkpoint(self.unet.up_blocks[i].upsamplers[j], hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 1)
                    pano_hidden_states = checkpoint(self.pano_unet.up_blocks[i].upsamplers[j], pano_hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

                torch.cuda.empty_cache()
            
            
        # 4.post-process
        if self.unet is not None:
            sample = self.unet.conv_norm_out(hidden_states)
            sample = self.unet.conv_act(sample)
            sample = self.unet.conv_out(sample)   #[20, 4, 16, 16, 16]
            
            
            sample = rearrange(sample, '(b m) c f h w -> b m c f h w', m=m).contiguous() 
        else:
            sample = None

        pano_sample = self.pano_unet.conv_norm_out(pano_hidden_states)
        pano_sample = self.pano_unet.conv_act(pano_sample)
        if self.pano_pad:
            pano_sample = pad_pano(pano_sample, 1)
        pano_sample = self.pano_unet.conv_out(pano_sample)
        if self.pano_pad:
            pano_sample = unpad_pano(pano_sample, 1)
        

        return sample, pano_sample