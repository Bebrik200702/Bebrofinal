import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .poolings import *
from transformers import AutoConfig, AutoModel

class CustomEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        if type(self.config.eos_token_id) != list:
            self.config.pad_token_id = self.config.eos_token_id
        else:
            self.config.pad_token_id = self.config.eos_token_id[0]
        if cfg.turn_off_drop:
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.config,
            torch_dtype=cfg.torch_dtype
        )
        if cfg.use_lora:
            peft_config = LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.lora_alpha,
                lora_dropout=cfg.lora.lora_dropout,
                bias=cfg.lora.bias,
                #task_type='SEQ_CLS',
                use_dora=cfg.lora.use_dora,
                target_modules=cfg.lora.target_modules
            )
            self.model = get_peft_model(self.model, peft_config)
        
        self.pool = get_pooling(cfg,config=self.config)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        if self.cfg.pool == 'last_token':
            pooled_output = self.pool(input_ids, outputs[0]) 
        elif self.cfg.pool == 'mean':
            pooled_output = self.pool(outputs[0],attention_mask)
        else:
            pooled_output = self.pool(outputs,attention_mask)
            

        return BaseModelOutputWithPooling(
            pooler_output=pooled_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )