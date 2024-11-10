import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers.modeling_outputs import SequenceClassifierOutput, MultipleChoiceModelOutput
from .poolings import *
from transformers import AutoConfig, AutoModel, T5EncoderModel

class CustomMultipleChoiceEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        if cfg.not_padding_token:
            if type(self.config.eos_token_id) != list:
                self.config.pad_token_id = self.config.eos_token_id
            else:
                self.config.pad_token_id = self.config.eos_token_id[0]
        if cfg.turn_off_drop:
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        
        if cfg.use_only_encoder:
            self.model = T5EncoderModel.from_pretrained(
                cfg.model,
                config=self.config,
                torch_dtype=cfg.torch_dtype
            )

        else:
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
                target_modules=cfg.lora.target_modules,
                layers_to_transform=cfg.lora.layers_to_transform
            )
            self.model = get_peft_model(self.model, peft_config)
        
        self.pool = get_pooling(cfg,config=self.config)
        
        if cfg.cls_drop_type == 'stable':
            self.cls_drop = StableDropout(cfg.cls_drop)
        elif cfg.cls_drop_type == 'multi':
            self.cls_drop = Multisample_Dropout(cfg.multi_drop_range)
        else:
            self.cls_drop = nn.Dropout(cfg.cls_drop)
            
        if self.cfg.pool != 'lstm_cat':
            self.fc = nn.Linear(self.config.hidden_size,self.cfg.num_labels)
        else:
            self.fc = nn.Linear(self.config.hidden_size * self.config.num_hidden_layers,self.cfg.num_labels)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if 'initializer_range' not in self.config.to_dict().keys():
            self.config.initializer_range = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = ( 
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        outputs['last_hidden_state'] = outputs['last_hidden_state'].squeeze(-1)
        outputs['hidden_states'] = [x.squeeze(-1) for x in outputs['hidden_states']]
        
        if self.cfg.pool == 'last_token':
            pooled_output = self.pool(input_ids, outputs[0])
        elif self.cfg.pool == 'mean':
            pooled_output = self.pool(outputs[0],flat_attention_mask)
        else:
            pooled_output = self.pool(outputs,flat_attention_mask)
            
        if self.cfg.cls_drop_type != 'multi':
            pooled_output = self.cls_drop(pooled_output)
            logits = self.fc(pooled_output)
        else:
            logits = self.cls_drop(pooled_output,self.fc)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
