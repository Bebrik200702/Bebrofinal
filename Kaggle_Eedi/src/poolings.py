import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

class MeanPooling(nn.Module):
    def __init__(self, clamp_min=1e-9):
        super(MeanPooling, self).__init__()
        self.clamp_min = clamp_min

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=self.clamp_min)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings

class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings

class Multisample_Dropout(nn.Module):
    def __init__(self,drop_range=5,predrop=0.0,):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(predrop)
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(drop_range)])

    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0)

class Multisample_StableDropout(nn.Module):
    def __init__(self,drop_range=5,predrop=0.0,):
        super(Multisample_Dropout, self).__init__()
        self.dropout = StableDropout(predrop)
        self.dropouts = nn.ModuleList([StableDropout((i+1)*.1) for i in range(drop_range)])

    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0)

class WeightedLayerPooling(nn.Module):
    def __init__(self, layers = 12):
        super(WeightedLayerPooling, self).__init__()
        self.layers = layers
        self.layer_weights = nn.Parameter(
                torch.tensor([1] * layers, dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states, dim=0)
        all_layer_embedding = all_hidden_states[-self.layers:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class Weighted_Linear(nn.Module):
    def __init__(self, hidden_size, n_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*3

        self.layer_pooler = WeightedLayerPooling(n_layers)
        self.sequence_pooler = MeanPooling()

    def forward(self, x, mask):
        x = self.layer_pooler(x.hidden_states)

        x = self.sequence_pooler(x, mask).half()

        return x

class Cat_LSTM(nn.Module):
    def __init__(self, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*n_layers
        self.n_layers = n_layers

        self.sequence_pooler = MeanPooling(1e-9)
        self.rnn = Bi_RNN_FOUT(self.cat_size, self.cat_size//2)

    def forward(self, x, mask):
        
        x = torch.cat(x.hidden_states[-self.n_layers:], dim=-1)

        hidden_mask = mask.unsqueeze(-1).expand(x.size()).float()
        x = (x * hidden_mask) #.half()

        x = self.rnn(x)
        x = self.sequence_pooler(x, mask) #.half()

        return x

class LayerBaseLSTM(nn.Module):
    def __init__(self, hidden_size,n_layers,extra_head_instances):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*n_layers
        self.n_layers = n_layers
        self.pooler = LSTM_Layer_Pooling(hidden_size, num_hidden_layers=self.n_layers)

    def forward(self, x, mask):

        x = self.pooler(x.hidden_states, mask)

        return x

class LSTM_Layer_Pooling(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers=12, is_lstm=True,bidirectional=True):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.is_lstm = is_lstm

        if self.is_lstm:
            self.lstm = nn.LSTM(
                self.hidden_size,
                self.hidden_size,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            self.lstm = nn.GRU(
                self.hidden_size,
                self.hidden_size,
                bidirectional=self.bidirectional,
                batch_first=True
            )


        self.pooling = MeanPooling(.0)

    def forward(self, all_hidden_states, mask):

        hidden_states = torch.stack([self.pooling(layer_i, mask).half()
                                     for layer_i in all_hidden_states[-self.num_hidden_layers:]], dim=1)
        out, _ = self.lstm(hidden_states)
        out = out[:, -1, :]
        return out

class Bi_RNN(nn.Module):
    def __init__(self, size, hidden_size, layers=1):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)

    def forward(self, x):
        x, hidden = self.rnn(x)
        return torch.cat((x[:,-1,:self.hidden_size], x[:,0,self.hidden_size:]), dim=-1)

class Bi_RNN_FOUT(nn.Module):
    def __init__(self, size, hidden_size, layers=1):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)
        self.initialize_lstm(self.rnn)
    
    def initialize_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        x, hidden = self.rnn(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
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

    def forward(self, x, mask):
        last_hidden_states = x[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

class LSTMPooling(nn.Module):
    def __init__(self, hidden_size,num_layers=1,drop=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=1,
            dropout=drop,
            batch_first=True,
            bidirectional=True
        )
        self.pool = MeanPooling()
        self.initialize_lstm(self.lstm)

    def initialize_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, mask):
        last_hidden_states = x['last_hidden_states']
        feature, hc = self.lstm(last_hidden_states)
        feature = self.pool(feature, mask)
        return feature

class Weighted_Linear_Attn(nn.Module):
    def __init__(self, hidden_size, n_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*3

        self.layer_pooler = WeightedLayerPooling(n_layers)
        self.sequence_pooler = AttentionPooling(hidden_size)

    def forward(self, x, mask):
        x = self.layer_pooler(x.hidden_states)

        x = self.sequence_pooler([x])

        return x

class Weighted_Linear_LSTM(nn.Module):
    def __init__(self, hidden_size, n_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.cat_size = hidden_size*3

        self.layer_pooler = WeightedLayerPooling(n_layers)
        self.sequence_pooler = LSTMPooling(hidden_size)

    def forward(self, x, mask):
        x = self.layer_pooler(x.hidden_states)

        x = self.sequence_pooler({'last_hidden_states':x},mask)

        return x

class LastTokenPooling(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
    def forward(self,input_ids,hidden_states):
        batch_size = input_ids.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        pooled_logits = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        return pooled_logits

def get_pooling(cfg,config=None):
    if cfg.pool == 'last_token':
        return LastTokenPooling(config)
    elif cfg.pool == 'mean':
        return MeanPooling()
    elif cfg.pool == 'max':
        return MaxPooling()
    elif cfg.pool == 'min':
        return MinPooling()
    elif cfg.pool == 'attention':
        return AttentionPooling(self.config.hidden_size)
    elif cfg.pool == 'lstm_simple':
        return LSTMPooling(config.hidden_size)
    elif cfg.pool == 'lstm_cat':
        return Cat_LSTM(config.hidden_size,self.config.num_hidden_layers)
    elif cfg.pool == 'lstm_layer_base':
        return LSTM_Layer_Pooling(config.hidden_size,self.config.num_hidden_layers)
    elif cfg.pool == 'weighted_linear_mean':
        return WeightedLayerPooling(config.hidden_size,self.config.num_hidden_layers)
    elif cfg.pool == 'weighted_linear_attn':
        return Weighted_Linear_Attn(config.hidden_size,self.config.num_hidden_layers)
    elif cfg.pool == 'weighted_linear_lstm':
        return Weighted_Linear_LSTM(config.hidden_size,self.config.num_hidden_layers)
