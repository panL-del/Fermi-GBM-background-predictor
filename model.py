import torch
import torch.nn as nn
import torch.nn.functional as f

class FeatureGroupTokenizer(nn.Module):
    def __init__(self, group_col_id, token_size):
        super(FeatureGroupTokenizer, self).__init__()
        
        self.group_col_id = group_col_id

        self.GroupEncoders = nn.ModuleDict({})

        for group in group_col_id:
            num_vars = len(group_col_id[group])
            self.GroupEncoders[f'{group}'] = nn.Sequential(
                nn.Linear(num_vars, token_size),
                nn.ReLU(),
                nn.Linear(token_size, token_size),
                nn.ReLU(),
                nn.Linear(token_size, token_size),
                nn.LayerNorm(token_size)
                )

            self.GroupEncoders[group].apply(init_DNN)

    def forward(self, x):
        token_list = []
        for group in self.group_col_id:
            col_ids = self.group_col_id[group]
            token = self.GroupEncoders[f'{group}'](x[:, col_ids])
            token_list.append(token)
   
        # tokens = torch.stack(token_list)
        # tokens = torch.permute(tokens, (1,0,2))
        # [batch, n_group, token_size]
        tokens = torch.stack(token_list, dim=1)
        return tokens


class FermiGBMground(nn.Module):
    def __init__(self, input_dim, 
                 output_dim, 
                 group_col_id,
                 token_size=32,
                 nhead=8, 
                 n_transformers=6, 
                 embed_dim=256,
                 dropout=0.1):
        
        super(FR_BERTground, self).__init__()

        # group tokenizer
        self.tokenizer = FeatureGroupTokenizer(group_col_id, token_size=token_size)

        # cls token: 小范围随机初始化
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_size))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_size, 
            nhead=nhead,
            dropout=dropout, 
            activation='gelu',
            dim_feedforward=embed_dim, 
            batch_first=True,
            norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=n_transformers)

        # 改进的门控层
        self.gate_layer = nn.Sequential(
            nn.Linear(token_size*2, token_size),
            nn.ReLU(),
            nn.Linear(token_size, 1),
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(token_size, 128),  
            nn.ReLU(),
            nn.Dropout(dropout),             
            nn.Linear(128, output_dim)
        )
        
        self.apply(init_DNN)
        
    def forward(self, x):
        bs = x.shape[0]

        tokens = self.tokenizer(x)

        cls_tokens = self.cls_token.expand(bs, -1, -1)

        tokens = torch.cat((cls_tokens, tokens), dim = 1)

        h = self.transformer_encoder(tokens)

        cls_output = h[:,0,:]
        group_outputs = h[:, 1:, :].mean(dim=1)

        alpha = self.gate_layer(torch.cat([cls_output, group_outputs], dim=-1))
        combined = alpha * cls_output + (1 - alpha) * group_outputs

        out = self.fc(combined)
        
        return out



def init_DNN(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
