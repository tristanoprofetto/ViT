import math
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """
    Turns images into Patched Embeddings required as model input
    """

    def __init__(self, config):
        super().__init__()
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.hiddden_size = config['hidden_size']
        self.num_channels = config['num_channels']
        self.num_patches = ( self.img_size // self.patch_size ) ** 2
        self.projection = nn.Conv2d(in_channels=self.num_channels, out_channels=self.hiddden_size, kernel_size=self.patch_size, stride=self.patch_size)

    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class Embeddings(nn.Module):
    """
    Creates input Embeddings by combining Patch Embeddings with CLS tokens and positional embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbedding(config)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, config['hidden_size']))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_channels + 1, config['hidden_size']))
        self.dropout = nn.Dropout(config['hidden_dropout'])


    def forward(self, x):
        x = self.patch_embeddings(x)

        batch_size, _, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.position_embeddings

        x = self.dropout(x)

        return x
    

class AttentionHead(nn.Module):
    """
    Computes Attention weights from calculating Q, K, V from a sequence of input Embeddings
    """

    def __init__(self, hidden_size, attention_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.query = nn.Linear(hidden_size, attention_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_size, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Calculate Attention Scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, v)

        return (output, attention_probs)
    

class MultiHeadedAttention(nn.Module):
    """
    Applies several attention mechanisms in parallel
    """

    def __init__(self, config):
        super().__init__()
        self.attention_dropout = config['attention_probs_dropout']
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_size = self.hidden_size // self.num_attention_heads
        
        self.total_attention_size = self.num_attention_heads * self.total_attention_size
        
        self.qkv_bias = config['qkv_bias']
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                hidden_size=self.hidden_size,
                attention_size=self.attention_size,
                dropout=self.attention_dropout,
                bias=self.qkv_bias
            )
            self.heads.append(head)

        self.output_projection = nn.Linear(self.total_attention_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config['hidden_dropout'])


    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention for attention, _ in attention_outputs], dim=-1)
        attention_output = self.projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([prob for _, prob in attention_outputs])
            return (attention_output, attention_probs)


class GELU():
    """
    GELU activation function implementation
    """

    def forward(self, x):
        return 0.5 * x * ( 1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))) ) 


class MLP(nn.Module):
    """
    Multi Layer Perceptron
    """

    def __init__(self, config):
        super().__init__()
        self.d1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.d2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['hidden_dropout'])
        self.activation = GELU()


    def forward(self, x):
        x = self.d1(x)
        x = self.activation(x)
        x = self.d2(x)
        x = self.dropout(x)

        return x
    

class Block(nn.Module):
    """
    Transformer Block
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadedAttention(config)
        self.mlp = MLP(config)
        self.layer_norm = nn.LayerNorm(config['hidden_size'])


    def forward(self, x, output_attentions=False):
        attention, probs = self.attention(self.layer_norm(x), output_attention=output_attentions)
        x = x + attention

        mlp_output = self.mlp(self.layer_norm(x))
        x = x + mlp_output

        if not output_attentions:
            return (x, None)
        else:
            return (x, probs)


class Encoder(nn.Module):
    """"
    Encodes input sequence
    """

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            block = Block(config)
            self.blocks.append(block)

    
    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            
            if output_attentions:
                all_attentions.append(attention_probs)
            
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class LabelClassifier(nn.Module):
    """"
    Label classifier: maps outputs from the encoder to the given number of classes
    """

    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.output = nn.Linear(hidden_size, num_classes)


    def predict(self ,x):
        return self.output(x)


class ViT(nn.Module):
    """
    ViT model for classification
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config['img_size']
        self.hidden_size = config['hidden_size']
        self.num_classes = config['num_classes']
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = LabelClassifier(config['hidden_size'], config['num_classes'])
        self.apply(self._init_weights)

    
    def forward(self, x, output_attentions=False):
        embeddings = self.embedding(x)
        encoder_ouptut, all_attentions = self.encoder(embeddings, output_attentions=output_attentions)
        logits = self.classifier.predict(encoder_ouptut[:, 0])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal(module.weight, mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0, stf=self.config['initializer_range']
                ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), 
                mean=0.0, 
                std=self.config['initializer_range']
                ).to(module.cls_token.dtype)
                




    


    




