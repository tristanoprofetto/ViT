# Vision Transformers (ViT)

This is a simple implementation of the paper [An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929.pdf) using the PyTorch framework: Transformers for Image Recognition at Scale.

### About the Dataset
This model was trained on the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)


### Model Training
Here is a list of the hyperparameter set for model training:
```json
{
    "patch_size": 4,  
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 182,
    "hidden_dropout": 0.0,
    "attention_probs_dropout": 0.0,
    "initializer_range": 0.02,
    "img_size": 32,
    "num_classes": 10, 
    "num_channels": 3,
    "qkv_bias": true,
    "use_faster_attention": true
}
```

