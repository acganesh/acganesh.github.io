---
layout: post
title: GPT in words and code
date: 2023-07-04
math: true
draft: true
description: Notes on transformers / LLMs from the ground up.
---

I find that the best way to understand how machine learning papers work is to write the code for a model forward pass.  If you can load the weights from a pre-trained model and get the same outputs from a single model inference, you can be pretty confident that you've re-implemented all of the details from a model.  The advantages of doing this are:

- Does not require any training, which can be time-consuming and expensive.
- Can test model outputs layer-by-layer to validate the correctness of different components.
- Get a satisfying payoff at the end with a working model.

This is the strategy adopted in Jay Mody's [picoGPT](https://github.com/jaymody/picoGPT) and Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

For a good exercise to replicate GPT-inference, I would recommend reimplementing the `gpt2` function in the picoGPT repo above, located [here](https://github.com/jaymody/picoGPT/blob/main/gpt2.py#L90C20-L90C20).  picoGPT makes this especially easy because the weight loading code is already written and you can just write the forward pass in NumPy. A link to my implementation can be found [here](https://github.com/acganesh/picoGPT).

## Model architecture 

Here I will break down the GPT architecture into its components.  Here I will focus on GPT-3, since the architecture has been [described in the 2020 paper](https://arxiv.org/pdf/2005.14165.pdf).

Given `$n_{\text{ctx}} = 2048$` tokens, GPT-3 will output a probability distribution over its vocabulary size of 50257 tokens.  Decoding the next token can be done by grabbing the `argmax` over this distribution.

GPT-3 has the following architectural parameters:
- `$n_{\text{params}} = 175B$`
- `$n_{\text{layers}} = 96$`
- `$d_{\text{model}} = 12288$`
- `$n_{\text{heads}} = 96$`
- `$d_{\text{head}} = 128$`

### 1) Byte-Pair Embedding Tokenizer

`$n_{ctx} = 2048$` tokens of text `$\to$` one-hot tensor of shape `$n_{ctx} \times  n_{vocab} = 2048 \times 50257$`.

The first step is to convert words to numbers using a tokenizer.  GPT uses [byte-pair encoding](https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe) (BPE).  In BPE, the most common words are mapped to single tokens while less common words will be broken down into chunks and mapped to multiple tokens.

OpenAI's [Tokenizer](https://platform.openai.com/tokenizer) tool shows how different sentences will be broken into tokens under BPE.  In this example, most words get mapped to a single token, except for "Sylvester," which gets chunked into three tokens: `Sy`, `lves`, and `ter`.

TODO: Fix image.
![Tokenizer](./img/transformers-tokenizer.png)

### 2A) Word Embeddings

`$n_{ctx} \times n_{vocab} \to n_{ctx} \times d_{\text{model}}$`

Now we convert the sparse one-hot tokens tensor into a dense embedding matrix.  This is done by a linear layer.

### 2B) Positional Encodings 

`$n_{\text{ctx}} \times 1 \to n_{\text{ctx}} \times d_{\text{model}}$`

Transformers are invariant to the order of inputs, so we need to tell the model which position each word is in.  In GPT-3, this is done with (EQUATION).

### 2C) Sum

At this step, we sum up the Word Embeddings and Positional Embedding to aggregate them into a single embedding of shape n_ctx x d_model.

### 3) Multi-Head Attention

To explain how multi-head attention works in GPT-3, we will start with single-head attention.

We define the *attention* operation as follows: 
`$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left ( \frac{\mathbf{QK^T}}{\sqrt{d_k}} \right ) \mathbf{V}.
$$`

In code, this looks like this:
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
```

GPT-3 uses multi-head attention, which means we do the following:

### 4) FFN

This is just a linear layer.

### 5) Add and Norm

We use a residual connection and then run layer norm.

### 5) Decode

At the end, we have a big word embedding matrix.  We decode by multiplying by the transpose of `$W_E$`.


### Demo:

