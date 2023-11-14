---
layout: post
title: Speculative decoding for LLM inference
date: 2023-11-07
math: true
description: When two language models are faster than one.
---

Speculative decoding is a neat trick that provides significant speedups to LLM inference.  This post was originally inspired by this Karpathy [tweet](https://twitter.com/karpathy/status/1697318534555336961) -- I will aim to discuss how the algorithm works in more detail and prove its correctness.

The idea is based on the following facts:
- LLM inference for batch size `$k$` takes approximately the same time as inference for batch size `$1$`, for surprisingly large values of `$k$`.  In particular, for low batch sizes, LLMs are bound by memory-bandwidth.
- Many tokens at inference time are easy, and can be predicted by a cheap model.  For example, predicting `for i in range(N)` in Python code, or predicting articles like `the` or `a` required for grammatical correctness.

The main algorithm does the following:

- Sample a candidate sequence of `$K$` tokens from a smaller cheap model.
- Run the cheap model's predictions through the main model with batch size `$K$` to obtain logits. 
- Skip forward through all tokens that agree with the draft model.  Once we hit a disagreement with the target model, throw the remaining draft tokens away and repeat.

Here's an example of what this looks like in practice:

![Speculative decoding example](/img/spec-example.png)


# Precursor: LLMs are bound by memory-bandwidth at inference time

Below is the hierarchy of memory on a system with a CPU and A100 GPU.  {{< marginnote >}}Source: The [FlashAttention paper](https://arxiv.org/pdf/2205.14135.pdf).{{< /marginnote >}}

<img src="/img/a100-hierarchy.png" alt="A100 hierarchy" style="width:50%;">

The key mental model for GPUs is that we need to move data from high-bandwidth memory (HBM) to static random-access memory (SRAM), where computation occurs.  Some relevant stats for an A100-40GB are below.  As can be seen, GPU compute has grown significantly faster than memory bandwidth.

- HBM: 40GB
- SRAM: 40MB
- Memory bandwidth: 1935 GB/s
- Compute: 312 TFLOPS with FP16

This [post](https://finbarr.ca/how-is-llama-cpp-possible/#fn:3) does analysis that breaks down the latency incurred by memory-bandwith and by compute.  Let `$P$` be the number of parameters in a language model.  Let `$n_{\text{bytes}}$` denote the number of bytes in each number (16 for FP16, 8 for INT8, etc.).  Let `$B$` be the batch size.  It turns out we can express the latency incurred by compute and memory bandwidth as follows:

`$$
\begin{aligned} \text { latency }_{\text {model }} & =\max \left(\text { latency }_{\text {compute }}, \text { latency }_{\text {memory }}\right) \\ \text { latency }_{\text {memory }} & =\frac{2 \cdot P \cdot n_{\text {bytes }}}{n_{\text {memory bandwidth }}}, \\ \text { latency }_{\text {compute }} & =\frac{2 \cdot P \cdot B}{n_{\text {flops }}}\end{aligned}
$$`

Plugging in the above values for memory bandwidth and FLOPs, Finbarr Timbers [concludes](https://finbarr.ca/how-is-llama-cpp-possible/#fn:3) that memory bandwidth dominates compute latency for batch sizes smaller than 161.

# Going through the algorithm in detail

The two main references that describe speculative decoding are [Chen et al. 2023](https://arxiv.org/abs/2302.01318) and [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192).  I will focus on the presentation in the first paper as I found it easier to read.

The algorithm works as follows.

<style>
.algorithm p {
  line-height: .25;
}
</style>


**Algorithm -- Speculative Decoding** 

Inputs:
- High latency target model `$q$`.
- Low latency draft model `$p$`.
- Initial prompt `$x_0, \dots, x_t$`.
- Lookahead `$K$`.
- Minimum target sequence length `$T$`.

<div class="algorithm">

`$\text{Initialize } n \leftarrow t.$`

`$\textbf{while } n < T \textbf{ do}$`

`$\quad \textbf{for } t = 1 : K \textbf{ do}$`

`$ \quad \quad \text{Sample draft auto-regressively  } \tilde{x}_t \sim p(x|\,x_1, \ldots, x_n, \tilde{x}_1, \ldots, \tilde{x}_{t-1}) $`

`$ \quad \textbf{end for} $`

`$ \quad \text{In parallel, compute  } K + 1 \text{ sets of logits from drafts  } \tilde{x}_1, \ldots, \tilde{x}_K : $`

`$ \quad q(x|\,x_1, \ldots, x_n), q(x|\,x_1, \ldots, x_n, \tilde{x}_1), \ldots, q(x|\,x_1, \ldots, x_n, \tilde{x}_1, \ldots, \tilde{x}_K) $`

`$ \quad \textbf{for} \text{  } t = 1 : K \textbf{do} $`

`$ \quad \quad \text{Sample  } r \sim U[0, 1] \text{ from a uniform distribution.} $`

`$ \quad \quad \textbf{if} \text{  } r < \min \left( 1, \frac{q(x|\,x_1, \ldots, x_{n+t-1})}{p(x|\,x_1, \ldots, x_{n+t-1})} \right), \textbf{then} $`

`$ \quad \quad \quad \text{Set  } x_{n+t} \leftarrow \tilde{x}_t \text{ and  } n \leftarrow n + 1. $`

`$ \quad \quad \textbf{else} $`

`$ \quad \quad \quad \text{Sample  } x_{n+t} \sim (q(x|\,x_1, \ldots, x_{n+t-1}) - p(x|\,x_1, \ldots, x_{n+t-1})) \text{ and exit for loop.} $`

`$ \quad \quad \textbf{end if} $`

`$ \quad \textbf{end for} $`

`$ \quad \textbf{if} \text{ all tokens  } x_{n+1}, \ldots, x_{n+K} \text{ are accepted, sample extra token  } x_{n+K+1} \sim q(x|\,x_1, \ldots, x_n, x_{n+K}) \text{ and} $`

`$ \quad \text{set  } n \leftarrow n + 1. $`

`$ \textbf{end while} $`

</div>

## Proof of correctness

Let's prove Theorem 1 from the paper, which states the following:

**Theorem 1.**   Modified rejection sampling recovers the target distribution.

As above, let `$q$` be the draft model, and `$p$` be the target model.  Let `$X$` be the final sample produced by the algorithm above.  We will show that `$P(X = x)$` is equal to `$p(x)$`.

The first step is to break `$P(X = x)$` into two cases.  Either `$\tilde{x} = x$` and we accept the draft sample, or we reject it and resample.


So we can write:

`$$
\begin{aligned}
P(X = x) &= P(\tilde{x} = x) P(\tilde{x} \text{ accepted} | \tilde{x} = x) + P(\tilde{x} \text{ rejected}) P(X = x | \tilde{x} \text{ rejected})
\end{aligned}
$$`

Let's calculate each of the two terms.  We will start with the acceptance probability.

`$$
\begin{aligned}
P(\tilde{x} = x) P(\tilde{x} \text{ accepted} | \tilde{x} = x) &= p(x) \min \left ( \frac{q(x)}{p(x)}  \right ) \\
&= \min ( p(x), q(x)),
\end{aligned}
$$`

where we have used the fact that we can multiply through the `$\min$` operator.

For the next term, we can calculate as follows:
`$$
\begin{align}
P(\tilde{x} \text { rejected})&=1-P(\tilde{x} \text { accepted}) \\
&=1-\sum_{x^{\prime}} P\left(X=x^{\prime}, \tilde{x} \text { accepted}\right) \\
&=1-\sum_{x^{\prime}} \min \left(p\left(x^{\prime}\right), q\left(x^{\prime}\right)\right) \\
&=\sum_{x^{\prime}} q\left(x^{\prime}\right)-\min \left(p\left(x^{\prime}\right), q\left(x^{\prime}\right)\right) \\
&=\sum_{x^{\prime}} \max \left(0, q\left(x^{\prime}\right)-p\left(x^{\prime}\right)\right)
\end{align}
$$`

Also, from the algorithm above, we have that:
`$$
P(X = x | \tilde{x} \text{ rejected}) = \frac{\max(0, q(x) - p(x))}{\sum_{x'} \max \left( 0, q(x') - p(x') \right)}
$$`

Multiplying through, we have:

`$$
\begin{align}
& P(\tilde{x} = x) P(\tilde{x} \text{ accepted} | \tilde{x} = x) + P(\tilde{x} \text{ rejected}) P(X = x | \tilde{x} \text{ rejected}) \\
 &= \min (p(x), q(x)) + \left ( \frac{\max(0, q(x) - p(x))}{\sum_{x'} \max \left( 0, q(x') - p(x') \right)} \right ) \sum_{x^{\prime}} \max \left(0, q\left(x^{\prime}\right)-p\left(x^{\prime}\right)\right) \\
&= \min (p(x), q(x)) + \max (0, q(x) - p(x)) \\
&= q(x).
\end{align}
$$`

This finishes the proof of Theorem 1.

## Inference Latency Results

![Speculative decoding latency](/img/speculative-latency.png)

The referenced DeepMind paper tried this technique on Chinchilla 70B.  They find an approximate speedup of 2x, which is a great improvement.  Interestingly, the speedup differs by domain, which makes sense, because different domains may have different frequencies of "easy tokens."





