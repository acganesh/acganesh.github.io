---
layout: post
title: An introduction to diffusion models
date: 2024-05-16
math: true
description: Exploring how diffusion models work.
---

(Work in progress)

In this post, we'll explore how diffusion models work.  This post will follow the presentation in ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/pdf/2006.11239).

# How do diffusion models work?

Diffusion models are Markov chains trained to produce samples matching the data after finite time.  We start with a diffusion process that destroys the data: we apply multiple Markov chain transitions until a datapoint `$\mathbf{x}_0$` gets gradually converted to noise.  Diffusion models learn transitions that reverse this process. 

Formally, diffusion models are latent variables of the form:
`$$p_{\theta}(\mathbf{x}_0) = \int p_{\theta}(\mathbf{x}_{0:T}) d \mathbf{x_{1:T}}$$`

Here, `$\mathbf{x}_1, \dots, \mathbf{x}_T$` are latent variable models which are the same dimensionality as the data `$\mathbf{x}_0 \sim q(\mathbf{x}_0)$`.

We call `$p_{\theta} (\mathbf{x}_{0:T})$` the reverse process, which is defined as follows:

`$$p_{\theta}(x_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_{\theta} (\mathbf{x}_{t-1} | \mathbf{x}_t),$$`
where the conditional probabilities are modeled as Gaussians:
`$$
p_{\theta}(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N} (\mathbf{x}_{t-1}; \mathbf{u}_{\theta}(\mathbf{x}_t, t), \Sigma_{\theta}(\mathbf{x}_t, t)).
$$`

The diffusion process is defined by a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule `$\beta_1, \dots, \beta_T$`.

`$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1}),
$$`
where 

`$$
q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right):=\mathcal{N}\left(\mathbf{x}_t ; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}\right).
$$`

The usual approach would be to optimize this lower bound of the negative log likelihood:

`$$
\mathbb{E}\left[-\log p_\theta\left(\mathbf{x}_0\right)\right] \leq \mathbb{E}_q\left[-\log \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\right]=\mathbb{E}_q\left[-\log p\left(\mathbf{x}_T\right)-\sum_{t \geq 1} \log \frac{p_\theta\left(\mathbf{x}_{t-1} \mid \mathbf{x}_t\right)}{q\left(\mathbf{x}_t \mid \mathbf{x}_{t-1}\right)}\right]=: L
$$`


