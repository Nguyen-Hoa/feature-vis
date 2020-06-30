# Generating Images with Perceptual Similarity Metrics based on Deep Networks by Dosovitskiy et al.

This paper introduces metrics (loss functions) that result in richer generated images. The image generator network is composed of three convoloutional networks:

- Generator $$G_{\theta}: \mathbb{R}^{I} \rightarrow \mathbb{R}^{W x H x C}$$.
- Discriminator $$D_{\varphi}$$
- Comparator $$C: \mathbb{R}^{W x H x C} \rightarrow \mathbb{R}^{F}$$

The goal of the network is learn the parameters $$\theta$$ for the differentiable generator $$G_{\theta}$$ that best approximates the dependency between the input $$x$$and target $$y$$. The loss for this task is measured by $$\mathcal{L}(G_{\theta}(x), y)$$. This paper proposes a loss function:

$$\mathcal{L} =
\lambda_{feat}\mathcal{L}_{feat} +
\lambda_{adv}\mathcal{L}_{adv} +
\lambda_{img}\mathcal{L}_{img}
$$

## Loss in Feature Space

$$\mathcal{L}_{feat} = \sum_{i}||C(G_{\theta}(x_{i})) - C(y_{i})||^{2}_{2}$$
