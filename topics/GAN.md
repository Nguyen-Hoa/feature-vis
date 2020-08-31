# NIPS 2016 Tutorial: Generative Adversarial Networks (GAN)

## How Do Generative Models Work

### Density Estimation

Given a set of training points sampled from data generating distribution $$p_{data}$$, find a model, $$p_{model}$$, which estimates $$p_{data}$$. The exact distribution of $$p_{data}$$ is unknown, and we wish to estimate it from the training examples. If $$p_{data}$$ lies within $$p_{model}$$, it can be recovered exactly. *See figure 1.*

The distribution can be estimated *explicitly*, so that the entire estimated distribution is known, or *implicitly*, where the model generates samples from the distribution based on a datapoint.

GANs perform density estimation, making it more suitable for multi-modal problems. Compared to a network that uses *Mean Squared Error (MSE)*, which is more suitable for single correct answers by taking the average of the best solution, GANs are able to take the best sample from a distribution. *See figure 3*.

---

### Maximum Likelihood

One way to perform *density estimation* is through *maximum likelihood*. The likelihood is the probability that the model assigns the training data. So maximizing the likelihood yields a better model.

For $$m$$ training examples and model parameters $$\theta$$, *likelihood* is defined by:

$$\prod_{i=1}^{m} p_{model}(x^{(i)}, \theta)$$

In *maximum likelihood* we wish to find parameters $$\theta^{*}$$ such that the model maximizes the likelihood of the training data.

$$\theta^{*} = \arg\max_{\theta} \prod_{i=1}^{m}p_{model}(x^{(i)}, \theta)$$

Recall that applying $$\log$$ does not change the location of the maximum. So performing the transforming the equation using logs results in a sum rather than products, which reduces computational errors such as underflow.

$$\arg\max_{v} f(v)$$ == $$\arg\max_{v} \log f(v)$$

$$\theta^{*} = \arg\max_{\theta} \sum_{i=1}^{m} \log p_{model}(x^{(i)}, \theta)$$

### KL Divergence

Another way to perform density estimation is by minimizing the *KL Divergence* between $$p_{data}$$ and $$p_{model}$$.

$$\theta^{*} = \arg\min_{\theta} D_{KL}(p_{data}(x) || p_{model}(x, \theta))$$

The training samples are used to approximate $$p_{data}$$ with an empirical distribution, denoted $$\hat{p}_{data}$$, where mass is placed only where the samples are available.

*Minimizing KL Divergence is exactly equal to maximizing log likelihood of the training set.*

$$\arg\max_{\theta} \sum_{i=1}^{m} \log p_{model}(x^{(i)}, \theta) == \arg\min_{\theta} D_{KL}(p_{data}(x) || p_{model}(x, \theta))$$
