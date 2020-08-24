# NIPS 2016 Tutorial: Generative Adversarial Networks (GAN)

## Density Estimation

*Density Estimation* : Given a set of training points sampled from data generating distribution $$p_{data}$$, find a model $$p_{model}$$ which estimates $$p_{data}$$. The exact distribution of $$p_{data}$$ is unknown, and we wish to estimate it from the training examples. If $$p_{data}$$ lies withing $$p_{model}$$, it can be recovered exactly. *See figure 1.*

The distribution can be estimated *explicitly*, where the entire estimated distribution is known, or *implicitly*, where the model generates samples from the distribution based on a datapoint.

GANs perform density estimation, making it more suitable for multi-modal problems. Compared to a network that uses *Mean Squared Error (MSE)*, which is more suitable for single correct answers because it takes the average of the best solution, GANs are able to take the best sample from a distribution. *See figure 3*.

---
## Maximum Likelihood

One way to perform *density estimation* is through *Maximum Likelihood*. The likelihood is the probability that the model assigns the training data. 