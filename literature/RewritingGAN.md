# Rewriting a Deep Generative Model by Bau et al.

* GANs learns an implicit distribution, or *rules* about the target distribution
* *Rules* are encoded in layers of GAN
* Treat layers of GAN as linear associative memory -> novel way to manipulate the memory, therefore the encoded *rules*

## Why is rewriting useful

By changing the rules in the model, all images produced will be affected. This is a more general way for the user to manipulate the model. The example given in this paper is that by changing the rule for drawing tower tops to draw trees, anytime a tower top is to be drawn, trees will instead be drawn. Observing these changes allows the user to find what the model has learned about tower tops. This general way of editing generated data is much more efficient than editing individual datum.

In short, edit the *generator* instead of the *images*. This allows an infinite set of potential images, allowing the transfer of edits to many images. The example from the paper removes watermarks from images, this can be tedious when done individually, but by making it a rule, the model is able to remove all watermarks. What this means is, say the model has learned what a watermark is (not the contribution), by rewriting the model to remove this learned feature (remove rule, this is the contribution), it can generally apply this to any image with watermarks.

*Model Rewriting* adds, removes, or alters the semantic and physical rules of a pretrained deep network.

A key contribution of this paper is how to perform *model rewriting*. Initially it is described as 'generalizing the idea of a linear associative memory to a nonlinear convolutional layer of a deep generator'. I need to better understand what a linear associative memory (LAM) is, and why that is beneficial for model rewriting.

[*Latent Space*](https://www.quora.com/What-is-the-meaning-of-latent-space) is the space where features lie. In machine learning, we map the input in input space to latent space to extract meaningful structure. In image processing, images of chairs, will have different values in input space, but the similarity in features is captured in latent space and can used to classify all chairs. Feature extraction is mapping input space to latent space.

When rewriting a model, we want to modify a specific rule (through parameters), while keeping the rest of the parameters (not relating to the rule) the same. The initial approach in this paper is to provide a target image, and optimize the parameters so that the rule is applied while keeping the rest of the parameters the same; 'Changing a rule with minimal collateral damage'. The optimization objective then is to find parameters such that they 1) remain largely the same, while 2) better fitting the target.

$$ \theta_{1} = \arg\min_{\theta} \mathcal{L}_{smooth}(\theta) + \lambda\mathcal{L}_{constraint}(\theta)$$

$$\mathcal{L}_{smooth}(\theta) = \mathbb{E}_{z}[\ell(G(z;\theta_{0}),G(z;\theta))]$$

$$\lambda\mathcal{L}_{constraint}(\theta) = \sum_{i} \ell(x_{*i}, G(z_{i};\theta))$$
