# Understanding Neural Networks Through Deep Visualization by Yosinski et al

This paper highlights its two main contributions: a tool to visualize neuron activation values in real time and regularization methods for more interpretable images.

The objective function used to obtain an image representing neuron activation is:

$$x^{*} = \arg\max_x (a_{i}(x) - R_{\theta}(x))$$

Where $$a_{i}(x)$$ is the activation for unit $$i$$ with input $$x$$, and $$R_{\theta}(x)$$ is a regularization term. The paper uses gradient ascent to solve for $$x^*$$ with step directions defined by:

$$x_{k+1} = r_{\theta}(x_{k} + \eta\frac{\partial a_{i}}{\partial x})$$

Where $$\eta$$ is the step size and $$r_{\theta}(\cdot)$$ is an operator that maps $$x$$ to a slightly regularized version of itself.

Since $$x \in \mathbb{R}^{H x W x C}$$ and we know the neural network is differentiable with respect to the input, solving the objective function is a matter of obtaining the output of specific neurons in a neural network. The challenge lies in obtaining useful results, which involves finding a way to make them more user friendly.

## Regularization

Regularization is defined as the process of adding information in order to solve an ill-posed problem or to prevent overfitting. In the context of neuron visualization, where images are often too noisy for visual interpretation, the addition of a regularizing term leads to a more interpretable image. The follow is a list of regularizers the authors found, that when used in combination yields exceptional results.

### L2 Decay

$$r_{\theta}(x) = (1 - {\theta}_{decay})*x$$

Based on the intuition that extreme single pixel values are not natural and do not help with visualization, this regularizer penalizes large values and prevents them from dominating the image.

### Gaussian Blur

$$r_{\theta} = GaussianBlur(x, {\theta}_{width})$$

A result of using gradient ascent (activation maximization?) is that results can have [high frequency](/topics/computer_vision.md) information. By applying a Gaussian filter, the high frequency is penalized, however this method can be costly, so the authors propose applying the filter in intervals and using smaller filters (multiple small width filters have the same effect as one large width filter).

### Clipping Pixels with Small Norm

$$
x_{ij} = \left\{
        \begin{array}{ll}
            0 & ||x_{ij}|| \leq {\theta}_{n\_pct} \\
            x_{ij}
        \end{array}
    \right.
$$

After applying the previous two filters, which suppress high frequency and high amplitude, there will remain some unoptimized (non-zero) pixel values, a result of a non-zero gradient. This leads to an undesired shift in the output pattern. To show only the main object, and let regions not needed be zero, a bias term is added, implemented through a thresholding function that sets any pixel with a small norm to zero. The threshold value $${\theta}_{n\_pct}$$ is specified as a percentile of all pixel norms in x.

### Clipping Pixels with Small Contributions

$$
x_{ij} = \left\{
        \begin{array}{ll}
            0 & |a_{i}(x) - a_{i}(x_{\_j})| \leq {\theta}_{c\_pct} \\
            x_{ij}
        \end{array}
    \right.
$$

Similar to the previous approach, but instead of the norm, the contribution of the pixel is considered. The contribution is measured by taking the absolute value of the difference between the activations for input $$x$$ and $$x_{\_j}$$, where $$x_{\_j}$$ is $$x$$ with the $$j^{th}$$ pixel set to zero. So a pixel with high contribution will result in a larger absolute value. Pixel contributions less than the threshold $${\theta}_{c\_pct}$$ will be clipped (set to zero). Computing the activations in this method require many forward passes (one for each pixel), so the authors propose an approximation by linearlizing $$a_{i}(x)$$ around $$x$$ (taylor approximation?) and then estimating the contribution by an element-wise product of $$x$$ and the gradient.

$$
x_{ij} = \left\{
        \begin{array}{ll}
            0 & |\sum_{c}x \cdot \nabla_{x}a_{i}(x)| \leq {\theta}_{c\_pct} \\
            x_{ij}
        \end{array}
    \right.
$$

$$\nabla_{x}a_{i}(x)$$ represents how much each pixel affects the activation, which is similar to comparing the activation between different input values.
