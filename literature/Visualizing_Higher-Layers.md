# Visualizing Higher-Layer Features of a Deep Network by Erhan et al.

The paper is an early (2009) introduction to neuron visualization through activation maximization. It applies two neuron visualization techniques to two kinds of neural networks: Deep Belief Networks (DBN) and Stacked Denoising Autoencoders (SDAE). Although these networks experiemented on are not the convolutional networks found in more recent papers, the technique used to excite and interpret neurons is largely the same. The two techniques discussed, activation maximization and sampling a unit, in this paper have differences based on the network they are applied to, DBN and SDAE respectively.

Say we want to visualize (see what most excites) neuron $$i$$ at layer $$j$$. We can do so by simply finding what inputs to that neuron leads to the highest activation. If $$j = 1$$, or the neuron lies in the first layer, whose input is the raw image, finding the input that maximizes its activation yields a result in image space. However, for neurons in deeper layers, whose input comes from other neurons, mapping the maximum activation to image space requires more computation, namely considering the derivatives from the first layer all the way down to the target neuron. This is typically done during training of the network, where the derviative with respect to the parameters are computed to minimize loss later in the network. This is more easily understood if we view a neural network as a single function, composed of many functions, each representing a layer.

$$ f_{network}(x) = f_{n}(f_{n-1}(\cdots(x))$$

Here $$f_{network}$$ takes $$x$$ as input, and applies the compsition of $$n$$ functions to get an output or prediction. After a network is trained, we can fix the parameters (weights and biases) and compute the derivative with respect to an input $$x$$ to find $$x^*$$, an input which maximizes a particular neuron.

## Activation maximization

$$ x^{*} = \arg\max_{x} h_{ij}(\theta,x), \quad x \quad s.t. ||x|| = \rho$$

Here we have $$h_{ij}$$, a function of parameters $$\theta$$ and input $$x$$, representing the activation of neuron $$i$$ at layer $$j$$.Assuming $$\theta$$ is fixed, we can solve this non-convex optimization problem via gradient ascent, with a step defined by:

$$ x_{k+1} = x_k + \epsilon\nabla_{x}h_{ij}(x)$$

Where $$\epsilon$$ is the step size and $$\nabla_{x}h_{ij}(x)$$ is the gradient of the activation with respect to $$x$$.
