# Automatic Differentiation

## Derviatives

### Univariate, Scalar output

Let $$\phi : \mathbb{R} \rightarrow \mathbb{R}$$ be a real valued function of one variable with the first derivative defined by

$$\phi'(\alpha)=\frac{\partial{\phi}}{\partial{\alpha}}=\lim_{\epsilon \rightarrow 0} \frac{\phi(\alpha+\epsilon)-\phi(\alpha)}{\epsilon}$$

and the second derivative (by applying the derviative to $$\phi'(\alpha)$$)

$$\phi''(\alpha)=\frac{\partial^2{\phi}}{\partial{\alpha}^2}=\lim_{\epsilon \rightarrow 0} \frac{\phi'(\alpha+\epsilon)-\phi'(\alpha)}{\epsilon}$$

To illustrate the chain rule, suppose $$\alpha$$ depends on a value $$\beta$$. So we have

$$\phi(\alpha(\beta))$$

Then the derivative of $$\phi$$ with respect to $$\beta$$ is defined by

$$\frac{\partial\phi(\alpha(\beta))}{\partial(\beta)}=\frac{\partial\phi}{\partial\alpha}\frac{\partial\alpha}{\partial\beta}=\alpha'(\beta)\phi'(\alpha)$$

### Multivariate, Scalar output

Let $$f:\mathbb{R}^n\rightarrow\mathbb{R}$$ be a real valued function of $$n$$ independent variables, $$x=(x_1,x_2,\ldots,x_n)^{T}$$.

$$f$$ is differentiable at $$x$$ if there exists a vector $$g\in\mathbb{R}^n$$ such that

$$\lim_{y \rightarrow 0}\frac{f(x+y)-f(x)-g^{T}y}{||y||}=0$$

Where $$f(x+y)-f(x)$$ is equivalent to taking a step in the direction $$y$$, and subtacting the change in $$f$$ in direction $$y$$, $$g^Ty$$ (directional derivative).

$$\nabla f(x) = g$$ is called the gradient of $$f$$ if the above holds.

$$\nabla f(x) = \begin{bmatrix}
                    \frac{\partial f}{\partial x_1} \\
                    \frac{\partial f}{\partial x_2} \\
                    \vdots \\
                    \frac{\partial f}{\partial x_n}
                \end{bmatrix}$$

### Multivariate, Vector output

Suppose now we have a *vector valued* function $$f:\mathbb{R}^n \rightarrow \mathbb{R}^m$$ with $$n$$ independent variables. Then

$$\nabla f(x) \in \mathbb{R}^n \rightarrow \mathbb{R}^m \quad s.t. \quad \nabla f_i(x) = \frac{\partial f_i}{\partial x}$$

<!-- The Jacobian is a matrix of partial derviatives, with each element defined by

$$J_{ij}(x)=\frac{\partial f_i}{\partial x_j}$$ -->

Let $$x=x(t)$$, then we have $$h(t)=f(x(t))$$. Then the gradient of $$f$$ with respect to $$t$$ is

$$\nabla h(t)=\sum^n_{i=1}\frac{\partial f}{\partial x_i}\nabla x_i(t)=\nabla x(t) \nabla f(x(t))$$

recall in the univariate case we had

$$\frac{\partial\phi(\alpha(\beta))}{\partial(\beta)}=\frac{\partial\phi}{\partial\alpha}\frac{\partial\alpha}{\partial\beta}=\alpha'(\beta)\phi'(\alpha)$$

Make the connection.

### Forward Mode
