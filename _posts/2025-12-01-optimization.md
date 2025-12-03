---
layout: post
title: 'A short note on optimizing NNs'
date: 2025-10-31
categories: [research, blog]
---
> TLDR: A technical blog to revisit the fundamentals of what, in the crudest sense, makes Deep Learning work. A SGD-to-Muon tour, derived from first principles in math and then implemented from scratch in Jax.

### Table of Contents
- [Implementing SGD](#implementing-sgd)
- [Adam](#adam)
  - [Momentum](#momentum)
  - [Adaptive Learning Rates](#adaptive-learning-rates)
  - [Bias Correction](#bias-correction)
  - [Implementing Adam](#implementing-adam)
- [AdamW](#adamw)
  - [Implementing AdamW](#implementing-adamw)
- [Muon](#muon)
  - [Implementing Muon](#implementing-muon)

Take a NN parametrized with parameters $\theta$. 
During training, the parameters are updated using differential information relating the performance obtained to the weights used, i.e. using $\nabla L (\theta) = \sum_{i \in \mathcal{D}} \nabla \ell_i (\theta)$, so that weights are iteratively updated according to:

$$ \theta_{t} \leftarrow f (\theta_{t-1}, \nabla L (\theta_{t-1})), $$

where $f$ is some function of the weights $\theta_{t-1}$ and gradients $\nabla L (\theta_{t-1})$.

For both **conceptual** and **computational** reason, one typically does not use the exact gradient of the loss $\nabla L (\theta)$, and rather relies on $\tfrac{1}{\vert \mathcal B \vert } \sum_{i \in \mathcal{B}} \nabla \ell_i (\theta)$, referred to as the *stochastic gradient* for the mini-batch $\mathcal B \subset \mathcal D: \mathcal B \sim \mathcal D$.
On a conceptual level, stochastic gradients suffer less from poor initialization than their deterministic counterpart, which proves particularly useful in the context of non-convex optimization.
Computationally, estimating the full gradient requires processing *all* the samples in $\mathcal D$ through the network at all times, which is simply prohibitive for large-scale datasets, resulting in the--purely computational--need to process mini-batches $\mathcal B \subset \mathcal D: \vert \mathcal B \vert \ll \vert \mathcal D \vert$.
Note how SGD still performs an update for the entire parameter vector $\theta$, although it exclusively relies on limited information regarding $\mathcal D$, in particular using $\mathcal B \subset \mathcal D$.

### Implementing SGD <a id="implementing-sgd"></a>
```python
from typing import TypeAlias
import jax.numpy as jnp

# I like typing! It makes everything clearer to read :)
ModelParameters: TypeAlias = dict[str, dict[str, jnp.ndarray]]
Gradients: TypeAlias = dict[str, dict[str, jnp.ndarray]]
OptimizerState: TypeAlias = dict[str, dict[str, jnp.ndarray]]

# We should never divide by 0, should we
EPSILON = 1e-12

def sgd_update(
    params: ModelParameters, 
    grad: Gradients, 
    learning_rate: float
) -> OptimizerState:

    return {
        "params": jax.tree.map(
            lambda p, g: p - learning_rate * g, params, grad
        )
    }
```

In most practical scenarios, researchers do not use SGD anymore in favour of more advanced optimizers.
Today, the most widely used optimizer in practice is *Adam*, or its weight-decay variant, *AdamW*.

## [Adam](https://arxiv.org/pdf/1412.6980) <a id="adam"></a>

Start from the end: the [infamous Adam](https://x.com/2prime_PKU/status/1948549824594485696) update rule proposes weights to be updated using:

$$
\theta_{t} = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

There are multiple aspects in this update rule.
Together, they make Adam a best-of-both-worlds optimization algorithm when it comes down to combining (1) momentum, ${m}_t$ (2) adaptive learning rates, ${v}_t$ and (3) bias corrections, $\hat{\bullet}$.
Implementing Adam's update rule is relatively straightforward. In this note, I will cover all these fundamental *concepts* behind Adam, unpacking its update rule and hinting at what might be coming next (loaded spoiler: [*Muon*](https://kellerjordan.github.io/posts/muon/)).

### Momentum, or $m_t$ <a id="momentum"></a>

The intuition behind momentum is to reuse previous differential information to improve and stabilize optimization. 
In that, momentum typically smoothens the trajectory of more standard SGD by aggregating previous ($1, \dots, \tau<t$) gradients into the timestep-$t$ update.

In practice, by defining a coefficient $\beta$ regulating the relevance of *past* gradients when forming the *current* update, one can derive a modified update rule defined as:

$$
\begin{align*}
\theta_{t} &= \theta_{t-1} - \eta \cdot m_t , \quad m_t = \beta m_{t-1} + (1-\beta) g_t \\
g_t &= \tfrac{1}{\vert \mathcal B \vert} \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_{t-1})
\end{align*}
$$

This update rule maintains previous gradients relevant according to the parameter $\beta$: for $\beta \to 1$ previous gradients dominate the current gradient estimate, whereas for $\beta \to 0$ it is the current gradient to have the most impact on the parameter update.
Momentum was first introduced by the Soviet mathematician Polyak in the 1960s.
Alternatively, $m_t = \beta m_{t-1} + g_t$ is also a valid momentum formulation.

#### Implementing Momentum
```python
def momentum_update(
    params: ModelParameters, 
    grad: Gradients, 
    momentum: Gradients, 
    beta: float, 
    learning_rate: float
) -> OptimizerState:

    momentum_updated = jax.tree.map(
        lambda m, g: beta * m + (1 - beta) * g, momentum, grad
    )

    return {
        "params": jax.tree.map(
            lambda p, m: p - learning_rate * m, params, momentum_updated),
        # Optimizer state
        "momentum": momentum_updated,
        "beta": beta
    }
```

#### Nesterov Momentum
A popular variant of momentum is Nesterov-accelerated momentum. Differently from Polyak's momentum, Nesterov's acceleration uses the momentum $m_t$ as a coarse approximation for $g_t$, and critically only leverages differential information to adjust said approximation *after* having performed a parameter update. Formally,

$$
\begin{align*}
\theta_{t} &= \theta_{t-1} - \eta \cdot m_t , \quad m_t = \beta m_{t-1} + (1-\beta) g_t \\
g_t &= \tfrac{1}{\vert \mathcal B \vert } \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_{t-1} - \eta \beta m_{t-1})
\end{align*}
$$

Effectively, by using $\ell_k (\theta_{t-1} - \eta \beta m_{t-1})$ in the parameter update, gradient information is employed to perform corrections to the direction of the accumulated momentum.

Crucially, while both standard and Nesterov momentum naturally accomodate for a possibly time-dependant learning rate $\eta = \eta_t$, momentum still uses an *equal learning rate for all parameters*, resulting in the need to perform significant tuning of $\eta$ to improve practical performance.

### Adaptive Learning Rates, or $v_t$ <a id="adaptive-learning-rates"></a>
Momentum proves useful in guaranteeing smoother, more stable optimization routines in practice, embedding inertia into the optimization process by reusing differential information collected earlier in the training process.
However, it tragically suffers from the need to sensitivity to hyper-parameters, including both the learning rate $\eta$ and momentum factor $\beta$.
While hyperparameter tuning is oftentimes simply necessary to have obtain good performance, in the 2010s many works ([AdaGrad, 2011](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), [Adadelta, 2012](https://arxiv.org/pdf/1212.5701), [RMSProp, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)) set out to reduce the dependancy of the optimization process on the identification of a "good" learning rate (who would even launch training with a bad one?), proposing *adaptive scalers* $v_t$ of a given initial learning rate $\eta$.

Different in how previous gradient information is used, AdaGrad, Adadelta and RMSProp all rely on normalizing the learning rate $\eta$ per parameter by the scale of the updates received during training until $t$.
In this, the intuition behind the different methods is that parameters that receive updates less often (i.e., parameters which stay closer to their initialized value during training) should--to improve on convergence--use larger stepsizes $\eta$ than parameters which receive updates often during training, which in turn should--to increase stability--be updated less drastically.
Therefore, given a measure of variability until a given training timestep $t$, learning rates should be scaled *up* given *small $v_t$*, and *down* given *large $v_t$*.

Formally, such intuition results in an update rule like:

$$
\begin{align*}
\theta_{t} &= \theta_{t-1} - 
\frac{\eta}{v_t} g_t, \\
g_t &= \tfrac{1}{\vert \mathcal B \vert} \sum_{k \in \mathcal{B}} \nabla \ell_k (\theta_{t-1}),
\end{align*}
$$

where the term $v_t \in \mathbb R^d \ni \theta_t \, \forall t$ is used to scale the learning rate per-parameter $\theta_{t,i} \in \theta_t$.

**AdaGrad** uses the sum of the squared gradients up to $t$ to scale the learning rate.
Formally,

$$
\begin{align*}
v_t &= \operatorname{diag}(G_t)^{\tfrac12}, \implies \theta_{t} = \theta_{t-1} - 
\eta \operatorname{diag}(G_t)^{-\tfrac 12} \odot g_t, \, \tag{AdaGrad} \\
G_t &= \sum_{i=1}^t g_i g_i^\top = G_{t-1} + g_t g_t^\top
\end{align*}
$$

The matrix $G_t$ is formed iteratively via a sequence of rank-1 updates, and serves as *an accumulator* of the information contained in the updates up to timestep $t$. 
In particular it can be understood as a *measure of the magnitude of per-parameter updates* up to $t$.
Indeed, considering the $j$-th parameter in $\operatorname{diag}G_t$ is the same as measuring the Root Mean Square (RMS) of the variations that intervened on that very same $j$ up to a multipliticative factor depending on $t$--in particular, $\sqrt(t)$.
This result follows from:
$$
\operatorname{RMS}(g_1, g_2 \dots, g_t) = \sqrt{\frac 1t \sum_{i=1}^t g_i^2 } \implies \operatorname{diag}(G_t)^{\tfrac 12} = \sqrt{t} \cdot \operatorname{RMS}(g_1, g_2 \dots, g_t)
$$

By scaling the learning rate for parameter $j$, $\eta$, by $\sqrt{t} \cdot \operatorname{RMS}((g_1)_j, (g_2)_j, \dots, (g_t)_j)$ one has that, at the same point in training (i.e., for the same $t$), less frequently updated parameters (for which the RMS tends to be smaller) receive *larger* updates compared to more often updated parameters, for which the RMS of previous gradients is larger.

#### Implementing AdaGrad
```python
def adagrad_update(
    params: ModelParameters, 
    grad: Gradients, 
    gsquare: Gradients, 
    learning_rate: float
) -> OptimizerState:
    
    gsquare = jax.tree.map(
        lambda old_g, g: old_g + g**2, gsquare, grad
    )

    return {
        "params": jax.tree.map(
            lambda p, gs, g: p - (learning_rate/(jnp.sqrt(gs) + EPSILON)) * g, 
            params, gsquare, grad
        ),
        # Optimizer state
        "gsquare": gsquare
    }

```

However, when computing $G_t = \sum_{i=1}^t g_i g_i^\top$ all past gradients have similar weight across $1, \dots, t$. 
Because earlier in the training procedure gradients are likely to be large due to initialization, considering all gradients equally may result in an excessive shrinking of the learning rate, which ultimately hinders performance.

**RMSProp** directly addresses the shrinking learning rate phenomenon by maintaining a (soft) receptive field of $\frac{1}{1-\gamma}$ steps, forming $G_t$ according to:

$$
G_t = \gamma \cdot G_{t-1} + (1-\gamma) \cdot g_t g_t^\top.
$$
This choice follows from the intuition that more recent gradients are to be preferred over older ones, as older differential information can be considered less relevant over the course of training.
In turn, RMSProp effectively maintains the summation of squared gradients more aligned with the current optimization state, and mitigates the aforementioned excessive shrinking of the learning rate.

Differently from AdaGrad, RMSProp is less sensitive to poor initialization, and just like AdaGrad it retains the need to define a global learning rate $\eta$ to be scaled down according to $v_t$.

#### Implementing RMSProp
```python
def rmsprop_update(
    params: ModelParameters, 
    grad: Gradients, 
    gsquare: Gradients, 
    learning_rate: float, 
    gamma: float
) -> OptimizerState:

    windowed_gsquare = jax.tree.map(
        lambda old_g, g: gamma * old_g + (1 - gamma) * g**2, gsquare, grad
    )

    return {
        "params": jax.tree.map(
            lambda p, gs, g: p - (learning_rate/(jnp.sqrt(gs) + EPSILON)) * g, 
            params, windowed_gsquare, grad
        ),
        # Optimizer state
        "gsquare": windowed_gsquare,
        "gamma": gamma
    }
```

> Sidenote: **AdaDelta** is another optimization algorithm that learns *without* defining a global learning rate. In that, it maintains a running average of the square parameter update, and uses it alongside the RMSProp-like average of square gradients to completely sidestep the need to define a global learning rate $\eta$.

### Bias correction, or $\hat{\bullet}$ <a id="bias-correction"></a>

Both momentum $m_t$ and learning rate scalers $v_t$ are typically initialized as vectors of all zeros $m_0 = v_0 = \mathbf{0}$ in practice.
This results in rather biased (small) estimates for both $m_t$ and $v_t$, especially early on in the training process.
Critically, this source of bias might complicate the optimization process.

In Adam, Kingma et al. propose bias-correcting the current estimate for $m_t$ and $v_t$ using the momentum coefficient $\beta$ and "forgetting-factor" $\gamma$.
Formally,

$$
\begin{align*}
\hat{m}_t &= \frac{m_t}{1 - \beta^t} \implies \hat{m}_t \xrightarrow[]{t \to \infty} m_t \\
\hat{v}_t &= \frac{v_t}{1 - \gamma^t} \implies \hat{v}_t \xrightarrow[]{t \to \infty} v_t\\
\end{align*}
$$

Together with Momentum and the RMSProp update, bias correction fully describes the Adam update rule: momentum keeps previous gradient information influencing the optimization process, while RMSProp scales the learning rate to allow an efficient exploration of the parameter space.
Bias correction ties everything together, improving on otherwise poor initialization of the first ($m_t$) and second ($v_t$) momentum estimates used by Adam.


### Implementing Adam <a id="implementing-adam"></a>
```python
def adam_update(
    params: ModelParameters, 
    grad: Gradients, 
    momentum: Gradients, 
    gsquare: Gradients, 
    beta: float, 
    gamma: float, 
    learning_rate: float, 
    training_step: int
) -> OptimizerState:
    
    # Use 1-based step index for bias correction
    t = training_step + 1
    momentum = jax.tree.map(
        lambda m, g: beta * m + (1 - beta) * g, momentum, grad
    )
    gsquare = jax.tree.map(
        lambda old_g, g: gamma * old_g + (1 - gamma) * g**2, gsquare, grad
    )

    momentum_corrected = jax.tree.map(
        lambda m: m / (1 - beta ** t), momentum
    )

    gsquare_corrected = jax.tree.map(
        lambda gs: gs / (1 - gamma ** t), gsquare
    )

    return {
        "params": jax.tree.map(
            # Adam's update rule!
            lambda p, m, v: p - (learning_rate / (jnp.sqrt(v) + EPSILON)) * m,
            params, momentum_corrected, gsquare_corrected
        ),
        # Optimizer state
        "momentum": momentum,
        "gsquare": gsquare,
        "training_step": training_step,
        "beta": beta,  # beta1 in official implementations
        "gamma": gamma  # beta2 in official implementations
    }
```

## [AdamW](https://arxiv.org/abs/1711.05101) <a id="adamw"></a>

Regularizing neural networks can improve train–test generalization. Besides data augmentation and architectural constraints, a common approach is to add an explicit penalty term  

$$
L(\theta)=\sum_{i\in\mathcal D}\ell_i(\theta)+\lambda\|\theta\|,
$$
where different norms induce different effects (e.g., $\Vert \bullet \Vert_1$ for sparsity, $\Vert \bullet \Vert_2$ for discouraging large weights).

With Adam, however, directly adding an L2 penalty mixes the decay gradient $\lambda\theta$ into the raw gradient  

$$
g_t=\nabla L(\theta)+\lambda\theta,
$$ 
so the moving averages $m_t$ and $v_t$ start tracking not only the data signal but also the decay term. This distorts Adam’s adaptive normalization and leads to parameter-dependent, history-dependent shrinkage—an undesirable side effect of “naïve” L2 when combined with adaptive optimizers.

AdamW addresses this by **decoupling** weight decay from the gradient. 
Instead of adding the regularization term $\lambda\theta$ to the loss, decay is applied directly to the parameters, after computing Adam’s moments on the *data gradient only*.
*Naïvely using L2 with Adam* would result in $g_t = \nabla L(\theta)+\lambda\theta$, inducing $m_t, v_t$ to track both signal _and_ decay.
Conversely, AdamW avoid this by maintaining the gradient exclusively dependent on data $g_t = \nabla L(\theta)$, and explicitly decaying the parameters using a decoupled update rule, 

$$\theta_t = (1-\eta\lambda)\,\theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

In turn, this makes the update explicitly decomposable into (1) a *pure multiplicative decay* $(1-\eta\lambda)\theta$ and (2) a traditional Adam step on the data signal.

#### Implementing AdamW <a id="implementing-adamw"></a>
```python
def adamw_update(
    params: ModelParameters, 
    grad: Gradients, 
    momentum: Gradients, 
    gsquare: Gradients, 
    beta: float, 
    gamma: float, 
    learning_rate: float, 
    training_step: int, 
    lambda_wd: float
) -> OptimizerState:

    """Applies the Adam update and then perform weight regularization"""
    adam_state = adam_update(
        params=params, 
        grad=grad, 
        momentum=momentum, 
        gsquare=gsquare, 
        beta=beta, 
        gamma=gamma, 
        learning_rate=learning_rate, 
        training_step=training_step
    )

    p = jax.tree.map(
        lambda p, old_p: p - learning_rate * lambda_wd * old_p,
        adam_state["params"], 
        params
    )

    # last coming key from weight-decay update overwrites old parameters
    return adam_state | {"params": p, "lambda_wd": lambda_wd}
```

In practice, AdamW will work just fine for your problem and--although not always fully understood--it became the first optimizer anyone tries on their problem (because is good).
If you are curious about something new and are ready for more exotic, a new optimizer has arrived in town (spoiler: Muon!). 
This new optimizer is relatively straightforward to derive at an intuitive level, and is increasing popular on the [best website to be in the loop with the latest in ML](https://x.com).

## The land of no $v_t$, i.e. [Muon](https://jeremybernste.in/writing/deriving-muon) <a id="muon"></a>

Muon ([Keller Jordan's blogpost](https://kellerjordan.github.io/posts/muon/) and [Jeremy Bernstein's blogpost](https://jeremybernste.in/writing/deriving-muon)) addresses a fundamental limitation of practical first-order optimization when training NNs: updates for 2D parameters (i.e., _matrices_--there are a ton in NNs) are oftentimes dominated by specific dimensions in practice.

Formally, this plays out making the update matrix for any set of 2D-parameters $\nabla_W L(\theta), \theta \supset W$ to be (almost) low rank.
In some sense, adaptive learning rates attempt at mitigating this very issue by scaling down the learning rate to prevent too large updates on these few update directions. 
However, adaptive learning rates do ultimately fail at ensuring enough dimensions in the parameter space are updated when the update matrix is empirically low rank due to the parameters (i.e., the rows of $\nabla_W L(\theta)$) being considered *independently*.
Put it simply, there is just no way adaptive learning rates can _fix_ the low-rankness of an update matrix (but they can help).
In practice, low-rankness plays out in the update matrix $\nabla_W L(\theta)$ being more elliptical than spherical--from a spectral perspective, the distribution of its eigenvalues is skewed--resulting in poorer performance over the course of training.
Changing the spectrum of the update matrix so that there are more directions being updated at once--*orthogonalizing*--is precisely what the Muon update rule prescribes:

$$
\begin{align*}
M_t &= \beta M_{t-1} + \nabla_{W_{t-1}} L(\theta) \\
O_t &= \operatorname{Orthogonalize(M_t)} \\
W_t &= W_{t-1} - \eta O_t, \tag{Muon}
\end{align*}
$$

Notice how the Muon update rule is exclusively valid for 2D layers.
A theoretically-justified (yet excessively expensive) orthogonalization technique is Singular Value Decomposition (SVD), resulting in $O_t = U_tV_t^\top$, where the update matrix is SVD-decomposed according to $M_t = U_t \Sigma_t V_t^\top$.
Numerically, SVD rapidly proves prohibitive from a computational standpoint, inducing the need to develop _approximate_ orthogonalization techniques.
One such approximate orthogonalization routine is Netwon-Schultz-$k$ (NS-$k$), which approximates $U_t V_t^\top$ applying $k$ times an odd-matrix polynomial of degree $N$ (typically, $N \in \{3, 5\}$) on $M_t$, resulting in fast, approximate orthogonalization.

Formally, NS-$k$ can be justified considering that given an odd-matrix polynomial of degree $N, \, p_N(X)$ commutes with the SVD decomposition of $X$, resulting in $p_N(U \Sigma V^\top) = U p_N (\Sigma) V^\top$.
Furthermore one can show that 

$$
\underbrace{p_N \circ p_N \dots \circ p_N}_{k \text{ times}}(X) = U p^k_N(\Sigma) V^\top .
$$

Therefore, finding $p: p(\Sigma) \approx I$ and applying it to $X$ is an effective way to orthogonalize the matrix $X$, thereby approximating its orthogonalized form $UV^\top$ (clearly, this reasoning applies to update matrices $M_t$ as well).

Searching for said polynomial is arbitrarily complex, and in practice the process can be simplified by fixing the degree of the polynomial, and applying it $k$ times. 
The degree of the polynomical can also depend on an index $j=1, \dots, k$, so that it can change over time
Together, this results in the Newton-Schultz-$k$ routine used in Muon to orthogonalize the gradient momentum $M_t$.

#### Implementing Muon <a id="implementing-muon"></a>
```python
def orthogonalize(M: jnp.ndarray) -> jnp.ndarray:
    """from https://docs.modula.systems/algorithms/newton-schulz/
    
    Notice how here the polynomial coefficients depends on 
    the index in which polynomials are applied!"""
    
    assert M.ndim == 2, "Orthogonalization is implemented for 2D tensors only!"
    
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / jnp.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    
    if transpose:
        M = M.T
    
    return M

def _muon_update(
    p: jnp.ndarray, 
    g: jnp.ndarray, 
    m: jnp.ndarray, beta: 
    float, learning_rate: float
) -> jnp.ndarray:
    o = orthogonalize(m)
    
    # no adaptive gradients!
    return p - learning_rate * o


def muon_update(
    params: ModelParameters, 
    grad: Gradients, 
    momentum: Gradients,  
    gsquare: Gradients, 
    beta: float, 
    gamma: float, 
    learning_rate: float, 
    training_step: int, 
    lambda_wd: float
) -> OptimizerState:

    def _parameter_update(p, g, m, gs):
        """2D parameters are updated using Muon, other params using AdamW."""
        if p.ndim==2 and g.ndim==2 and m.ndim==2 and gs.ndim==2:
            return _muon_update(p, g, m, beta, learning_rate)
        else:
            return _adamw_update(p, g, m, gs, beta, gamma, learning_rate, training_step, lambda_wd)
        
    momentum = jax.tree.map(
        lambda m, g: beta * m + (1 - beta) * g, momentum, grad
    )
    
    gsquare = jax.tree.map(
        lambda old_g, g: gamma * old_g + (1 - gamma) * g**2, gsquare, grad
    )
    
    return {
        "params": jax.tree.map(
            _parameter_update, params, grad, momentum, gsquare
        ), 
        # Optimizer state
        "momentum": momentum,
        "gsquare": gsquare,
        "training_step": training_step,
        "beta": beta,
        "gamma": gamma,
        "lambda_wd": lambda_wd
    }
```

## Acknowledgments
_Grazie_ to [Joan Velja](https://x.com/joanvelja) and [Francesco Pappone](https://x.com/tensorqt).