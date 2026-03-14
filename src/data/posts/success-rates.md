> TLDR: Stop reporting the standard deviation around your success rate! In the best case scenario, it is just an expensive way of measuring uncertainty; in the worst, it is a meaningless measure of spread. Use Wilson score intervals instead for a cheap, principled measure of uncertainty around your success rates.

## Summary

In robot learning, the **success rate is often *the* metric**.
This is one of the few cases in machine learning where the central quantity of interest is almost offensively concrete: either the robot succeeds at slotting a battery in or it doesn't; either the gripper lifts the object or it doesn't; the drone either reaches its target, or it crashes on its way there.
No middle ground, and no compromises, and I personally find this beautiful: robotics is so stubbornly outcome-oriented!

To start from the absolute basics, measuring the success rate of a given policy performing a given task requires defining an evaluation protocol, and then running a finite (typically small) number of evaluation rollouts $n$, recording whether each rollout was successful.
If $k$ out of $n$ rollouts succeed, one reports the empirical success rate $p = \frac{k}{n}.$

The point estimate $p$ matters, of course, but it is not enough.
A reported $p = 90\%$ obtained over $10$ rollouts is not the same scientific object as a reported $p = 90\%$ obtained over $100$ rollouts: the first is suggestive, while the second is substantially more informative, let alone on the reliability (or lack of) of the policy which resulted in said 90/100 successful runs.
Therefore, quite reasonably, it is very common to try attaching some measure of spread or uncertainty to $p$, and while correct in spirit (statistical rigor always is!), this step is often enough where things can go very, very wrong.

<img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/uncertainty_with_evidence.png" alt="" style="max-width:100%;height:auto;display:block;margin:2em 0;" />

Across robotics, it is rather common to see the success rate accompanied by a standard deviation, similarily to how spreads around point estimates are reported for other other evaluation metrics in machine learning, like the cumulative return.
Unfortunately, such measure of spread is not only inefficient, but fundamentally wrong.
This blogpost aims at clarifying this point regarding evaluating robot learning policies in a reliable, robust manner.
It is mainly inspired by reading [this tweet](https://x.com/kvablack/status/2001109700151316519?s=20) from Kevin Black (the first author of pi0! Know your robot learning researchers).

## Key takeaway
This blogpost's claim is simple: **for evaluation success rates, the quantity that should usually be reported is a confidence interval for a binomial proportion, not a standard deviation.**
[Wilson Score intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) are a very good default for it.
Skip the rest of this blogpost and use the following snippet if you are in a hurry for CoRL (note: I am, 😅), or stick around for a derivation of why Wilson Score intervals are important and why you should use them too.

You do not need any scientific Python stack for this.
```python
import math
from statistics import NormalDist

# *Don't* compute_gaussian_confidence_interval(success_rates, confidence=0.95) 
# *Do* wilson_score_interval(successes, total_trials, confidence=0.95)

def wilson_score_interval(successes: int, total_trials: int, confidence: float = 0.95):
    if total_trials <= 0:
        return 0.0, 0.0

    n = total_trials
    p_hat = successes / n
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    denominator = 1.0 + (z**2) / n
    center = p_hat + (z**2) / (2.0 * n)
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * n**2))

    lower = (center - radius) / denominator
    upper = (center + radius) / denominator

    return max(0.0, lower), min(1.0, upper)

successes = 9
trials = 10

low, high = wilson_score_interval(successes, trials)
print(f"Success rate: {successes/trials}")
print(f"95% Wilson CI: [{low:.2f}, {high:.2f}]")
```

## Quantifying what quantity's uncertainty one wants to quantify
..."quantity", just because I haven't used this word enough time in this section's title (alright, sorry for the joke).

Let $X_i \in \{0, 1\}$ denote the outcome of rollout $i$, with $X_i = 1$ if the rollout succeeds and $X_i = 0$ otherwise.
Under the usual simplifying assumption that evaluation episodes are conditionally independent given the same policy $\pi$ with weights $\theta$, and that the evaluation environment is stationary at test-time, we can model

$$
X_i \sim \operatorname{Bernoulli}(p),
\qquad
k = \sum_{i=1}^n X_i \sim \operatorname{Binomial}(n, p),
$$

where $p$ is the true, *unknown* probability that $\pi$ succeeds at its task.
When estimating $p$ from experiments, there are multiple distinct sources of variability one needs to be aware of:

1. The variability of individual rollout outcomes, namely $\operatorname{Var}(X_i) = p(1-p)$
2. The variability of the estimator $p = \tfrac{1}{n}\sum_{i=1}^n X_i$, namely $\operatorname{Var}(p) = \tfrac{p(1-p)}{n}$.
3. The variability across independently trained policies, e.g. different random seeds, datasets, checkpoints, or hyperparameter choices.

When one evaluates a **single, fixed policy** $\pi$, it typically aims at answering: *given a finite number of rollouts $n$ with $k$ successes, how uncertain am I about the underlying success probability $p$?*
At its core, this is a question about a binonial distribution (by definition!)---not exactly the best question to answer using a standard deviation!

## Why standard deviation is often the wrong number
Let start separating the two most immediate uses of standard deviation.
I want to argue that while the first one is mostly unhelpful, the second is merely expensive, and risks being misleading.

### (Wrong) Option 1: Collect everything, then quantify the uncertainty

Let's start with the standard deviation **across** rollouts. 
Suppose I run $n$ rollouts once and compute the standard deviation of the binary outcomes $X_1, \dots, X_n$.
This quantity estimates the spread of **individual outcomes**, not the uncertainty on $p$.
Under (1) the assumption that each rollout is i.i.d. $\operatorname{Bernoulli}(p)$, this spread is the square root of $\operatorname{Var}(X_i) = \hat p ( 1 - \hat p)$, with $\hat{p}$ estimated from rollouts.

Notice a first problem: this quantity is completely independent by the amount of empirical evidence collected.
In practice, both $\hat p = 9/10$ successes and $\hat p = 90/100$ result in the same uncertainty estimate!
The reported point estimate is the same in both cases, but the second evaluation is obviously much more informative than the first.

This alone should be enough to make one uncomfortable: the adoption of an uncertainty measure that is invariant with the amount of empirical evidence collected is rather difficult to justify.

### (Wrong) Option 2: Batch and repeat evaluations

Alternatively, one could re-run the entire evaluation protocol carried out to estimate $\hat p$ in the first place $t$-times, obtaining $\hat p_1, \dots, \hat{p}_t$, and then compute the standard deviation across these repeated estimates.
Essentially, this would result in estimating uncertainty by running an experiment of experiments.
Unfortunately, the resulting data requirements may prove prohibitive in robotics, as it is common for evaluations to take up to minutes, even in simulation.
Running evaluation protocols in the real world requires resets, strains hardware, or depends on a human supervising the setup, further hindering just repeating the same evaluation protocol many times.


<img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/meta_experiments.png" alt="" style="max-width:100%;height:auto;display:block;margin:2em 0;" />


Importantly, the Central Limit Theorem tells us that $\operatorname{Bernoulli}(n, p) \xrightarrow{n \to \infty} \mathcal N(np, np(1-p))$, with a rate of convergence that depends on the skewedness of $\operatorname{Bernoulli}(n, p)$: essentially, convergence to Gaussian is faster for $p \simeq 0.5$, and progressively slower for $p \to 0$ or $p \to 1$. Typically, one hopes that the success rate of a given policy is as close to 1 as possible, further hindering the possibility of fitting Gaussian distributions around the estimated $\hat p$.

## A binomial object for a binomial distribution

Once one accepts that the evaluation metric comes from Bernoulli trials, the right object to use to measure uncertainty becomes much more obvious: we should report an interval for the unknown binomial proportion $p$.

There are several ways to do this.
Some are too conservative for the small evaluation budgets common in robotics; some rely on normal approximations that behave badly near the edges. **Wilson score intervals** (see [here](https://www.econometrics.blog/post/the-wilson-confidence-interval-for-a-proportion/) for extra resources) are a very strong candidate as they are:

* bounded in $[0,1]$ (great to measure the average rate of a Bernoulli trial!)
* asymmetric (great for skewed $p$!)

In particular, given a confidence level associated with a Gaussian quantile $z$ and an observed success rate $\hat p = k/n$, the Wilson interval can be obtained analytically as:

$$
\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}
$$

As stated before, this interval is always bound between 0 and 1, and can also be asymmetric, which matches the nature of the underlying (asymmetric) Binomial distribution modeled---great properties for a measure of spread around success rates!
### Citation

If you find this useful for your work, please consider citing it.

<div id="capuanoWilsonScore2026">
<pre>
@misc{capuanoWilsonScore2026,
  title = {A primer on measuring the uncertainty of success rates},
  author = {Capuano, Francesco},
  year = 2026,
  month = {January},
  url = {https://fracapuano.github.io/blog/success-rates}
}
</pre>
</div>
