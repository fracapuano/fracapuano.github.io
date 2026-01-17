> TLDR: Stop reporting the standard deviation around the success rate! In the best case scenario, it is just an expensive way of measuring uncertainty; in the worst, it is a meaningless measure of spread. Use Wilson score intervals instead for a cheap, principled measure of uncertainty.

## Introduction

In the robot learning community, the **Success Rate**  on arbitrary tasks is the key metric for both researchers and practitioners.
Regardless of the sophistication of one particular technique, if it does not attain the desired goal---slotting batteries into their compartment, tying a not, or moving a cube around---there is little justification to investigating it further.
The Success Rate $p$ for a given policy on a given task is thus the primary metric that is used to climb leaderboards and evaluation benchmarks, compare multiple versions of the same algorithm or entirely different algorithms, and even claim state-of-the-art performance.
In practice, the success rate is measured by running an evaluation protocol for a finite (typically small) number of times, $n \simeq 10$, count the successful rollouts and then report the fraction of them that is successful, $p \approx \tfrac{n^+}{n}$, with $n^+$ being the number of succesful ($+$) rollouts.

While the success rate is important in itself, measures of its spread are also important. Ideally, we'd want policies that succeed with high probability an overwhelming majority of the time. For some higher risk applications, for instance, one might even be willing to trade-off some percentage of success for more robustness. That is, a reliable 85% success rate might be preferrable over a less reliable 90%.

Mathematically, this observation can be quantified looking at measured of spread of the average success rate reported.
If we naively assume that the success rate over $n$ trials is distributed normally, then we could run $k$ evaluations---i.e., $k \cdot n$ rollouts---and report the average success rate $\mu_p$ alongside the measured standard deviation $\sigma_p$. 
All in all, modeling the success rate distribution with a Gaussian offers benefits including the fact that we expect the entropy of the success rate distribution to be low---so that the distribution is concentrated around a given value---and the usual mathematical amenability of the normal distribution.
However, assuming $p \sim \mathcal N(\mu_p, \sigma_p^2)$ poses computational and interpretability challenges.
For starters, one needs a large number of rollouts to build such normal approximation.
Doing robot learning research in the real world means a $k$-fold increase in the time to run a single evaluation protocol, with $k$-times the resets, $k$-times the strain on the physical robot and experimental setups, etc.
Further, from the point of view of interpretability, a Normal distribution is inherently suboptimal as $\text{supp}(\mathcal N) \equiv \mathbb R$, meaning the result might as well look like $\mu_p - \sigma_p < 0$ or $\mu_p + \sigma_p > 1$. 
Put it plainly, how does one interpret having a -10% or 104% success rate?

In this blogpost, I want to argue in favour of a way of quantifying the uncertainty around the success rate that is (1) computationally efficient and (2) theoretically solid: Wilson Score intervals.
I have gotten interested in this subject after reading [this tweet](https://x.com/kvablack/status/2001109700151316519?s=20) from Kevin Black (first author of $\pi_0$), and found myself realizing that pretending to be scientifically rigorous is almost as despicable as not being rigorous at all. Excuse my wittyness here, although this is still a blogpost about statistical rigor.


## 1. The Nature of the Metric
To understand the error, we have to look at the distribution. In standard regression or reinforcement learning tasks (like MuJoCo locomotion), our return is a continuous variable. If you run enough seeds, the Central Limit Theorem (CLT) kicks in, and the distribution of means approaches a Gaussian.

Success in robotics, however, is binary.
* A grasp is either successful or it isn't.
* The drone reaches the target or it crashes.

Each evaluation rollout is a **Bernoulli trial** ($X \in \{0, 1\}$) with an unknown parameter $p$, the true probability of success. When we run $N$ evaluation episodes, the total count of successes $k$ follows a **Binomial distribution**:

$$k \sim \text{Binomial}(N, p)$$

Our goal during evaluation is to estimate this underlying $p$ given our finite set of observations.

## 2. The Evaluation Bottleneck
If we had infinite compute, we could estimate the uncertainty of $p$ using a "brute force" Monte Carlo approach:
1.  Run the full evaluation protocol (e.g., 50 episodes) to get a success rate $\hat{p}_1$.
2.  Repeat this entire process $K$ times (where $K$ is large).
3.  Compute the standard deviation of these $K$ success rates.

In this hypothetical scenario, the distribution of $\hat{p}$ would indeed look Gaussian, and standard deviation would be a valid metric.

**But in reality, evaluation is expensive.**
In real-world robotics, we often have a budget of $N=10$ or $N=20$ trials *total*. We cannot afford to repeat the "experiment of experiments." We have one single sample of $N$ trials, and we need to derive our uncertainty from that alone.

## 3. The Symmetry Trap
When we report Mean $\pm$ Standard Deviation on a single set of binary trials, we are implicitly forcing a Gaussian distribution onto a Binomial problem. This fails for two critical reasons:

### The Boundary Problem
Success rates are strictly bounded between $[0, 1]$. A Gaussian distribution extends from $-\infty$ to $+\infty$. When your success rate is near the boundaries (e.g., 5% or 95%), a symmetric standard deviation will almost always cross into impossible territory (like -2% or 102%).

### The Skewness Reality
The Binomial distribution is only symmetric when $p=0.5$.
* If $p=0.9$, the distribution is heavily **left-skewed** (tail points to 0).
* If $p=0.1$, it is **right-skewed**.

By reporting a single number ($\sigma$) for uncertainty, you are painting a symmetric picture of a highly asymmetric reality. You are underestimating the risk of failure (the long tail) and overestimating the potential for "super-success" (the hard ceiling).

## 4. The Solution: Wilson Score Intervals
If we assume the underlying mechanism is Binomial, we shouldn't use an interval derived for a Gaussian mean. We should use a **Binomial Confidence Interval**.

While there are several options (like the ultra-conservative Clopper-Pearson), the **Wilson Score Interval** is widely regarded as the best balance of coverage and precision for $N$ ranges typical in robotics.

### The Intuition
The Wilson interval works by **inverting the hypothesis test**. Instead of asking "Given $\hat{p}$, where might the true $p$ be?", it effectively asks:
> *"For which values of $p$ would our observed data be considered 'not surprising'?"*

The math accounts for the fact that the **variance of a Binomial distribution changes with its mean**.
$$\sigma^2 = p(1-p)$$
As $p$ approaches 1, the variance naturally shrinks to 0. The Wilson formulation builds this changing variance directly into the interval.

The result is an interval that is **asymmetric**:
* If $\hat{p} = 0.95$, the interval might be $[0.85, 0.99]$.
* It "pushes" against the ceiling of 1.0 but stretches down into the tail of 0.0.

### Pure Python Implementation
You don't need complex libraries to implement this. It is a closed-form solution that you can add to your evaluation scripts today.


```python
import math

def wilson_score_interval(successes, total_trials, confidence=0.95):
    """
    Computes the Wilson Score Interval for a binomial proportion.
    
    Args:
        successes: Number of successful rollouts
        total_trials: Total number of rollouts
        confidence: Desired confidence level (default 0.95)
    
    Returns:
        (lower_bound, upper_bound)
    """
    if total_trials == 0: return 0.0, 0.0
    
    # 1. Get the z-score (1.96 for 95% confidence)
    # Using hardcoded value for independence, or use scipy.stats.norm.ppf
    z = 1.96
    
    # 2. Compute the components
    n = total_trials
    p_hat = successes / n
    
    denominator = 1 + (z**2) / n
    center_adjusted = p_hat + (z**2) / (2 * n)
    spread = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))
    
    # 3. Calculate bounds
    lower = (center_adjusted - spread) / denominator
    upper = (center_adjusted + spread) / denominator
    
    # Clip to valid [0, 1] range (handles float precision issues)
    return max(0.0, lower), min(1.0, upper)

# Example Usage
# You ran 10 trials, got 9 successes.
low, high = wilson_score_interval(9, 10)
print(f"Success Rate: 0.90")
print(f"95% CI: [{low:.2f}, {high:.2f}]")
# Output: [0.60, 0.98]
```

<div id="capuanoWilsonScore2026">
<pre>
    @misc{capuanoWilsonScore2026,
    title = {Sounder Robot Learning Evaluation with Wilson Score Intervals},
    author = {Capuano, Francesco},
    year = 2026,
    month = {January},
    url = {https://fracapuano.github.io/blog/}
    }
</pre>
</div>