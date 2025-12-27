export interface BlogPost {
  id: string;
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  content: string;
}

export const blogPostsData: BlogPost[] = [
  {
    id: "1",
    slug: "understanding-diffusion-models",
    title: "Understanding Diffusion Models: A First Principles Approach",
    date: "December 15, 2024",
    excerpt: "An intuitive explanation of diffusion models from the perspective of score matching and Langevin dynamics.",
    content: `
Diffusion models have revolutionized generative modeling. In this post, I'll explain them from first principles.

## The Forward Process

Given a data distribution $p_0(x)$, we define a forward noising process:

$$x_t = \\sqrt{\\alpha_t} x_0 + \\sqrt{1 - \\alpha_t} \\epsilon$$

where $\\epsilon \\sim \\mathcal{N}(0, I)$ and $\\alpha_t$ is a noise schedule.

## The Reverse Process

The key insight is that we can reverse this process by learning the score function:

$$\\nabla_x \\log p_t(x)$$

A simple implementation in PyTorch:

\`\`\`python
import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))
\`\`\`

## Connection to Score Matching

The training objective can be written as:

$$\\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\| \\epsilon_\\theta(x_t, t) - \\epsilon \\|^2 \\right]$$

This is equivalent to denoising score matching, where we learn $s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)$.
    `
  },
  {
    id: "2",
    slug: "geometric-deep-learning",
    title: "Notes on Geometric Deep Learning",
    date: "November 3, 2024",
    excerpt: "Key concepts from the geometric deep learning blueprint and their applications to molecular modeling.",
    content: `
These are my notes on geometric deep learning, following the framework of Bronstein et al.

## Symmetry and Invariance

The key principle is that neural networks should respect the symmetries of the problem. For a group $G$ acting on input space $\\mathcal{X}$:

$$f(g \\cdot x) = \\rho(g) \\cdot f(x)$$

where $\\rho$ is a representation of $G$.

## Equivariant Layers

An equivariant linear layer has the form:

$$W(g \\cdot x) = \\rho_{out}(g) W x$$

For SE(3) equivariance in 3D molecular modeling:

\`\`\`python
# Spherical harmonics basis
def spherical_harmonics(l, pos):
    # Y_l^m(theta, phi) basis functions
    return e3nn.o3.spherical_harmonics(l, pos, normalize=True)
\`\`\`

The representation theory of SO(3) gives us irreducible representations indexed by $l = 0, 1, 2, \\ldots$
    `
  },
];
