> TLDR: I am doubling down on (interactive, compositional) world models for my research, and have used the holidays to put together a short draft of how I have been thinking about world models lately. This is all very much up for discussion, [reach out](https://x.com/_fracapuano) if you would like to discuss this or collaborate!

## Abstract

World models can prove tranformative in achieving autonomous agency across complex, unlabeled environments. Advancing their fidelity and practical utility in decision making problems is paramount for their effective use in developing intelligent systems.    
While current WMs primarily serve as simulators, they can be limited by (1) spectral biases and (2) overreliance on statistical correlation over causal entanglement. My research seeks to bridge the gap between high-fidelity non-interactive generative models and agent-centric simulations by integrating neural representations that preserve fine-grained temporal and perceptual dynamics.
When available, such improved world models could be highly useful in decision making problems by (1) enabling the alignment disparate action spaces by studying dynamics similarity and (2) deriving rewards from functionals dynamics-aware representations of the environment and the goals achievable in it, possibly facilitating the emergence of (long-horizon) goal-directed behaviors with minimal external supervision.

## Research interests
My interests lie in developing and using interactive World Models (WM) suited for autonomous agency. 
My long-term goal is improving autonomous decision making by leveraging dynamics-aware information that is independently learned via interactions with an environment with as little explicit supervision as possible.

### World Models for simulation

Besides the long-term objective of enabling *in-imagination* decision-making, *interactive* world models (WMs) have proven transformative in domains where real-world data collection is prohibitively expensive. For instance, research in autonomous driving and robotics is increasingly shifting toward interactive WMs—such as [Hu et al. (2023)](#huGAIA1GenerativeWorld2023), [Russell et al. (2025)](#russellGAIA2ControllableMultiView2025), and [Li et al. (2025)](#liRoboticWorldModel2025)—as high-fidelity alternatives to otherwise task-specific, manually engineered simulators. 
Beyond physical embodiments, interactive WMs for digital embodied agents—e.g., those used in the gaming industry—offer a compelling path to bypass the labor-intensive process of procedural generation and manual game-logic definition. Both digital and physical embodied agents present a unique set of challenges in terms of (1) generative consistency, (2) long-horizon temporal stability, and (3) stringent hardware and inference-latency requirements for near real-time or real-time interactions at test-time. 

A critical frontier in this domain lies in narrowing the fidelity gap between non-interactive generative video models [[OpenAI (2024)](#openaiVideoGenerationModels2024); [Wiedemer et al. (2025)](#wiedemerVideoModelsAre2025)] and their interactive counterparts [[Bruce et al. (2024)](#bruceGenieGenerativeInteractive2024)]. Research works including [Alonso et al. (2024)](#alonsoDiffusionWorldModeling2024) show that perceptual fidelity is indeed a rather relevant factor for agents trained in latent spaces *à la* Dreamer [[Hafner et al. (2020)](#hafnerDreamControlLearning2020)]. Consequently, integrating advancements from non-interactive neural representations such as [Mildenhall et al. (2020)](#mildenhallNeRFRepresentingScenes2020), [Kerbl et al. (2023)](#kerbl3DGaussianSplatting2023), and [Mescheder et al. (2025)](#meschederSharpMonocularView2025) presents a promising avenue for enhancing the realism and grounding of agent-centric, interactive simulations.

### World models for decision making

WMs hold the premise of endowing autonomous systems with reactive forms of intelligence [[LeCun (2025)](#yannlecunYannLeCunSelfSupervised2025)], surpassing autoregressive generation in favor of inherently declarative, dynamics-aware intelligent behaviors grounded in reasoning and planning within said WM. While research in this domain was established by [Ha & Schmidhuber (2018)](#haWorldModels2018), [Hafner et al. (2020)](#hafnerDreamControlLearning2020), [Hafner et al. (2022)](#hafnerMasteringAtariDiscrete2022), and [Hafner et al. (2024)](#hafnerMasteringDiverseDomains2024), it has been significantly advanced by recent architectural [[Assran et al. (2025)](#assranVJEPA2SelfSupervised2025)] and conceptual [[Sutton (2025)](#richsuttonRichSuttonOaK2025)] developments. 

*Beyond simulation*, WMs could be used in decision-making leveraging their inherently dynamics-aware representations to estimate normative signals, moving beyond descriptive transition modeling toward the implicit derivation of complex reward functions, using WMs as the natural backbone for learning reward models in highly dynamical settings and long-horizon problems. In turn, these learned reward models could facilitate the emergence of intelligent behaviors in complex, unlabeled environments where explicit external supervision is unavailable or prohibitively expensive to define.


## Project Proposals
Progress toward my long-term goal of building more capable autonomous agents can be made pursuing (1) better WMs to be used (2) beyond simulation.

### (Better) World models...
* **Spectral bias mitigation in world models**: Deep neural networks exhibit a notorious spectral bias, favoring low-frequency structural components over high-frequency details [[Tancik et al. (2020)](#tancikFourierFeaturesLet2020a)]. In the context of WMs, this bias may result in the oversmoothing of rapid environmental transitions and transient dynamics. Making WMs robust to the spectral properties of environment dynamics is a prerequisite for preserving fine-grained temporal fidelity, ensuring that critical information remains stable during long-horizon imagination, thereby improving perceptual and dynamical generation fidelity.
* **Causal invariance in transition dynamics**: Current WMs often fail to decouple invariant causal laws from transient, task-specific parameters, necessitating exhaustive retraining when environmental conditions shift. Identifying and isolating fundamental causal structures [[Lei et al. (2024a)](#leiCompeteComposeLearning2024); [Lei et al. (2024b)](#leiSPARTANSparseTransformer2024)] allows for the recovery of a modular transition logic. Grounding WM in such composable representation rather than purely statistical correlation might enable the repurposing of learned knowledge across diverse domains, bypassing the computational cost of model adaptation whenever distribution shifts occur.

### ...for decision making
* **Generalized action spaces**: Developing generalized agents requires reconciling disparate control interfaces. While latent-space projections enable cross-modality inputs, a fundamental challenge lies in aligning action spaces based on the resulting observed dynamics rather than on raw control signals. Bridging heterogeneous kinematics—such as different games environments, or even differing robot morphologies—through a *generalized action space* would enable the transfer of behaviors between otherwise disconnected environments. In turn, this could be useful in improving few-shot transfer of specialized policies *across* different worlds.
* **Dynamics-based reward models**: Manual reward specification remains a major bottleneck to scaling autonomous agency, particularly in high-dimensional, unlabeled environments and in tasks involving complex, long-horizon objectives. The self-supervised representations learned by WMs encode rich latent structures that can be exploited to infer normative signals directly from environmental dynamics. By treating rewards as functionals of state-space transitions rather than as externally engineered scalars, goal-directed behaviors and even adaptable foundational behavior models [[Touati & Ollivier (2021)](#touatiLearningOneRepresentation2021); [Sikchi et al. (2025)](#sikchiFastAdaptationBehavioral2025)] can emerge as a consequence of optimal agency, grounded in a deeper understanding of the environment's underlying dynamics.

### Citation

Please consider citing this blogpost if you find it useful for your research. [Reach out](https://x.com/_fracapuano) if you want to discuss these topics further!

<div id="capuanoScatteredThoughtsWorld2025">
<pre>
@misc{capuanoScatteredThoughtsWorld2025,
  title = {Scattered thoughts on world models},
  author = {Capuano, Francesco},
  year = 2025,
  month = {December},
  url = {https://fracapuano.github.io/blog/world-models-1}
}
</pre>
</div>


## References

<div id="alonsoDiffusionWorldModeling2024">
<pre>
@misc{alonsoDiffusionWorldModeling2024,
  title = {Diffusion for {{World Modeling}}: {{Visual Details Matter}} in {{Atari}}},
  author = {Alonso, Eloi and Jelley, Adam and Micheli, Vincent and Kanervisto, Anssi and Storkey, Amos and Pearce, Tim and Fleuret, Fran{\c c}ois},
  year = 2024,
  doi = {10.48550/arXiv.2405.12399}
}
</pre>
</div>

<div id="assranVJEPA2SelfSupervised2025">
<pre>
@misc{assranVJEPA2SelfSupervised2025,
  title = {V-{{JEPA}} 2: {{Self-Supervised Video Models Enable Understanding}}, {{Prediction}} and {{Planning}}},
  author = {Assran, Mido and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and others},
  year = 2025,
  doi = {10.48550/arXiv.2506.09985}
}
</pre>
</div>

<div id="bruceGenieGenerativeInteractive2024">
<pre>
@misc{bruceGenieGenerativeInteractive2024,
  title = {Genie: {{Generative Interactive Environments}}},
  author = {Bruce, Jake and Dennis, Michael and Edwards, Ashley and {Parker-Holder}, Jack and others},
  year = 2024,
  doi = {10.48550/arXiv.2402.15391}
}
</pre>
</div>

<div id="haWorldModels2018">
<pre>
@article{haWorldModels2018,
  title = {World {{Models}}},
  author = {Ha, David and Schmidhuber, J{\"u}rgen},
  year = 2018,
  doi = {10.5281/zenodo.1207631}
}
</pre>
</div>

<div id="hafnerDreamControlLearning2020">
<pre>
@misc{hafnerDreamControlLearning2020,
  title = {Dream to {{Control}}: {{Learning Behaviors}} by {{Latent Imagination}}},
  author = {Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  year = 2020,
  doi = {10.48550/arXiv.1912.01603}
}
</pre>
</div>

<div id="hafnerMasteringAtariDiscrete2022">
<pre>
@misc{hafnerMasteringAtariDiscrete2022,
  title = {Mastering {{Atari}} with {{Discrete World Models}}},
  author = {Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  year = 2022,
  doi = {10.48550/arXiv.2010.02193}
}
</pre>
</div>

<div id="hafnerMasteringDiverseDomains2024">
<pre>
@misc{hafnerMasteringDiverseDomains2024,
  title = {Mastering {{Diverse Domains}} through {{World Models}}},
  author = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  year = 2024,
  doi = {10.48550/arXiv.2301.04104}
}
</pre>
</div>

<div id="huGAIA1GenerativeWorld2023">
<pre>
@misc{huGAIA1GenerativeWorld2023,
  title = {{{GAIA-1}}: {{A Generative World Model}} for {{Autonomous Driving}}},
  author = {Hu, Anthony and Russell, Lloyd and Yeo, Hudson and Murez, Zak and others},
  year = 2023,
  doi = {10.48550/arXiv.2309.17080}
}
</pre>
</div>

<div id="kerbl3DGaussianSplatting2023">
<pre>
@misc{kerbl3DGaussianSplatting2023,
  title = {{{3D Gaussian Splatting}} for {{Real-Time Radiance Field Rendering}}},
  author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  year = 2023,
  doi = {10.48550/arXiv.2308.04079}
}
</pre>
</div>

<div id="leiCompeteComposeLearning2024">
<pre>
@misc{leiCompeteComposeLearning2024,
  title = {Compete and {{Compose}}: {{Learning Independent Mechanisms}} for {{Modular World Models}}},
  author = {Lei, Anson and Nolte, Frederik and Sch{\"o}lkopf, Bernhard and Posner, Ingmar},
  year = 2024,
  doi = {10.48550/arXiv.2404.15109}
}
</pre>
</div>

<div id="leiSPARTANSparseTransformer2024">
<pre>
@misc{leiSPARTANSparseTransformer2024,
  title = {{{SPARTAN}}: {{A Sparse Transformer Learning Local Causation}}},
  author = {Lei, Anson and Sch{\"o}lkopf, Bernhard and Posner, Ingmar},
  year = 2024,
  doi = {10.48550/arXiv.2411.06890}
}
</pre>
</div>

<div id="liRoboticWorldModel2025">
<pre>
@misc{liRoboticWorldModel2025,
  title = {Robotic {{World Model}}: {{A Neural Network Simulator}} for {{Robust Policy Optimization}} in {{Robotics}}},
  author = {Li, Chenhao and Krause, Andreas and Hutter, Marco},
  year = 2025,
  doi = {10.48550/arXiv.2501.10100}
}
</pre>
</div>

<div id="meschederSharpMonocularView2025">
<pre>
@misc{meschederSharpMonocularView2025,
  title = {Sharp {{Monocular View Synthesis}} in {{Less Than}} a {{Second}}},
  author = {Mescheder, Lars and Dong, Wei and Li, Shiwei and others},
  year = 2025,
  doi = {10.48550/arXiv.2512.10685}
}
</pre>
</div>

<div id="mildenhallNeRFRepresentingScenes2020">
<pre>
@misc{mildenhallNeRFRepresentingScenes2020,
  title = {{{NeRF}}: {{Representing Scenes}} as {{Neural Radiance Fields}} for {{View Synthesis}}},
  author = {Mildenhall, Ben and Srinivasan, Pratul P. and Tancik, Matthew and others},
  year = 2020,
  doi = {10.48550/arXiv.2003.08934}
}
</pre>
</div>

<div id="openaiVideoGenerationModels2024">
<pre>
@misc{openaiVideoGenerationModels2024,
  title = {Video Generation Models as World Simulators},
  author = {OpenAI},
  year = 2024,
  url = {https://openai.com/index/video-generation-models-as-world-simulators/}
}
</pre>
</div>

<div id="richsuttonRichSuttonOaK2025">
<pre>
@misc{richsuttonRichSuttonOaK2025,
  title = {Rich {{Sutton}}: {{The OaK Architecture}}: {{A Vision}} of {{SuperIntelligence}} from {{Experience}}},
  author = {{Rich Sutton}},
  year = 2025
}
</pre>
</div>

<div id="russellGAIA2ControllableMultiView2025">
<pre>
@misc{russellGAIA2ControllableMultiView2025,
  title = {{{GAIA-2}}: {{A Controllable Multi-View Generative World Model}} for {{Autonomous Driving}}},
  author = {Russell, Lloyd and Hu, Anthony and Bertoni, Lorenzo and others},
  year = 2025,
  doi = {10.48550/arXiv.2503.20523}
}
</pre>
</div>

<div id="sikchiFastAdaptationBehavioral2025">
<pre>
@misc{sikchiFastAdaptationBehavioral2025,
  title = {Fast {{Adaptation}} with {{Behavioral Foundation Models}}},
  author = {Sikchi, Harshit and Tirinzoni, Andrea and Touati, Ahmed and others},
  year = 2025,
  doi = {10.48550/arXiv.2504.07896}
}
</pre>
</div>

<div id="tancikFourierFeaturesLet2020a">
<pre>
@misc{tancikFourierFeaturesLet2020a,
  title = {Fourier {{Features Let Networks Learn High Frequency Functions}} in {{Low Dimensional Domains}}},
  author = {Tancik, Matthew and Srinivasan, Pratul P. and Mildenhall, Ben and others},
  year = 2020,
  doi = {10.48550/arXiv.2006.10739}
}
</pre>
</div>

<div id="touatiLearningOneRepresentation2021">
<pre>
@inproceedings{touatiLearningOneRepresentation2021,
  title = {Learning {{One Representation}} to {{Optimize All Rewards}}},
  author = {Touati, Ahmed and Ollivier, Yann},
  year = 2021,
  booktitle = {Advances in Neural Information Processing Systems}
}
</pre>
</div>

<div id="wiedemerVideoModelsAre2025">
<pre>
@misc{wiedemerVideoModelsAre2025,
  title = {Video Models Are Zero-Shot Learners and Reasoners},
  author = {Wiedemer, Thadd{\"a}us and Li, Yuxuan and Vicol, Paul and others},
  year = 2025,
  doi = {10.48550/arXiv.2509.20328}
}
</pre>
</div>

<div id="yannlecunYannLeCunSelfSupervised2025">
<pre>
@misc{yannlecunYannLeCunSelfSupervised2025,
  title = {Yann {{LeCun}} \textbar{} {{Self-Supervised Learning}}, {{JEPA}}, {{World Models}}, and the Future of {{AI}}},
  author = {{Yann LeCun}},
  year = 2025
}
</pre>
</div>