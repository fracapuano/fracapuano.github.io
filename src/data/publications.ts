export interface Publication {
  id: string;
  title: string;
  authors: string;
  venue: string;
  year: string;
  links: { label: string; href: string }[];
  award?: string;
}

export const publicationsData: Publication[] = [
  {
    id: "1",
    title: "Robot Learning: A Tutorial",
    authors: "Francesco Capuano*, Caroline Pascal, Adil Zouitine, Thomas Wolf, Michel Aractingi (*Corresponding author)",
    venue: "Preprint",
    year: "2025",
    links: [
      { label: "PDF", href: "https://arxiv.org/pdf/2510.12403" },
      { label: "code", href: "https://github.com/fracapuano/robot-learning-tutorial" },
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:hSXxhBgJEzUJ:scholar.google.com/&output=citation&scisdr=ChVr8jrdEO7moMbcsWU:ABGrvjIAAAAAaQTaqWUHTNANg9eUbBTFxJflqSM&scisig=ABGrvjIAAAAAaQTaqdZnzqjdYhP5CPnKGPV168U&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1" }
    ]
  },
  {
    id: "2",
    title: "Opinion: Small VLAs Self-Learn Consistency",
    authors: "Francesco Capuano*, Adil Zouitine, Michel Aractingi (*Corresponding author)",
    venue: "NeurIPS 2025 (Embodied World Models Workshop)",
    year: "2025",
    links: [
      { label: "PDF", href: "https://openreview.net/pdf?id=CtVPGdXFzD" }
    ]
  },
  {
    id: "3",
    title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics",
    authors: "Mustafa Shukor*, Dana Aubakirova*, Francesco Capuano*, ..., Matthieu Cord, Thomas Wolf, Remi Cadene* (*Core team)",
    venue: "Preprint",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2506.01844" },
      { label: "code", href: "https://github.com/huggingface/lerobot" }
    ]
  },
  {
    id: "4",
    title: "Sim-is-more: Randomizing HW-NAS with Synthetic Devices",
    authors: "Francesco Capuano*, Gabriele Tiboni, Niccolò Cavagnero, Giuseppe Averta (*Corresponding author)",
    venue: "Preprint (Rejected AutoML 2025)",
    year: "2025",
    links: [
      { label: "PDF", href: "https://arxiv.org/pdf/2504.00663" },
      { label: "code", href: "https://github.com/fracapuano/sim-is-more" },
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:pAS7xDsmQskJ:scholar.google.com/&output=citation&scisdr=CgIQ5_D2ENjryZGfDSw:AAZF9b8AAAAAaCyZFSwRo7_u4QAm2JjcyR-0u1w&scisig=AAZF9b8AAAAAaCyZFezZ-86eUX9R4rGpGoTyz4A&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1" }
    ]
  },
  {
    id: "5",
    title: "Shaping Laser Pulses with Reinforcement Learning",
    authors: "Francesco Capuano*, Davorin Peceli, Gabriele Tiboni (*Corresponding author)",
    venue: "Reinforcement Learning Conference (RLC)",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2503.00499" },
      { label: "code", href: "https://github.com/fracapuano/lase-rl" },
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:QokepdSWCM0J:scholar.google.com/&output=citation&scisdr=CgIQ5_D2ENjryZGehF0:AAZF9b8AAAAAaCyYnF0fWVPoGLS5UB1bAHyPBA4&scisig=AAZF9b8AAAAAaCyYnKrMgmwkxwFb0TYcgE8k3ss&scisf=4&ct=citation&cd=-1&hl=en" }
    ]
  },
  {
    id: "6",
    title: "High-Power Laser Pulse Shape Optimization with Hybrid Stochastic Optimization Algorithms",
    authors: "Ishraq Md Anjum*, Davorin Peceli, Francesco Capuano, Bedřic Rus (*Corresponding author)",
    venue: "Frontier in Optics (FiO)",
    year: "2024",
    links: [
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:XI-RykQ2kQcJ:scholar.google.com/&output=citation&scisdr=CgIQ5_D2ENjryZGSu3s:AAZF9b8AAAAAaCyUo3uxkK5JGIfF1w_OIXAMA6w&scisig=AAZF9b8AAAAAaCyUoy1lQd1_JPnE7Eu0D8sGDDU&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1" }
    ]
  },
  {
    id: "7",
    title: "TempoRL: laser pulse temporal shape optimization with deep reinforcement learning",
    authors: "Francesco Capuano*, Davorin Peceli, Gabriele Tiboni, Bedřic Rus (*Corresponding author)",
    venue: "SPIE Optics and Optoelectronics",
    year: "2023",
    links: [
      { label: "PDF", href: "https://arxiv.org/pdf/2304.12187" },
      { label: "code", href: "https://github.com/fracapuano/TempoRL" },
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:QjD8Z9wkfjQJ:scholar.google.com/&output=citation&scisdr=CgIQ5_D2ENjryZGQVco:AAZF9b8AAAAAaCyWTcpin4SJ7PCEV38fIwyCiWM&scisig=AAZF9b8AAAAAaCyWTZ2S_3oPYmx9B7Is-NaSORE&scisf=4&ct=citation&cd=-1&hl=fi&scfhb=1" }
    ],
    award: "Best Student Paper Award"
  },
  {
    id: "8",
    title: "Laser pulse duration optimization with numerical methods",
    authors: "Francesco Capuano*, Davorin Peceli, Gabriele Tiboni, Alexandr Špaček, Bedřic Rus (*Corresponding author)",
    venue: "PCaPAC",
    year: "2022",
    links: [
      { label: "PDF", href: "https://accelconf.web.cern.ch/pcapac2022/papers/pcapac2022-proceedings.pdf#page=43" },
      { label: "code", href: "https://github.com/fracapuano/TempoRL" },
      { label: "BibTeX", href: "https://scholar.googleusercontent.com/scholar.bib?q=info:DlnlMBN0Wd8J:scholar.google.com/&output=citation&scisdr=CgIQ5_D2ENjryZGKmbI:AAZF9b8AAAAAaCyMgbKLMijX-y00hejaBOwoSLI&scisig=AAZF9b8AAAAAaCyMgWjozJEVM3c0U5Xq4EOI1Jk&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1" }
    ]
  }
];
