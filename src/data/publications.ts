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
    title: "Robot Learning Tutorial",
    authors: "Francesco Capuano",
    venue: "Preprint",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2503.05217" },
      { label: "code", href: "https://github.com/fracapuano/robot-learning-tutorials" }
    ]
  },
  {
    id: "2",
    title: "Small VLAs Self-Learn from Mistakes When Promptly Corrected",
    authors: "Francesco Capuano, Moritz Reuss, Oier Mees, Ingmar Posner, Rudolf Lioutikov",
    venue: "NeurIPS'25 Open World Agents Workshop",
    year: "2025",
    links: []
  },
  {
    id: "3",
    title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robot Learning",
    authors: "Francesco Capuano, Andrea Lombardi, et al.",
    venue: "Preprint",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2506.01844" },
      { label: "code", href: "https://github.com/huggingface/lerobot" }
    ]
  },
  {
    id: "4",
    title: "Sim-is-more: Randomizing Simulation proves Sufficient for Generalizable End-to-End Autonomous Driving",
    authors: "Francesco Capuano, Aditya Sharma, Georg Martius, Oier Mees",
    venue: "AutoML Conference 2025",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2503.14258" },
      { label: "code", href: "https://github.com/fracapuano/sim-is-more" }
    ]
  },
  {
    id: "5",
    title: "Shaping Laser Pulses with Reinforcement Learning",
    authors: "Francesco Capuano, Bedřich Rus, Roberto Samoila",
    venue: "Reinforcement Learning Conference (RLC) 2025",
    year: "2025",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2503.05217" },
      { label: "code", href: "https://github.com/fracapuano/LasER" }
    ]
  },
  {
    id: "6",
    title: "High-Power Laser Pulse Shape Optimization via Reinforcement Learning",
    authors: "Francesco Capuano, Bedřich Rus, Roberto Samoila",
    venue: "Frontiers in Optics + Laser Science (FiO) 2024",
    year: "2024",
    links: []
  },
  {
    id: "7",
    title: "TempoRL: laser pulse temporal shape optimization with Deep Reinforcement Learning",
    authors: "Francesco Capuano, Giulio Folpini, Bedřich Rus",
    venue: "SPIE Optics + Optoelectronics 2023",
    year: "2023",
    links: [
      { label: "arXiv", href: "https://arxiv.org/abs/2211.09tried" },
      { label: "code", href: "https://github.com/fracapuano/TempoRL" }
    ],
    award: "Best Student Paper Award"
  },
  {
    id: "8",
    title: "Optimization of the laser pulse duration at ELI-Beamlines",
    authors: "Francesco Capuano, Bedřich Rus, Giulio Folpini",
    venue: "PCaPAC 2022",
    year: "2022",
    links: [
      { label: "PDF", href: "https://accelconf.web.cern.ch/pcapac2022/papers/wep10.pdf" },
      { label: "code", href: "https://github.com/fracapuano/ELIopt" }
    ]
  },
];
