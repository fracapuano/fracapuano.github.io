export interface Publication {
  id: string;
  title: string;
  authors: string;
  venue: string;
  year: string;
  links: { label: string; href: string }[];
}

export const publicationsData: Publication[] = [
  {
    id: "1",
    title: "Geometric Diffusion Models for Molecular Generation",
    authors: "Your Name, Collaborator One, Advisor Name",
    venue: "Neural Information Processing Systems (NeurIPS)",
    year: "2024",
    links: [
      { label: "paper", href: "#" },
      { label: "code", href: "#" },
      { label: "arXiv", href: "#" }
    ]
  },
  {
    id: "2",
    title: "Equivariant Neural Networks for Drug Discovery",
    authors: "Your Name, Collaborator Two, Collaborator Three",
    venue: "International Conference on Machine Learning (ICML)",
    year: "2024",
    links: [
      { label: "paper", href: "#" },
      { label: "arXiv", href: "#" }
    ]
  },
  {
    id: "3",
    title: "On the Expressivity of Graph Neural Networks",
    authors: "Collaborator One, Your Name, Advisor Name",
    venue: "International Conference on Learning Representations (ICLR)",
    year: "2023",
    links: [
      { label: "paper", href: "#" },
      { label: "code", href: "#" }
    ]
  },
];
