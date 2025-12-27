export interface NewsEntry {
  id: string;
  date: string;
  content: string;
}

export const newsData: NewsEntry[] = [
  {
    id: "1",
    date: "Dec 2024",
    content: "Our paper on diffusion models for molecular generation was accepted to NeurIPS 2024!"
  },
  {
    id: "2",
    date: "Oct 2024",
    content: "Started my DPhil in Machine Learning at the University of Oxford."
  },
  {
    id: "3",
    date: "Sep 2024",
    content: "Presented our work on geometric deep learning at the ICML workshop."
  },
  {
    id: "4",
    date: "Jul 2024",
    content: "New preprint available: 'Equivariant Neural Networks for Drug Discovery'."
  },
  {
    id: "5",
    date: "May 2024",
    content: "Completed MSc in Computer Science with distinction at ETH Zürich."
  },
  {
    id: "6",
    date: "Mar 2024",
    content: "Research internship at DeepMind London on protein structure prediction."
  },
  {
    id: "7",
    date: "Jan 2024",
    content: "Teaching assistant for the Advanced Machine Learning course at ETH."
  },
  {
    id: "8",
    date: "Nov 2023",
    content: "Won Best Paper Award at the ML4Science workshop."
  },
];
