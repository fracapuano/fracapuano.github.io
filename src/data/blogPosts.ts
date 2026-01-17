import optimizingNns from './posts/optimizing-nns.md?raw';
import privilege from './posts/privilege-sounds-like-you-work-too-much.md?raw';
import oxfordThankYou from './posts/oxford-thank-you.md?raw';
import worldModels1 from './posts/world-models.md?raw';
// import successRates from './posts/success-rates.md?raw';
import nonno from './posts/nonno.md?raw';

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
    slug: "optimizing-nns",
    title: "A short note on optimizing NNs",
    date: "November 20, 2025",
    excerpt: "A technical blog to revisit the fundamentals of what, in the crudest sense, makes Deep Learning work. A SGD-to-Muon tour, derived from first principles in math and then implemented from scratch in Jax.",
    content: optimizingNns
  },
  {
    id: "2",
    slug: "privilege-sounds-like-you-work-too-much",
    title: "Privilege sounds like \"You work too much\"",
    date: "June 18, 2025",
    excerpt: "Privilege sounds like “You work too much” because the majority of ordinary people work to make ends meet, regardless of balance. Outside of the tech bubble, people live very different lives.",
    content: privilege
  },
  {
    id: "3",
    slug: "oxford-thank-you",
    title: "My thank-you note for Oxford",
    date: "October 31, 2025",
    excerpt: "I just started my PhD at Oxford, and I could not even imagine this would become real over the last few years, when for the majority of the time I truly acted without a plan.",
    content: oxfordThankYou
  },
  {
    id: "4",
    slug: "world-models-1",
    title: "Scattered thoughts on world models",
    date: "December 29, 2025",
    excerpt: "I am doubling down on (interactive, compositional) world models for my research, and have used the holidays to put together a short draft of how I have been thinking about world models lately.",
    content: worldModels1
  },
  {
    id: "5",
    slug: "my-grandpa",
    title: "In memory of my grandfather, Oscar",
    date: "January 12, 2026",
    excerpt: "My grandfather, Oscar, passed away on January 12, 2026. I remember him as a good, firm and great man. He was a key figure in my life, and taught me many of the lessons I now abide by.",
    content: nonno
  },
  // {
  //   id: "6",
  //   slug: "success-rates",
  //   title: "Do you actually understand your success rates?",
  //   date: "January 15, 2026",
  //   excerpt: "Stop reporting the standard deviation around the success rate! In the best case scenario, it is just an expensive way of measuring uncertainty; in the worst, it is a meaningless measure of spread.",
  //   content: successRates
  // }
].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
