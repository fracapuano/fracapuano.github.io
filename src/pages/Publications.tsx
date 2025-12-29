import Layout from "@/components/Layout";
import Section from "@/components/Section";
import PublicationItem from "@/components/PublicationItem";
import { publicationsData } from "@/data/publications";

const Publications = () => {
  // Group publications by year
  const publicationsByYear = publicationsData.reduce((acc, pub) => {
    if (!acc[pub.year]) {
      acc[pub.year] = [];
    }
    acc[pub.year].push(pub);
    return acc;
  }, {} as Record<string, typeof publicationsData>);

  const years = Object.keys(publicationsByYear).sort((a, b) => parseInt(b) - parseInt(a));

  return (
    <Layout>
      <h1 className="text-2xl md:text-3xl font-bold mb-8">Publications</h1>
      
      {years.map((year) => (
        <Section key={year} title={year}>
          {publicationsByYear[year].map((pub) => (
            <PublicationItem
              key={pub.id}
              title={pub.title}
              authors={pub.authors}
              venue={pub.venue}
              year={pub.year}
              links={pub.links}
              award={pub.award}
            />
          ))}
        </Section>
      ))}
    </Layout>
  );
};

export default Publications;
