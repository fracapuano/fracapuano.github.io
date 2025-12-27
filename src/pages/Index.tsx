import { useState } from "react";
import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import Section from "@/components/Section";
import NewsItem from "@/components/NewsItem";
import PublicationItem from "@/components/PublicationItem";
import { newsData } from "@/data/news";
import { publicationsData } from "@/data/publications";

const INITIAL_NEWS_COUNT = 5;

const Index = () => {
  const [showAllNews, setShowAllNews] = useState(false);
  const displayedNews = showAllNews ? newsData : newsData.slice(0, INITIAL_NEWS_COUNT);

  return (
    <Layout>
      {/* Hero / About */}
      <section className="mb-12">
        <h1 className="text-2xl md:text-3xl font-bold mb-4">Your Name</h1>
        <p className="text-muted-foreground mb-4">
          DPhil Student in Machine Learning<br />
          University of Oxford
        </p>
        <p className="mb-4 leading-relaxed">
          I am a doctoral student at Oxford, working on geometric deep learning 
          and generative models for scientific applications. My research focuses 
          on developing equivariant neural networks for molecular modeling and 
          drug discovery.
        </p>
        <p className="mb-4 leading-relaxed">
          Previously, I completed my MSc at ETH Zürich and interned at DeepMind. 
          I am broadly interested in the intersection of machine learning, 
          physics, and chemistry.
        </p>
        <p className="text-sm text-muted-foreground">
          Contact: <a href="mailto:your.email@ox.ac.uk">your.email@ox.ac.uk</a>
          {" · "}
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
          {" · "}
          <a href="https://scholar.google.com" target="_blank" rel="noopener noreferrer">Google Scholar</a>
        </p>
      </section>

      {/* News */}
      <Section title="News">
        <div>
          {displayedNews.map((news) => (
            <NewsItem key={news.id} date={news.date} content={news.content} />
          ))}
        </div>
        {newsData.length > INITIAL_NEWS_COUNT && (
          <button
            onClick={() => setShowAllNews(!showAllNews)}
            className="mt-4 text-sm text-link hover:text-link-hover transition-colors"
          >
            {showAllNews ? "Show less" : `Show all (${newsData.length})`}
          </button>
        )}
      </Section>

      {/* Selected Publications */}
      <Section title="Selected Publications">
        <div>
          {publicationsData.slice(0, 3).map((pub) => (
            <PublicationItem
              key={pub.id}
              title={pub.title}
              authors={pub.authors}
              venue={pub.venue}
              year={pub.year}
              links={pub.links}
            />
          ))}
        </div>
        <Link 
          to="/publications" 
          className="inline-block mt-4 text-sm text-link hover:text-link-hover"
        >
          View all publications →
        </Link>
      </Section>
    </Layout>
  );
};

export default Index;
