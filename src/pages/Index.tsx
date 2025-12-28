import { useState } from "react";
import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import Section from "@/components/Section";
import NewsItem from "@/components/NewsItem";
import PublicationItem from "@/components/PublicationItem";
import { newsData } from "@/data/news";
import { publicationsData } from "@/data/publications";
import profileImage from "@/assets/profile.jpeg";

const INITIAL_NEWS_COUNT = 5;

const Index = () => {
  const [showAllNews, setShowAllNews] = useState(false);
  const displayedNews = showAllNews ? newsData : newsData.slice(0, INITIAL_NEWS_COUNT);

  return (
    <Layout>
      {/* Hero / About */}
      <section className="mb-12">
        <div className="flex flex-col sm:flex-row gap-6 sm:gap-8 items-start">
          {/* Profile Photo */}
          <div className="shrink-0">
            <img 
              src={profileImage} 
              alt="Francesco Capuano" 
              className="w-32 h-40 object-cover border border-border"
            />
          </div>
          
          {/* Bio */}
          <div className="flex-1">
            <h1 className="text-2xl md:text-3xl font-bold mb-2">Francesco Capuano</h1>
            <p className="text-muted-foreground mb-4">
              DPhil Student in Applied AI<br />
              University of Oxford · <a href="https://ori.ox.ac.uk/labs/a2i/" target="_blank" rel="noopener noreferrer">A2I</a> & <a href="https://foersterlab.com/" target="_blank" rel="noopener noreferrer">FLAIR</a> Labs
            </p>
            <p className="mb-4 leading-relaxed">
              Ciao 👋 Francesco here :) I deal with Deep Learning, and I am particularly interested in Robot Learning. 
              I am a first-year DPhil student at Oxford working on enabling complex behavior in robots 🤖
            </p>
            <p className="mb-4 leading-relaxed">
              I am supervised by <a href="https://ori.ox.ac.uk/people/ingmar-posner/" target="_blank" rel="noopener noreferrer">Ingmar Posner</a> and <a href="https://www.jakobfoerster.com/" target="_blank" rel="noopener noreferrer">Jakob Foerster</a>.
            </p>
            <p className="text-sm text-muted-foreground">
              <a href="mailto:capuano@robots.ox.ac.uk">capuano@robots.ox.ac.uk</a>
              {" · "}
              <a href="https://github.com/fracapuano" target="_blank" rel="noopener noreferrer">GitHub</a>
              {" · "}
              <a href="https://scholar.google.it/citations?user=2lXGNlkAAAAJ" target="_blank" rel="noopener noreferrer">Scholar</a>
              {" · "}
              <a href="https://www.linkedin.com/in/fracapuano/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
              {" · "}
              <a href="https://x.com/fra__capuano" target="_blank" rel="noopener noreferrer">X</a>
            </p>
          </div>
        </div>
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
              award={pub.award}
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
