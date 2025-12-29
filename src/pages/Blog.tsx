import { Link } from "react-router-dom";
import Layout from "@/components/Layout";
import { blogPostsData } from "@/data/blogPosts";

const Blog = () => {
  return (
    <Layout>
      <h1 className="text-2xl md:text-3xl font-bold mb-8">Blog</h1>
      
      <div className="space-y-8">
        {blogPostsData.map((post) => (
          <article key={post.id} className="border-b border-border pb-6 last:border-b-0">
            <Link to={`/blog/${post.slug}`}>
              <h2 className="text-lg font-semibold mb-2 hover:text-link transition-colors">
                {post.title}
              </h2>
            </Link>
            <p className="text-sm text-muted-foreground mb-2">{post.date}</p>
            <p className="text-[0.95rem] text-muted-foreground leading-relaxed">
              {post.excerpt}
            </p>
            <Link 
              to={`/blog/${post.slug}`}
              className="inline-block mt-2 text-sm text-link hover:text-link-hover"
            >
              Read more →
            </Link>
          </article>
        ))}
      </div>
    </Layout>
  );
};

export default Blog;
