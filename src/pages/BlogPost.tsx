import { useParams, Link, Navigate } from "react-router-dom";
import Layout from "@/components/Layout";
import { blogPostsData } from "@/data/blogPosts";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

const BlogPost = () => {
  const { slug } = useParams<{ slug: string }>();
  const post = blogPostsData.find((p) => p.slug === slug);

  if (!post) {
    return <Navigate to="/blog" replace />;
  }

  // Simple markdown-like parser for the content
  const renderContent = (content: string) => {
    const lines = content.trim().split("\n");
    const elements: JSX.Element[] = [];
    let inCodeBlock = false;
    let codeContent = "";
    let codeLanguage = "";
    let key = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Code blocks
      if (line.startsWith("```")) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          codeLanguage = line.slice(3).trim();
          codeContent = "";
        } else {
          inCodeBlock = false;
          elements.push(
            <div key={key++} className="my-4">
              {codeLanguage && (
                <div className="text-xs text-muted-foreground mb-1 font-mono">
                  {codeLanguage}
                </div>
              )}
              <pre>
                <code>{codeContent.trim()}</code>
              </pre>
            </div>
          );
        }
        continue;
      }

      if (inCodeBlock) {
        codeContent += line + "\n";
        continue;
      }

      // Empty lines
      if (line.trim() === "") {
        continue;
      }

      // Headers
      if (line.startsWith("## ")) {
        elements.push(
          <h2 key={key++} className="text-xl font-bold mt-8 mb-4">
            {line.slice(3)}
          </h2>
        );
        continue;
      }

      if (line.startsWith("### ")) {
        elements.push(
          <h3 key={key++} className="text-lg font-bold mt-6 mb-3">
            {line.slice(4)}
          </h3>
        );
        continue;
      }

      // Display math ($$...$$)
      if (line.startsWith("$$") && line.endsWith("$$") && line.length > 4) {
        const math = line.slice(2, -2);
        elements.push(
          <div key={key++} className="my-6 overflow-x-auto">
            <BlockMath math={math} />
          </div>
        );
        continue;
      }

      // Multi-line display math
      if (line.startsWith("$$")) {
        let mathContent = "";
        i++;
        while (i < lines.length && !lines[i].startsWith("$$")) {
          mathContent += lines[i] + "\n";
          i++;
        }
        elements.push(
          <div key={key++} className="my-6 overflow-x-auto">
            <BlockMath math={mathContent.trim()} />
          </div>
        );
        continue;
      }

      // Regular paragraph with inline math
      const parts = line.split(/(\$[^$]+\$)/g);
      const parsedParts = parts.map((part, idx) => {
        if (part.startsWith("$") && part.endsWith("$")) {
          const math = part.slice(1, -1);
          return <InlineMath key={idx} math={math} />;
        }
        // Handle inline code
        const codeParts = part.split(/(`[^`]+`)/g);
        return codeParts.map((codePart, codeIdx) => {
          if (codePart.startsWith("`") && codePart.endsWith("`")) {
            return <code key={codeIdx}>{codePart.slice(1, -1)}</code>;
          }
          return codePart;
        });
      });

      elements.push(
        <p key={key++} className="mb-4 leading-relaxed">
          {parsedParts}
        </p>
      );
    }

    return elements;
  };

  return (
    <Layout>
      <article>
        <header className="mb-8">
          <Link 
            to="/blog" 
            className="text-sm text-muted-foreground hover:text-foreground mb-4 inline-block"
          >
            ← Back to blog
          </Link>
          <h1 className="text-2xl md:text-3xl font-bold mb-2">{post.title}</h1>
          <p className="text-muted-foreground">{post.date}</p>
        </header>
        
        <div className="prose-academic">
          {renderContent(post.content)}
        </div>
      </article>
    </Layout>
  );
};

export default BlogPost;
