import { useParams, Link, Navigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Layout from "@/components/Layout";
import { blogPostsData } from "@/data/blogPosts";
import "katex/dist/katex.min.css";
import { cn } from "@/lib/utils";

const BlogPost = () => {
  const { slug } = useParams<{ slug: string }>();
  const post = blogPostsData.find((p) => p.slug === slug);

  if (!post) {
    return <Navigate to="/blog" replace />;
  }

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
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeKatex]}
            components={{
              h1: ({ children }) => <h1 className="text-2xl md:text-3xl font-bold mt-8 mb-4">{children}</h1>,
              h2: ({ children }) => <h2 className="text-xl font-bold mt-8 mb-4">{children}</h2>,
              h3: ({ children }) => <h3 className="text-lg font-bold mt-6 mb-3">{children}</h3>,
              h4: ({ children }) => <h4 className="text-base font-bold mt-6 mb-3">{children}</h4>,
              p: ({ children }) => <p className="mb-4 leading-relaxed text-justify">{children}</p>,
              ul: ({ children }) => <ul className="pl-6 my-4 list-disc space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="pl-6 my-4 list-decimal space-y-1">{children}</ol>,
              li: ({ children }) => <li className="mb-1">{children}</li>,
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-muted-foreground/30 pl-4 italic text-muted-foreground my-6">
                  {children}
                </blockquote>
              ),
              figure: ({ className, children, ...props }) => {
                const isLeft = className?.includes('wrap-left');
                return (
                  <figure 
                    className={cn(
                      "mb-4 w-full md:w-auto md:max-w-[45%]",
                      isLeft ? "md:float-left md:mr-6" : "md:float-right md:ml-6",
                      className
                    )} 
                    {...props}
                  >
                    {children}
                  </figure>
                );
              },
              figcaption: ({ children }) => (
                <figcaption 
                  className="text-center text-[16px] text-[#1f1f1f] mt-2"
                  style={{ 
                    backgroundClip: 'unset',
                    WebkitBackgroundClip: 'unset' 
                  }}
                >
                  {children}
                </figcaption>
              ),
              a: ({ href, children }) => (
                <a 
                  href={href} 
                  className="text-link hover:text-link-hover hover:underline"
                  target={href?.startsWith('http') ? '_blank' : undefined}
                  rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
                >
                  {children}
                </a>
              ),
              code: ({ className, children, ...props }) => {
                const match = /language-(\w+)/.exec(className || "");
                const isInline = !match;
                return isInline ? (
                  <code className="font-mono text-sm bg-muted px-1 py-0.5 rounded border border-border" {...props}>
                    {children}
                  </code>
                ) : (
                  <div className="my-4">
                    {match && (
                      <div className="text-xs text-muted-foreground mb-1 font-mono uppercase">
                        {match[1]}
                      </div>
                    )}
                    <SyntaxHighlighter
                      style={oneLight}
                      language={match ? match[1] : undefined}
                      PreTag="div"
                      className="rounded-md border border-border !bg-muted/50"
                      customStyle={{ margin: 0, padding: '1rem', backgroundColor: 'transparent' }}
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                );
              },
              div: ({ className, children }) => {
                if (className === 'math-display') {
                  return <div className="my-6 overflow-x-auto">{children}</div>;
                }
                return <div className={className}>{children}</div>;
              }
            }}
          >
            {post.content}
          </ReactMarkdown>
        </div>
      </article>
    </Layout>
  );
};

export default BlogPost;
