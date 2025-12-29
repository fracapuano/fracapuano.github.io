interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock = ({ code, language = "python" }: CodeBlockProps) => {
  return (
    <div className="my-4">
      {language && (
        <div className="text-xs text-muted-foreground mb-1 font-mono">
          {language}
        </div>
      )}
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
