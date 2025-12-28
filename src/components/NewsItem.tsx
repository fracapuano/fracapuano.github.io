interface NewsItemProps {
  date: string;
  content: string;
}

const NewsItem = ({ date, content }: NewsItemProps) => {
  return (
    <div className="flex gap-4 py-3 border-b border-border last:border-b-0">
      <span className="text-muted-foreground text-sm shrink-0 w-24 md:w-28">
        {date}
      </span>
      <div 
        className="text-[0.95rem] [&>a]:text-link [&>a]:hover:text-link-hover [&>a]:underline"
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  );
};

export default NewsItem;
