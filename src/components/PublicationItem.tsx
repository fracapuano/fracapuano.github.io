interface PublicationItemProps {
  title: string;
  authors: string;
  venue: string;
  year: string;
  links?: { label: string; href: string }[];
  award?: string;
}

const PublicationItem = ({ title, authors, venue, year, links, award }: PublicationItemProps) => {
  return (
    <div className="py-4 border-b border-border last:border-b-0">
      <h3 className="text-base font-semibold mb-1 leading-snug">{title}</h3>
      <p className="text-muted-foreground text-[0.9rem] mb-1">{authors}</p>
      <p className="text-muted-foreground text-[0.9rem] italic">
        {venue}, {year}
      </p>
      {award && (
        <p className="text-sm text-amber-600 dark:text-amber-400 font-medium mt-1">
          🏆 {award}
        </p>
      )}
      {links && links.length > 0 && (
        <div className="flex gap-3 mt-2 text-sm">
          {links.map((link, index) => (
            <a
              key={index}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-link hover:text-link-hover"
            >
              [{link.label}]
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

export default PublicationItem;
