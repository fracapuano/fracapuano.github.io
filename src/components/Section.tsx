interface SectionProps {
  title: string;
  children: React.ReactNode;
  className?: string;
}

const Section = ({ title, children, className = "" }: SectionProps) => {
  return (
    <section className={`mb-12 ${className}`}>
      <h2 className="text-xl font-bold mb-4 pb-2 border-b border-border">
        {title}
      </h2>
      {children}
    </section>
  );
};

export default Section;
