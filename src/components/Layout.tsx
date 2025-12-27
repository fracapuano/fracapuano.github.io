import { Link, useLocation } from "react-router-dom";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const location = useLocation();
  
  const navLinks = [
    { href: "/", label: "Home" },
    { href: "/publications", label: "Publications" },
    { href: "/blog", label: "Blog" },
  ];

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container py-6">
          <nav className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <Link 
              to="/" 
              className="text-lg font-bold text-foreground no-underline hover:no-underline"
            >
              Your Name
            </Link>
            
            <ul className="flex items-center gap-6 text-[0.95rem]">
              {navLinks.map((link) => (
                <li key={link.href}>
                  <Link
                    to={link.href}
                    className={`transition-colors ${
                      isActive(link.href)
                        ? "text-foreground font-medium"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="container py-10 md:py-14">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="container py-6 text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} Your Name. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
