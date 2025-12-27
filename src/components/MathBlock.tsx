import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

interface MathBlockProps {
  math: string;
  display?: boolean;
}

const MathBlock = ({ math, display = false }: MathBlockProps) => {
  if (display) {
    return <BlockMath math={math} />;
  }
  return <InlineMath math={math} />;
};

export default MathBlock;
