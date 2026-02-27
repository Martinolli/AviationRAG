import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import styles from "../../styles/ChatWorkspace.module.css";

type MessageMarkdownProps = {
  content: string;
};

const markdownSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [
      ...((defaultSchema.attributes?.code as any[]) || []),
      ["className", "math-inline", "math-display"],
    ],
  },
};

export default function MessageMarkdown({ content }: MessageMarkdownProps) {
  return (
    <div className={styles.messageMarkdown}>
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkMath]}
        rehypePlugins={[[rehypeSanitize, markdownSchema], rehypeKatex]}
        components={{
          a: ({ ...props }) => <a {...props} target="_blank" rel="noreferrer noopener" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
