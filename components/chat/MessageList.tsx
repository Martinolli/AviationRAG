import styles from "../../styles/ChatWorkspace.module.css";
import { Citation, Message } from "../../types/chat";
import MessageMarkdown from "./MessageMarkdown";

type MessageListProps = {
  loadingHistory: boolean;
  activeMessages: Message[];
  onOpenCitation: (message: Message, citation: Citation) => void;
};

export default function MessageList({
  loadingHistory,
  activeMessages,
  onOpenCitation,
}: MessageListProps) {
  return (
    <div className={styles.messageList} data-testid="message-list">
      {loadingHistory ? <p className={styles.metaText}>Loading history...</p> : null}
      {activeMessages.length === 0 ? (
        <div className={styles.welcomeCard}>
          <h3>Ask certification and flight-test questions</h3>
          <p>Try: "According to Part 23, what are the landing gear drop test requirements?"</p>
        </div>
      ) : (
        activeMessages.map((message) => (
          <article
            key={message.id}
            className={`${styles.message} ${
              message.role === "user" ? styles.userMessage : styles.assistantMessage
            }`}
          >
            <div className={styles.messageMeta}>
              <span>{message.role === "user" ? "You" : "AviationAI"}</span>
              <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
            </div>
            {message.role === "assistant" ? (
              <MessageMarkdown content={message.content} />
            ) : (
              <p className={styles.messageText}>{message.content}</p>
            )}
            {message.citations && message.citations.length > 0 ? (
              <div className={styles.citations}>
                {message.citations.map((citation, idx) => (
                  <button
                    type="button"
                    className={styles.citationButton}
                    data-testid="citation-chip"
                    key={`${citation.filename}_${citation.chunk_id}_${idx}`}
                    onClick={() => onOpenCitation(message, citation)}
                  >
                    [{citation.filename} | {citation.chunk_id}]
                  </button>
                ))}
              </div>
            ) : null}
          </article>
        ))
      )}
    </div>
  );
}
