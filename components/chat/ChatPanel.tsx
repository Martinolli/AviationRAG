import MessageList from "./MessageList";
import Composer from "./Composer";
import styles from "../../styles/ChatWorkspace.module.css";
import { Citation, Message } from "../../types/chat";

type ChatPanelProps = {
  sidebarOpen: boolean;
  showSources: boolean;
  loadingHistory: boolean;
  activeMessages: Message[];
  errorText: string;
  input: string;
  sending: boolean;
  onToggleSidebar: () => void;
  onToggleSources: () => void;
  onOpenCitation: (message: Message, citation: Citation) => void;
  onInputChange: (value: string) => void;
  onSend: () => void;
};

export default function ChatPanel({
  sidebarOpen,
  showSources,
  loadingHistory,
  activeMessages,
  errorText,
  input,
  sending,
  onToggleSidebar,
  onToggleSources,
  onOpenCitation,
  onInputChange,
  onSend,
}: ChatPanelProps) {
  return (
    <section className={styles.chatPanel} data-testid="chat-panel">
      <header className={styles.chatHeader}>
        <div>
          <h2>Expert Chat</h2>
          <p>Grounded answers with citations for aviation documents and standards.</p>
        </div>
        <div className={styles.headerActions}>
          <button className={styles.ghostButton} onClick={onToggleSidebar} type="button">
            {sidebarOpen ? "Hide Sidebar" : "Show Sidebar"}
          </button>
          <button className={styles.ghostButton} onClick={onToggleSources} type="button">
            {showSources ? "Hide Sources" : "Show Sources"}
          </button>
        </div>
      </header>

      <div className={styles.chatBody}>
        <div className={styles.messageArea} data-testid="message-area">
          <MessageList
            loadingHistory={loadingHistory}
            activeMessages={activeMessages}
            onOpenCitation={onOpenCitation}
          />
        </div>
      </div>

      {errorText ? <div className={styles.errorBar}>{errorText}</div> : null}

      <Composer input={input} sending={sending} onInputChange={onInputChange} onSend={onSend} />
    </section>
  );
}
