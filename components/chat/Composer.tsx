import styles from "../../styles/ChatWorkspace.module.css";

type ComposerProps = {
  input: string;
  sending: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
};

export default function Composer({ input, sending, onInputChange, onSend }: ComposerProps) {
  return (
    <footer className={styles.composer} data-testid="composer">
      <textarea
        value={input}
        onChange={(event) => onInputChange(event.target.value)}
        placeholder="Ask a standards-based aviation question..."
        rows={3}
      />
      <button onClick={onSend} disabled={sending || !input.trim()}>
        {sending ? "Thinking..." : "Send"}
      </button>
    </footer>
  );
}
