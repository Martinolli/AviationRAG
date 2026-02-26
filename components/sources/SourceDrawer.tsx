import styles from "../../styles/ChatWorkspace.module.css";
import { SourceSnippet } from "../../types/chat";

type SourceDrawerProps = {
  showSources: boolean;
  selectedSource: SourceSnippet | null;
  onClose: () => void;
};

export default function SourceDrawer({
  showSources,
  selectedSource,
  onClose,
}: SourceDrawerProps) {
  return (
    <aside
      className={`${styles.sourceDrawer} ${showSources ? styles.drawerOpen : ""}`}
      data-testid="source-drawer"
    >
      <div className={styles.drawerHeader}>
        <h3>Source Viewer</h3>
        <button type="button" className={styles.drawerClose} onClick={onClose}>
          Close
        </button>
      </div>

      {selectedSource ? (
        <article className={styles.sourceCard}>
          <p className={styles.sourceTitle}>{selectedSource.filename}</p>
          <p className={styles.sourceMeta}>{selectedSource.chunk_id}</p>
          <pre className={styles.sourceText}>{selectedSource.text}</pre>
        </article>
      ) : (
        <p className={styles.sourceEmpty}>
          Click a citation to open the exact passage used in the answer.
        </p>
      )}
    </aside>
  );
}
