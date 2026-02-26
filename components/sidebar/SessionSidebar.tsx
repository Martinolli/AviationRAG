import styles from "../../styles/ChatWorkspace.module.css";
import { Session, SessionFilter } from "../../types/chat";

type SessionSidebarProps = {
  authEmail?: string | null;
  healthLabel: string;
  loadingSessions: boolean;
  filteredSessions: Session[];
  activeSessionId: string;
  sessionSearch: string;
  sessionFilter: SessionFilter;
  onClose: () => void;
  onSignOut: () => void;
  onCreateSession: () => void;
  onSessionSearchChange: (value: string) => void;
  onSessionFilterChange: (value: SessionFilter) => void;
  onOpenSession: (sessionId: string) => void;
  onTogglePinned: (session: Session) => void;
  onRenameSession: (session: Session) => void;
  onDeleteSession: (sessionId: string) => void;
};

export default function SessionSidebar({
  authEmail,
  healthLabel,
  loadingSessions,
  filteredSessions,
  activeSessionId,
  sessionSearch,
  sessionFilter,
  onClose,
  onSignOut,
  onCreateSession,
  onSessionSearchChange,
  onSessionFilterChange,
  onOpenSession,
  onTogglePinned,
  onRenameSession,
  onDeleteSession,
}: SessionSidebarProps) {
  return (
    <aside className={styles.sidebar} data-testid="session-sidebar">
      <div className={styles.sidebarTopRow}>
        <div className={styles.brandBlock}>
          <h1>AviationRAG</h1>
          <p>Safety and certification assistant</p>
        </div>
        <button type="button" className={styles.sidebarToggle} onClick={onClose}>
          Hide
        </button>
      </div>

      <div className={styles.userBlock}>
        <span>{authEmail || "Signed user"}</span>
        <button type="button" className={styles.userSignOut} onClick={onSignOut}>
          Sign out
        </button>
      </div>

      <button className={styles.primaryButton} onClick={onCreateSession} type="button">
        + New Conversation
      </button>

      <input
        className={styles.sessionSearch}
        type="text"
        placeholder="Search conversations..."
        value={sessionSearch}
        onChange={(event) => onSessionSearchChange(event.target.value)}
      />

      <div className={styles.filterRow}>
        <button
          type="button"
          className={`${styles.filterButton} ${
            sessionFilter === "all" ? styles.filterButtonActive : ""
          }`}
          onClick={() => onSessionFilterChange("all")}
        >
          All
        </button>
        <button
          type="button"
          className={`${styles.filterButton} ${
            sessionFilter === "recent" ? styles.filterButtonActive : ""
          }`}
          onClick={() => onSessionFilterChange("recent")}
        >
          Recent
        </button>
        <button
          type="button"
          className={`${styles.filterButton} ${
            sessionFilter === "pinned" ? styles.filterButtonActive : ""
          }`}
          onClick={() => onSessionFilterChange("pinned")}
        >
          Pinned
        </button>
      </div>

      <div className={styles.sidebarHeader}>
        <span>Conversations</span>
        <span className={styles.statusPill}>{healthLabel}</span>
      </div>

      <div className={styles.sessionList} data-testid="session-list">
        {loadingSessions ? <div className={styles.emptyState}>Loading sessions...</div> : null}
        {!loadingSessions && filteredSessions.length === 0 ? (
          <div className={styles.emptyState}>No sessions found.</div>
        ) : (
          filteredSessions.map((session) => (
            <div
              key={session.id}
              className={`${styles.sessionItem} ${
                session.id === activeSessionId ? styles.sessionItemActive : ""
              }`}
            >
              <button
                type="button"
                className={styles.sessionOpenButton}
                onClick={() => onOpenSession(session.id)}
              >
                <span className={styles.sessionTitle}>{session.title}</span>
                <span className={styles.sessionTime}>
                  {new Date(session.updatedAt).toLocaleString()}
                </span>
              </button>

              <div className={styles.sessionActions}>
                <button type="button" onClick={() => onTogglePinned(session)}>
                  {session.pinned ? "Unpin" : "Pin"}
                </button>
                <button type="button" onClick={() => onRenameSession(session)}>
                  Rename
                </button>
                <button type="button" onClick={() => onDeleteSession(session.id)}>
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
