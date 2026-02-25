import Head from "next/head";
import { useEffect, useMemo, useState } from "react";
import styles from "../styles/ChatWorkspace.module.css";

type Citation = {
  filename: string;
  chunk_id: string;
};

type SourceSnippet = {
  filename: string;
  chunk_id: string;
  text: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  sources?: SourceSnippet[];
};

type Session = {
  id: string;
  title: string;
  updatedAt: string;
};

function normalizeDisplayText(value: string) {
  return String(value || "")
    .normalize("NFKC")
    .replace(/�/g, "'")
    .replace(/\u2018|\u2019/g, "'")
    .replace(/\u201c|\u201d/g, '"')
    .replace(/\u2013|\u2014/g, "-");
}

function nowIso() {
  return new Date().toISOString();
}

function createSessionId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function shortTitleFromMessage(message: string) {
  const cleaned = message.trim().replace(/\s+/g, " ");
  if (!cleaned) {
    return "New Session";
  }
  return cleaned.split(" ").slice(0, 7).join(" ");
}

export default function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [messagesBySession, setMessagesBySession] = useState<Record<string, Message[]>>({});
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [healthLabel, setHealthLabel] = useState("checking...");
  const [showSources, setShowSources] = useState(true);
  const [selectedSource, setSelectedSource] = useState<SourceSnippet | null>(null);
  const [errorText, setErrorText] = useState("");

  const activeMessages = useMemo(
    () => messagesBySession[activeSessionId] || [],
    [messagesBySession, activeSessionId],
  );

  useEffect(() => {
    let alive = true;
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        if (!alive) return;
        setHealthLabel(data?.success ? "online" : "degraded");
      })
      .catch(() => {
        if (!alive) return;
        setHealthLabel("offline");
      });
    return () => {
      alive = false;
    };
  }, []);

  const createNewSession = () => {
    const id = createSessionId();
    const session: Session = {
      id,
      title: "New Session",
      updatedAt: nowIso(),
    };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(id);
    setMessagesBySession((prev) => ({ ...prev, [id]: [] }));
    setSelectedSource(null);
    setErrorText("");
  };

  const loadSessionHistory = async (sessionId: string) => {
    setLoadingHistory(true);
    setErrorText("");
    try {
      const res = await fetch(`/api/chat/history/${encodeURIComponent(sessionId)}?limit=20`);
      const data = await res.json();
      if (!data?.success) {
        throw new Error(data?.error || "Failed to load session history.");
      }

      const rows = Array.isArray(data.messages) ? [...data.messages].reverse() : [];
      const normalized: Message[] = [];

      rows.forEach((row: any) => {
        const timestamp = String(row.timestamp || nowIso());
        if (row.user_query) {
          normalized.push({
            id: `${timestamp}_u_${Math.random().toString(36).slice(2, 8)}`,
            role: "user",
            content: normalizeDisplayText(String(row.user_query)),
            timestamp,
          });
        }
        if (row.ai_response) {
          normalized.push({
            id: `${timestamp}_a_${Math.random().toString(36).slice(2, 8)}`,
            role: "assistant",
            content: normalizeDisplayText(String(row.ai_response)),
            timestamp,
          });
        }
      });

      setMessagesBySession((prev) => ({ ...prev, [sessionId]: normalized }));
      setActiveSessionId(sessionId);
      setSelectedSource(null);
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "History loading failed.");
    } finally {
      setLoadingHistory(false);
    }
  };

  const upsertSession = (sessionId: string, titleSeed: string) => {
    const title = shortTitleFromMessage(titleSeed);
    setSessions((prev) => {
      const exists = prev.find((s) => s.id === sessionId);
      if (!exists) {
        return [{ id: sessionId, title, updatedAt: nowIso() }, ...prev];
      }
      return prev
        .map((s) =>
          s.id === sessionId
            ? { ...s, title: s.title === "New Session" ? title : s.title, updatedAt: nowIso() }
            : s,
        )
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    });
  };

  const openCitationSource = (message: Message, citation: Citation) => {
    const source = (message.sources || []).find(
      (item) => item.filename === citation.filename && item.chunk_id === citation.chunk_id,
    );
    if (source) {
      setSelectedSource(source);
      return;
    }
    setSelectedSource({
      filename: citation.filename,
      chunk_id: citation.chunk_id,
      text: "Source text is not available in this message payload.",
    });
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || sending) return;

    const sessionId = activeSessionId || createSessionId();
    const userMsg: Message = {
      id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role: "user",
      content: message,
      timestamp: nowIso(),
    };

    setInput("");
    setErrorText("");
    setSending(true);
    setActiveSessionId(sessionId);
    upsertSession(sessionId, message);
    setMessagesBySession((prev) => ({
      ...prev,
      [sessionId]: [...(prev[sessionId] || []), userMsg],
    }));

    try {
      const res = await fetch("/api/chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message,
          store: true,
        }),
      });
      const data = await res.json();
      if (!data?.success) {
        throw new Error(data?.error || "Ask request failed.");
      }

      const sources: SourceSnippet[] = Array.isArray(data.sources)
        ? data.sources.map((source: any) => ({
            filename: String(source.filename || ""),
            chunk_id: String(source.chunk_id || ""),
            text: normalizeDisplayText(String(source.text || "")),
          }))
        : [];

      const assistantMsg: Message = {
        id: `a_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        role: "assistant",
        content: normalizeDisplayText(String(data.answer || "")),
        citations: Array.isArray(data.citations) ? data.citations : [],
        sources,
        timestamp: nowIso(),
      };

      setMessagesBySession((prev) => ({
        ...prev,
        [sessionId]: [...(prev[sessionId] || []), assistantMsg],
      }));
      upsertSession(sessionId, message);

      if (sources.length > 0) {
        setSelectedSource(sources[0]);
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Message failed.");
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <Head>
        <title>AviationRAG Console</title>
        <meta
          name="description"
          content="Aviation document-grounded assistant with citations and chat history."
        />
      </Head>

      <main className={styles.workspace}>
        <aside className={styles.sidebar}>
          <div className={styles.brandBlock}>
            <h1>AviationRAG</h1>
            <p>Safety and certification assistant</p>
          </div>

          <button className={styles.primaryButton} onClick={createNewSession}>
            + New Conversation
          </button>

          <div className={styles.sidebarHeader}>
            <span>Conversations</span>
            <span className={styles.statusPill}>{healthLabel}</span>
          </div>

          <div className={styles.sessionList}>
            {sessions.length === 0 ? (
              <div className={styles.emptyState}>No sessions yet.</div>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  className={`${styles.sessionItem} ${
                    session.id === activeSessionId ? styles.sessionItemActive : ""
                  }`}
                  onClick={() => loadSessionHistory(session.id)}
                >
                  <span className={styles.sessionTitle}>{session.title}</span>
                  <span className={styles.sessionTime}>
                    {new Date(session.updatedAt).toLocaleString()}
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className={styles.chatPanel}>
          <header className={styles.chatHeader}>
            <div>
              <h2>Expert Chat</h2>
              <p>Grounded answers with citations for aviation documents and standards.</p>
            </div>
            <button
              className={styles.ghostButton}
              onClick={() => setShowSources((v) => !v)}
              type="button"
            >
              {showSources ? "Hide Sources" : "Show Sources"}
            </button>
          </header>

          <div className={styles.chatBody}>
            <div className={styles.messageArea}>
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
                    <p className={styles.messageText}>{message.content}</p>
                    {showSources && message.citations && message.citations.length > 0 ? (
                      <div className={styles.citations}>
                        {message.citations.map((citation, idx) => (
                          <button
                            type="button"
                            className={styles.citationButton}
                            key={`${citation.filename}_${citation.chunk_id}_${idx}`}
                            onClick={() => openCitationSource(message, citation)}
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

            {showSources ? (
              <aside className={styles.sourcePanel}>
                <h3>Source Viewer</h3>
                {selectedSource ? (
                  <>
                    <div className={styles.sourceMeta}>
                      <span>{selectedSource.filename}</span>
                      <span>{selectedSource.chunk_id}</span>
                    </div>
                    <pre className={styles.sourceText}>{selectedSource.text}</pre>
                  </>
                ) : (
                  <p className={styles.sourceEmpty}>
                    Click a citation to open the exact passage used in the answer.
                  </p>
                )}
              </aside>
            ) : null}
          </div>

          {errorText ? <div className={styles.errorBar}>{errorText}</div> : null}

          <footer className={styles.composer}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a standards-based aviation question..."
              rows={3}
            />
            <button onClick={sendMessage} disabled={sending || !input.trim()}>
              {sending ? "Thinking..." : "Send"}
            </button>
          </footer>
        </section>
      </main>
    </>
  );
}

