import Head from "next/head";
import { useRouter } from "next/router";
import { signOut, useSession } from "next-auth/react";
import { useEffect, useMemo, useState } from "react";
import AppShell from "../components/layout/AppShell";
import SessionSidebar from "../components/sidebar/SessionSidebar";
import ChatPanel from "../components/chat/ChatPanel";
import SourceDrawer from "../components/sources/SourceDrawer";
import styles from "../styles/ChatWorkspace.module.css";
import { Citation, Message, Session, SessionFilter, SourceSnippet } from "../types/chat";

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

function shortTitleFromMessage(message: string) {
  const cleaned = message.trim().replace(/\s+/g, " ");
  if (!cleaned) return "New Session";
  return cleaned.split(" ").slice(0, 7).join(" ");
}

function parseSession(raw: any): Session {
  const updatedAt = String(raw?.updated_at || raw?.updatedAt || nowIso());
  const createdAt = String(raw?.created_at || raw?.createdAt || updatedAt);
  return {
    id: String(raw?.id || ""),
    title: String(raw?.title || "New Session"),
    createdAt,
    updatedAt,
    pinned: Boolean(raw?.pinned),
  };
}

function sortSessions(items: Session[]) {
  return [...items].sort((a, b) => {
    if (a.pinned !== b.pinned) {
      return a.pinned ? -1 : 1;
    }
    return b.updatedAt.localeCompare(a.updatedAt);
  });
}

export default function HomePage() {
  const router = useRouter();
  const { data: authSession, status: authStatus } = useSession();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [messagesBySession, setMessagesBySession] = useState<Record<string, Message[]>>({});
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [healthLabel, setHealthLabel] = useState("checking...");
  const [showSources, setShowSources] = useState(false);
  const [selectedSource, setSelectedSource] = useState<SourceSnippet | null>(null);
  const [errorText, setErrorText] = useState("");
  const [sessionSearch, setSessionSearch] = useState("");
  const [sessionFilter, setSessionFilter] = useState<SessionFilter>("all");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const activeMessages = useMemo(
    () => messagesBySession[activeSessionId] || [],
    [messagesBySession, activeSessionId],
  );

  const filteredSessions = useMemo(() => {
    const search = sessionSearch.trim().toLowerCase();
    let items = [...sessions];

    if (sessionFilter === "pinned") {
      items = items.filter((item) => item.pinned);
    } else if (sessionFilter === "recent") {
      items = items.filter((item) => !item.pinned).slice(0, 20);
    }

    if (search) {
      items = items.filter(
        (item) =>
          item.title.toLowerCase().includes(search) || item.id.toLowerCase().includes(search),
      );
    }

    return items;
  }, [sessions, sessionSearch, sessionFilter]);

  const loadSessions = async () => {
    setLoadingSessions(true);
    try {
      const res = await fetch("/api/chat/session?limit=200");
      if (res.status === 401) {
        await router.replace("/auth/signin");
        return;
      }
      const data = await res.json();
      if (!data?.success) {
        throw new Error(data?.error || "Failed to load sessions.");
      }
      const parsed = Array.isArray(data.sessions) ? data.sessions.map(parseSession) : [];
      setSessions(sortSessions(parsed));
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Session loading failed.");
    } finally {
      setLoadingSessions(false);
    }
  };

  useEffect(() => {
    if (authStatus !== "authenticated") {
      return;
    }
    void loadSessions();
  }, [authStatus]);

  useEffect(() => {
    if (authStatus === "unauthenticated") {
      void router.replace("/auth/signin");
    }
  }, [authStatus, router]);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      return;
    }
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
  }, [authStatus]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  useEffect(() => {
    if (window.innerWidth <= 980) {
      setSidebarOpen(false);
    }
  }, []);

  const createNewSession = async (titleSeed = "New Session") => {
    setErrorText("");
    try {
      const res = await fetch("/api/chat/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: shortTitleFromMessage(titleSeed) }),
      });
      const data = await res.json();
      if (!data?.success || !data?.session) {
        throw new Error(data?.error || "Failed to create session.");
      }
      const session = parseSession(data.session);
      setSessions((prev) => sortSessions([session, ...prev.filter((item) => item.id !== session.id)]));
      setActiveSessionId(session.id);
      setMessagesBySession((prev) => ({ ...prev, [session.id]: prev[session.id] || [] }));
      setSelectedSource(null);
      return session.id;
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Session creation failed.");
      return "";
    }
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
      setSessions((prev) =>
        sortSessions(
          prev.map((session) =>
            session.id === sessionId ? { ...session, updatedAt: nowIso() } : session,
          ),
        ),
      );
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "History loading failed.");
    } finally {
      setLoadingHistory(false);
    }
  };

  const patchSession = async (sessionId: string, payload: { title?: string; pinned?: boolean }) => {
    const res = await fetch(`/api/chat/session/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data?.success || !data?.session) {
      throw new Error(data?.error || "Failed to update session.");
    }
    const updated = parseSession(data.session);
    setSessions((prev) =>
      sortSessions(prev.map((session) => (session.id === updated.id ? updated : session))),
    );
  };

  const renameSession = async (session: Session) => {
    const nextTitle = prompt("Enter a new session title:", session.title);
    if (nextTitle === null) return;
    const title = nextTitle.trim();
    if (!title || title === session.title) return;

    try {
      await patchSession(session.id, { title });
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Rename failed.");
    }
  };

  const togglePinned = async (session: Session) => {
    try {
      await patchSession(session.id, { pinned: !session.pinned });
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Pin update failed.");
    }
  };

  const deleteSession = async (sessionId: string) => {
    const confirmed = window.confirm(
      "Delete this conversation and its stored history from AstraDB? This cannot be undone.",
    );
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/chat/session/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!data?.success) {
        throw new Error(data?.error || "Delete failed.");
      }

      setSessions((prev) => prev.filter((session) => session.id !== sessionId));
      setMessagesBySession((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });

      if (activeSessionId === sessionId) {
        setActiveSessionId("");
        setSelectedSource(null);
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Delete failed.");
    }
  };

  const openCitationSource = (message: Message, citation: Citation) => {
    const source = (message.sources || []).find(
      (item) => item.filename === citation.filename && item.chunk_id === citation.chunk_id,
    );
    if (source) {
      setSelectedSource(source);
      setShowSources(true);
      return;
    }
    setSelectedSource({
      filename: citation.filename,
      chunk_id: citation.chunk_id,
      text: "Source text is not available in this message payload.",
    });
    setShowSources(true);
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || sending) return;

    setErrorText("");
    setSending(true);

    let sessionId = activeSessionId;
    if (!sessionId) {
      sessionId = await createNewSession(message);
      if (!sessionId) {
        setSending(false);
        return;
      }
    }

    const userMsg: Message = {
      id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      role: "user",
      content: message,
      timestamp: nowIso(),
    };

    setInput("");
    setMessagesBySession((prev) => ({
      ...prev,
      [sessionId]: [...(prev[sessionId] || []), userMsg],
    }));

    const currentSession = sessions.find((item) => item.id === sessionId);
    const shouldPromoteTitle = !currentSession || currentSession.title === "New Session";

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

      setSessions((prev) =>
        sortSessions(
          prev.map((session) =>
            session.id === sessionId ? { ...session, updatedAt: nowIso() } : session,
          ),
        ),
      );

      if (shouldPromoteTitle) {
        await patchSession(sessionId, { title: shortTitleFromMessage(message) });
      }

      if (sources.length > 0) {
        setSelectedSource(sources[0]);
      }
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Message failed.");
    } finally {
      setSending(false);
    }
  };

  if (authStatus !== "authenticated") {
    return (
      <main className={styles.authLoading}>
        <p>{authStatus === "loading" ? "Checking authentication..." : "Redirecting to login..."}</p>
      </main>
    );
  }

  return (
    <>
      <Head>
        <title>AviationRAG Console</title>
        <meta
          name="description"
          content="Aviation document-grounded assistant with citations and chat history."
        />
      </Head>

      <AppShell
        sidebarOpen={sidebarOpen}
        showSources={showSources}
        onCloseSidebar={() => setSidebarOpen(false)}
        onCloseSources={() => setShowSources(false)}
        sidebar={
          <SessionSidebar
            authEmail={authSession?.user?.email}
            healthLabel={healthLabel}
            loadingSessions={loadingSessions}
            filteredSessions={filteredSessions}
            activeSessionId={activeSessionId}
            sessionSearch={sessionSearch}
            sessionFilter={sessionFilter}
            onClose={() => setSidebarOpen(false)}
            onSignOut={() => void signOut({ callbackUrl: "/auth/signin" })}
            onCreateSession={() => void createNewSession()}
            onSessionSearchChange={setSessionSearch}
            onSessionFilterChange={setSessionFilter}
            onOpenSession={(sessionId) => void loadSessionHistory(sessionId)}
            onTogglePinned={(session) => void togglePinned(session)}
            onRenameSession={(session) => void renameSession(session)}
            onDeleteSession={(sessionId) => void deleteSession(sessionId)}
          />
        }
        chatPanel={
          <ChatPanel
            sidebarOpen={sidebarOpen}
            showSources={showSources}
            loadingHistory={loadingHistory}
            activeMessages={activeMessages}
            errorText={errorText}
            input={input}
            sending={sending}
            onToggleSidebar={() => setSidebarOpen((value) => !value)}
            onToggleSources={() => setShowSources((value) => !value)}
            onOpenCitation={openCitationSource}
            onInputChange={setInput}
            onSend={() => void sendMessage()}
          />
        }
        sourceDrawer={
          <SourceDrawer
            showSources={showSources}
            selectedSource={selectedSource}
            onClose={() => setShowSources(false)}
          />
        }
      />
    </>
  );
}
