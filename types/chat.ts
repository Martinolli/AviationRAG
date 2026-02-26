export type Citation = {
  filename: string;
  chunk_id: string;
};

export type SourceSnippet = {
  filename: string;
  chunk_id: string;
  text: string;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  citations?: Citation[];
  sources?: SourceSnippet[];
};

export type Session = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  pinned: boolean;
};

export type SessionFilter = "all" | "recent" | "pinned";
