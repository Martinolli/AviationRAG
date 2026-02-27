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

export type UploadStatus =
  | "uploaded"
  | "processing"
  | "embedded"
  | "available"
  | "needs_review"
  | "failed";

export type UploadJob = {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: UploadStatus;
  message: string;
  created_at: string;
  updated_at: string;
};
