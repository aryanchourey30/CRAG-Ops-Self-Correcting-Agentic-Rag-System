import { useState } from "react";
import { sendChat, uploadPdf } from "./api";

function Message({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-header">
        <span>{message.role === "user" ? "You" : "CRAG-Ops"}</span>
        {message.traceId ? <code>{message.traceId}</code> : null}
      </div>
      <p>{message.content}</p>
      {message.citations?.length ? (
        <div className="citations">
          {message.citations.map((citation, index) => (
            <div className="citation" key={`${citation.source}-${index}`}>
              <strong>{citation.source}</strong>
              {citation.page ? <span>Page {citation.page}</span> : null}
              {citation.url ? (
                <a href={citation.url} target="_blank" rel="noreferrer">
                  Open source
                </a>
              ) : null}
              {citation.snippet ? <small>{citation.snippet}</small> : null}
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Ask a question in web mode or upload a PDF and switch to document mode for grounded CRAG answers.",
    },
  ]);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("web");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadState, setUploadState] = useState({
    documentId: "",
    filename: "",
    chunkCount: 0,
  });

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    try {
      const result = await uploadPdf(file);
      setUploadState({
        documentId: result.document_id,
        filename: result.filename,
        chunkCount: result.chunk_count,
      });
      setMode("pdf");
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `Indexed ${result.filename} into ${result.chunk_count} chunks across ${result.pages} pages.`,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
        },
      ]);
    } finally {
      setIsLoading(false);
      event.target.value = "";
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!query.trim()) return;

    const nextUserMessage = { role: "user", content: query };
    setMessages((current) => [...current, nextUserMessage]);
    setIsLoading(true);

    try {
      const response = await sendChat({
        query,
        mode,
        document_id: mode === "pdf" ? uploadState.documentId : null,
      });

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          traceId: response.trace_id,
        },
      ]);
      setQuery("");
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: error.message,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Self-Correcting Agentic RAG</p>
          <h1>CRAG-Ops</h1>
          <p className="lede">
            Retrieval gets scored, corrected, and validated before an answer reaches the user.
          </p>
        </div>
        <div className="control-card">
          <label className="mode-toggle">
            <span>Mode</span>
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="web">General Q&amp;A</option>
              <option value="pdf">PDF-based Q&amp;A</option>
            </select>
          </label>
          <label className="upload-box">
            <span>Upload PDF</span>
            <input type="file" accept="application/pdf" onChange={handleUpload} />
          </label>
          <div className="doc-meta">
            <span>{uploadState.filename || "No PDF indexed yet"}</span>
            {uploadState.documentId ? <code>{uploadState.documentId}</code> : null}
            {uploadState.chunkCount ? <small>{uploadState.chunkCount} chunks ready</small> : null}
          </div>
        </div>
      </section>

      <section className="chat-panel">
        <div className="message-list">
          {messages.map((message, index) => (
            <Message key={`${message.role}-${index}`} message={message} />
          ))}
          {isLoading ? <div className="loading">Running retrieval, evaluation, and grounding checks...</div> : null}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={4}
            placeholder="Ask about your PDF or a general research question..."
          />
          <button type="submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </section>
    </main>
  );
}
