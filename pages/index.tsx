import { useState } from 'react';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [query, setQuery] = useState('');
  interface QueryResponse {
    response: string;
    results: {
      chunk_id: string;
      similarity: number;
      text: string;
    }[];
  }
  const [response, setResponse] = useState<QueryResponse | null>(null);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse(data);
      console.log("Backend response in frontend:", data); // Log the backend response
      setResponse(data); // Assign the entire backend response
      console.log("Frontend state:", data); // Log the frontend state after updating
    } catch (error) {
      console.error('Error:', error);
      setResponse(null); // Reset response state on error
    }
  };
  return (
    <div className={styles.container}>
      <main className={styles.main}>
        <h1 className={styles.title}>AviationAI</h1>
        <p className={styles.description}>Enter your query below:</p>
        <form onSubmit={handleSubmit} className={styles.form}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className={styles.input}
            placeholder="Enter your query"
          />
          <button type="submit" className={styles.button}>Submit</button>
        </form>
        {response && response.results && response.results.length > 0 ? (
          <div className={styles.response}>
            <h2>Generated Response:</h2>
            <p>{response.response}</p>
            <h3>Retrieved Context:</h3>
            <ul>
              {response.results.map((result, index) => (
                <li key={index}>
                  <strong>Similarity:</strong> {result.similarity.toFixed(2)}
                  <br />
                  {result.text}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p>No results to display. Please submit a query.</p>
        )}
      </main>
    </div>
  );
}