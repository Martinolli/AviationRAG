// Updated index.tsx
import { useState } from 'react';
import styles from '../styles/Home.module.css';

export default function Home() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setResponse("Please enter a valid query.");
      return;
    }

    setIsLoading(true);
    setResponse("");
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error:', error);
      setResponse('An error occurred while processing your request.');
    } finally {
      setIsLoading(false);
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
        {isLoading && <div className={styles.loader}>Loading...</div>}
        {response && (
          <div className={styles.response}>
            <h2>Response:</h2>
            <p>{response}</p>
          </div>
        )}
      </main>
    </div>
  );
}
