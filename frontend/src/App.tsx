import { useState } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const handleSearch = async () => {

    setResults([]); 

    if (!query.trim()) return;

    const res = await fetch(`http://localhost:8000/search?q=${query}`);
    const data = await res.json();
    
    setResults(data.results);
  };

  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4 place-self-center">Tübi-Search</h1>
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          className="border p-2 flex-grow rounded text-black"
          placeholder="Search..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch} className="bg-slate-900 text-white px-4 py-2 rounded">
          Search
        </button>
      </div>
      <ul>
        {results.map((r, i) => (
          <li key={i} className="border-b py-2 text-white">{r}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;

