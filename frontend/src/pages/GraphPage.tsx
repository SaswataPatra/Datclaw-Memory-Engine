import { useState, useEffect } from 'react';
import { useAuth } from '../auth-context';

// TODO: Install react-flow-renderer
// npm install reactflow

// This is a placeholder - you'll need to:
// 1. npm install reactflow
// 2. Create an API endpoint to fetch graph data
// 3. Implement the visualization

export default function GraphPage() {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);

  return (
    <div className="h-screen bg-surface-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Knowledge Graph</h1>
        
        <div className="bg-surface-900 rounded-xl border border-surface-800 p-6">
          <p className="text-surface-400">
            Graph visualization coming soon!
          </p>
          
          <div className="mt-4 space-y-2 text-sm text-surface-500">
            <p>This will show:</p>
            <ul className="list-disc list-inside ml-4">
              <li>Your entities (people, places, concepts)</li>
              <li>Relationships between entities</li>
              <li>Memory connections</li>
              <li>Interactive exploration</li>
            </ul>
          </div>
          
          <div className="mt-6">
            <a
              href="http://localhost:8529"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-500 rounded-lg transition-colors"
            >
              Open ArangoDB Graph Viewer
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
