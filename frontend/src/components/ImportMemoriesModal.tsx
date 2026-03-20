import { useEffect, useMemo, useState } from 'react';
import type { AdapterInfo, IngestResponse } from '../api';
import { ApiError, user as userApi } from '../api';
import FileDropZone from './FileDropZone';
import ProgressStages from './ProgressStages';
import { adapterAccentClass, adapterEmoji, categoryLabel } from '../utils/adapterIcons';

type Step = 'select' | 'input';

type Props = {
  onClose: () => void;
  onSuccess?: (result: IngestResponse) => void;
};

function groupByCategory(adapters: AdapterInfo[]): Map<string, AdapterInfo[]> {
  const m = new Map<string, AdapterInfo[]>();
  for (const a of adapters) {
    if (!a.enabled) continue;
    const key = categoryLabel(a.category);
    if (!m.has(key)) m.set(key, []);
    m.get(key)!.push(a);
  }
  return m;
}

export default function ImportMemoriesModal({ onClose, onSuccess }: Props) {
  const [step, setStep] = useState<Step>('select');
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(true);

  const [selected, setSelected] = useState<AdapterInfo | null>(null);
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [pickedFile, setPickedFile] = useState<File | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingList(true);
      setLoadError(null);
      try {
        const res = await userApi.listIngestAdapters();
        if (!cancelled) setAdapters(res.adapters.filter((a) => a.enabled));
      } catch (e) {
        if (!cancelled)
          setLoadError(e instanceof ApiError ? e.message : 'Could not load adapters');
      } finally {
        if (!cancelled) setLoadingList(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const grouped = useMemo(() => groupByCategory(adapters), [adapters]);

  function pickAdapter(a: AdapterInfo) {
    setSelected(a);
    setUrl('');
    setText('');
    setPickedFile(null);
    setResult(null);
    setError(null);
    setStep('input');
  }

  function goBack() {
    setStep('select');
    setSelected(null);
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!selected) return;
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      if (selected.input_mode === 'file') {
        const file = pickedFile;
        if (!file) {
          setError('Choose a file to upload.');
          return;
        }
        const r = await userApi.ingestUpload(file);
        setResult(r);
        onSuccess?.(r);
        return;
      }

      const src =
        selected.input_mode === 'url' ? url.trim() : text.trim();
      if (!src) {
        setError(
          selected.input_mode === 'url'
            ? 'Enter a URL.'
            : 'Paste content (text or JSON).'
        );
        return;
      }

      const category = selected.category || 'llm_chat';
      const subtype = selected.subtype || 'chatgpt';
      const r = await userApi.ingestMemoriesSpec({
        source: src,
        category,
        subtype,
      });
      setResult(r);
      onSuccess?.(r);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Import failed');
    } finally {
      setSubmitting(false);
    }
  }

  const fileAccept =
    selected?.source_type === 'pdf_file'
      ? '.pdf,application/pdf'
      : '.txt,.md,.markdown,.rst,.text,text/plain,text/markdown';

  const fileHint =
    selected?.source_type === 'pdf_file'
      ? 'PDF documents'
      : 'Plain text, Markdown, README (.txt, .md, …)';

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-surface-900 border border-surface-700 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-surface-800 shrink-0">
          <div>
            <h2 className="text-xl font-semibold text-white">Import memories</h2>
            <p className="text-surface-500 text-xs mt-0.5">
              Choose a source, then provide a URL, file, or pasted content.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-surface-400 hover:text-white transition-colors p-1"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="overflow-y-auto flex-1 px-6 py-4">
          {loadError ? (
            <p className="text-red-400 text-sm">{loadError}</p>
          ) : loadingList ? (
            <p className="text-surface-400 text-sm">Loading sources…</p>
          ) : step === 'select' ? (
            <div className="space-y-6">
              {Array.from(grouped.entries()).map(([cat, items]) => (
                <div key={cat}>
                  <h3 className="text-surface-500 text-xs font-semibold uppercase tracking-wide mb-2">
                    {cat}
                  </h3>
                  <div className="grid sm:grid-cols-2 gap-3">
                    {items.map((a) => (
                      <button
                        key={a.source_type}
                        type="button"
                        onClick={() => pickAdapter(a)}
                        className={`text-left rounded-xl border bg-surface-950/50 px-4 py-3 transition-all hover:bg-surface-800/80 ${adapterAccentClass(a)}`}
                      >
                        <div className="flex items-start gap-2">
                          <span className="text-xl shrink-0">{adapterEmoji(a)}</span>
                          <div className="min-w-0">
                            <div className="text-white text-sm font-medium truncate">{a.name}</div>
                            <div className="text-surface-500 text-xs mt-1 line-clamp-2">{a.description}</div>
                            <div className="text-surface-600 text-[10px] mt-1 font-mono">
                              {a.source_type} · {a.input_mode}
                            </div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : selected ? (
            <div className="space-y-4">
              <button
                type="button"
                onClick={goBack}
                className="text-primary-400 text-sm hover:text-primary-300"
              >
                ← All sources
              </button>

              <div className="flex items-center gap-2">
                <span className="text-2xl">{adapterEmoji(selected)}</span>
                <div>
                  <div className="text-white font-medium">{selected.name}</div>
                  <div className="text-surface-500 text-xs">{selected.description}</div>
                </div>
              </div>

              {selected.input_mode === 'file' && (
                <div>
                  <FileDropZone
                    disabled={submitting}
                    accept={fileAccept}
                    hint={fileHint}
                    onFile={(f) => {
                      setPickedFile(f);
                      setError(null);
                    }}
                  />
                  {pickedFile ? (
                    <p className="text-surface-400 text-xs mt-2 truncate">Selected: {pickedFile.name}</p>
                  ) : null}
                </div>
              )}

              {selected.input_mode === 'url' && (
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://…"
                  disabled={submitting}
                  className="w-full px-4 py-3 rounded-lg bg-surface-800 border border-surface-700/50 text-white placeholder-surface-500 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/40"
                />
              )}

              {selected.input_mode === 'text' && (
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={
                    selected.source_type === 'session_json'
                      ? 'Paste JSON (MemoryBench / unified session format)…'
                      : 'Paste text to store as a memory…'
                  }
                  disabled={submitting}
                  rows={10}
                  className="w-full px-4 py-3 rounded-lg bg-surface-800 border border-surface-700/50 text-white placeholder-surface-500 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500/40 font-mono text-xs"
                />
              )}

              <ProgressStages loading={submitting && !result} stages={result?.pipeline_stages} />

              {error ? (
                <div className="p-3 rounded-lg bg-red-900/20 border border-red-800/40 text-red-300 text-sm">
                  {error}
                </div>
              ) : null}

              {result ? (
                <div
                  className={`p-4 rounded-lg text-sm space-y-2 ${
                    result.status === 'success'
                      ? 'bg-green-900/20 border border-green-800/40 text-green-200'
                      : 'bg-amber-900/20 border border-amber-800/40 text-amber-200'
                  }`}
                >
                  <p>
                    {result.status === 'success' ? '✅' : '⚠️'} {result.memories_created} memories created
                    from {result.chunks_parsed} chunks (session {result.session_id})
                  </p>
                  {result.errors?.length ? (
                    <ul className="text-xs list-disc pl-4 text-amber-300/90">
                      {result.errors.slice(0, 5).map((e, i) => (
                        <li key={i}>{e}</li>
                      ))}
                    </ul>
                  ) : null}
                  {result.elapsed_seconds != null ? (
                    <p className="text-surface-400 text-xs">
                      ⏱ {result.elapsed_seconds}s total
                      {result.seconds_per_memory != null
                        ? ` (~${result.seconds_per_memory}s / memory)`
                        : ''}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        {step === 'input' && selected ? (
          <div className="flex gap-3 px-6 py-4 border-t border-surface-800 shrink-0">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="flex-1 px-4 py-2.5 rounded-lg bg-surface-800 hover:bg-surface-700 text-surface-300 font-medium transition-colors disabled:opacity-50"
            >
              Close
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting}
              className="flex-1 px-4 py-2.5 rounded-lg bg-primary-600 hover:bg-primary-500 text-white font-medium transition-colors disabled:opacity-50"
            >
              {submitting ? 'Importing…' : 'Import'}
            </button>
          </div>
        ) : (
          <div className="px-6 py-4 border-t border-surface-800 shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="w-full px-4 py-2.5 rounded-lg bg-surface-800 hover:bg-surface-700 text-surface-300 font-medium"
            >
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
