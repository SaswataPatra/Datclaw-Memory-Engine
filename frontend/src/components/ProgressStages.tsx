import type { PipelineStage } from '../api';

type Props = {
  stages: PipelineStage[] | null | undefined;
  loading?: boolean;
};

export default function ProgressStages({ stages, loading }: Props) {
  if (loading && !stages?.length) {
    return (
      <div className="flex items-center gap-2 text-surface-400 text-sm py-2">
        <span className="inline-block w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        Working…
      </div>
    );
  }

  if (!stages?.length) return null;

  return (
    <ul className="text-xs space-y-1.5 py-2 border-t border-surface-800 mt-2">
      {stages.map((s, i) => (
        <li key={`${s.name}-${i}`} className="flex items-center justify-between gap-2 text-surface-400">
          <span className="flex items-center gap-1.5">
            <span>{s.ok ? '✓' : '✗'}</span>
            <span className="capitalize text-surface-300">{s.name.replace(/_/g, ' ')}</span>
          </span>
          <span className="text-surface-500 tabular-nums">{s.duration_ms?.toFixed?.(1) ?? '—'} ms</span>
          {s.detail ? <span className="text-surface-500 truncate max-w-[180px]">{s.detail}</span> : null}
        </li>
      ))}
    </ul>
  );
}
