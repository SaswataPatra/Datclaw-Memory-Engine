import type { AdapterInfo } from '../api';

/** Emoji for adapter card (by source_type or category/subtype). */
export function adapterEmoji(a: AdapterInfo): string {
  const st = a.source_type;
  if (st === 'chatgpt') return '💬';
  if (st === 'session_json') return '📋';
  if (st === 'text_file') return '📄';
  if (st === 'pdf_file') return '📕';
  if (st === 'text_paste') return '📝';
  if (a.category === 'file') return '📁';
  if (a.category === 'llm_chat') return '🔗';
  return '📥';
}

/** Tailwind classes for card border / hover ring. */
export function adapterAccentClass(a: AdapterInfo): string {
  const st = a.source_type;
  if (st === 'chatgpt') return 'border-emerald-700/50 hover:ring-emerald-500/30';
  if (st === 'pdf_file') return 'border-red-800/40 hover:ring-red-500/30';
  if (st === 'text_file' || st === 'text_paste') return 'border-surface-600 hover:ring-primary-500/30';
  if (st === 'session_json') return 'border-amber-800/40 hover:ring-amber-500/30';
  return 'border-surface-700 hover:ring-primary-500/20';
}

export function categoryLabel(category: string | null): string {
  if (!category) return 'Other';
  if (category === 'llm_chat') return 'LLM & chat';
  if (category === 'file') return 'Files & text';
  if (category === 'web') return 'Web';
  if (category === 'api') return 'API';
  return category;
}
