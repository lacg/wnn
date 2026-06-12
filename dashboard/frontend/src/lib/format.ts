/**
 * Shared display formatters (null/undefined → '—').
 */

export function formatPercent(x: number | null | undefined): string {
  if (x === null || x === undefined) return '—';
  return (x * 100).toFixed(2) + '%';
}

export function formatCE(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—';
  return x.toFixed(4);
}

/** Elapsed time between two timestamps; open-ended (end=null) means "until now". */
export function formatDuration(start: string | null, end: string | null): string {
  if (!start) return '—';
  const startDate = new Date(start);
  const endDate = end ? new Date(end) : new Date();
  const seconds = Math.max(0, Math.floor((endDate.getTime() - startDate.getTime()) / 1000));

  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}
