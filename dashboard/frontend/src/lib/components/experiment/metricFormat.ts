/**
 * Metric display formatters shared by the experiment detail components.
 * Extracted verbatim from routes/experiments/[id]/+page.svelte (P3 decomposition).
 */

import type { GenomeEvaluation, GenomeTier } from '$lib/types';

export function formatAcc(acc: number | null | undefined): string {
  if (acc === null || acc === undefined) return '—';
  return (acc * 100).toFixed(4) + '%';
}

export function formatAccShort(acc: number | null | undefined): string {
  if (acc === null || acc === undefined) return '—';
  return (acc * 100).toFixed(2) + '%';
}

export function formatF1(f1: number | null | undefined): string {
  if (f1 === null || f1 === undefined) return '—';
  return (f1 * 100).toFixed(2) + '%';
}

export function formatFPR(fpr: number | null | undefined): string {
  if (fpr === null || fpr === undefined) return '—';
  return (fpr * 100).toFixed(3) + '%';
}

export function formatRole(role: string): string {
  switch (role) {
    case 'elite': return '🏆 Elite';
    case 'top_k': return '🏆 Top-K';
    case 'offspring': return '📌 Offspring';
    case 'init': return '🌱 Init';
    case 'neighbor': return '🔗 Neighbor';
    case 'current': return '⭐ Current';
    default: return role;
  }
}

export function parseTier(g: GenomeEvaluation): { neurons: string; bits: string } {
  if (!g.tiers_json) return { neurons: '—', bits: '—' };
  try {
    const t: GenomeTier[] = JSON.parse(g.tiers_json);
    if (t.length === 0) return { neurons: '—', bits: '—' };
    if (t.length === 1) return { neurons: String(t[0].neurons), bits: String(t[0].bits) };
    // Multiple tiers: show per-tier values joined with +
    return {
      neurons: t.map(tier => String(tier.neurons)).join('+'),
      bits: t.map(tier => String(tier.bits)).join('+'),
    };
  } catch { return { neurons: '—', bits: '—' }; }
}
