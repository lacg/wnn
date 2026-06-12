/**
 * Shared status → color mapping (P2, 12/06/2026).
 *
 * Was duplicated per page (flows, experiments, and inline in the detail
 * pages); the shared copy also depends on --text-tertiary, which app.css
 * now actually defines.
 */
export function getStatusColor(status: string): string {
	switch (status) {
		case 'running': return 'var(--accent-blue)';
		case 'completed': return 'var(--accent-green)';
		case 'failed': return 'var(--accent-red)';
		case 'cancelled': return 'var(--text-tertiary)';
		default: return 'var(--text-secondary)';
	}
}
