export const meta = {
	name: 'paper-claims-audit',
	description: 'Extract every quantitative claim from a results/paper doc, verify each against the DB independently, then adjudicate what is safe to publish',
	whenToUse: 'Before a submission or camera-ready, or after a cohort lands and the tables were regenerated. Catches stale numbers, wrong-partition metrics, non-comparable cohorts, and best-of-N inflation that reading linearly will not.',
	phases: [
		{ title: 'Extract', detail: 'harvest atomic claims from the target doc(s)' },
		{ title: 'Verify', detail: 'one agent per claim, checked against actual DB values' },
		{ title: 'Refute', detail: 'adversarial second opinion on every claim that passed' },
		{ title: 'Adjudicate', detail: 'experiment-design rules supported / provisional / not supported' },
	],
}

// args: { docs?: string[], db?: string, focus?: string }
const DOCS = (args && args.docs) || ['docs/ids_results.md']
const DB = (args && args.db) || '/Volumes/20260401-WDBlack-SN850X-2TB/wnn/db/wnn.db'
const FOCUS = (args && args.focus) || 'all quantitative claims'

const CLAIMS_SCHEMA = {
	type: 'object',
	required: ['claims'],
	properties: {
		claims: {
			type: 'array',
			items: {
				type: 'object',
				required: ['id', 'text', 'location', 'metric'],
				properties: {
					id: { type: 'string' },
					text: { type: 'string', description: 'the claim as written, verbatim' },
					location: { type: 'string', description: 'file:line' },
					metric: { type: 'string', description: 'F1 / FPR / Acc / err° / stable° / size / speedup' },
					value: { type: 'string' },
					cohort: { type: 'string', description: 'cohort, flow, or run the number should come from' },
					partition: { type: 'string', description: 'as stated: train / during-search fold / held-out / report-seed / unstated' },
				},
			},
		},
	},
}

const VERDICT_SCHEMA = {
	type: 'object',
	required: ['id', 'status', 'reason'],
	properties: {
		id: { type: 'string' },
		status: { type: 'string', enum: ['supported', 'provisional', 'not_supported', 'unverifiable'] },
		reason: { type: 'string' },
		db_value: { type: 'string', description: 'what the DB actually says, or "not found"' },
		partition_ok: { type: 'boolean' },
		n: { type: 'string', description: 'number of runs behind the number' },
		issues: { type: 'array', items: { type: 'string' } },
	},
}

const REFUTE_SCHEMA = {
	type: 'object',
	required: ['id', 'refuted', 'reason'],
	properties: {
		id: { type: 'string' },
		refuted: { type: 'boolean' },
		reason: { type: 'string' },
	},
}

const RULES = `
Project measurement canon you MUST apply:
- Held-out only. IDS: val_cal held-out, never iterations.best_f1 (that is during-search k-fold = a LEAK).
  Controller: only the --report-seed HELD-OUT block; gen-line stable/err are optimistic and non-reproducible.
- Read ACTUAL values from the DB (sqlite3 -readonly ${DB}). Never compute, estimate, or infer a number.
  Use threshold_metadata (all 7 modes); best_genomes is incomplete.
- Cohorts differing in split family (random/temporal, 2-way vs _3way), protocol version, or scorer build
  are NOT comparable. Pre-20/06 GPU-scored controller numbers are bug-inflated — never citable.
- Watch for: base-rate artifacts (46M is 2.35% benign, so accuracy is nearly free — trust F1/FPR),
  best-of-N inflation (papers report best-of; N must be fixed and stated), n=1 rankings,
  and any metric quoted without its partition.
- If a number cannot be traced to a DB row, the status is "unverifiable" — NOT "supported".
`

phase('Extract')
const extracted = await parallel(DOCS.map(doc => () =>
	agent(
		`Read ${doc} in full. Extract EVERY atomic quantitative claim matching: ${FOCUS}.\n` +
		`An atomic claim is one number attached to one subject (e.g. "16b-Wb reaches F1 88.86 on UNSW-temporal held-out").\n` +
		`Split compound sentences into separate claims. Include the stated partition, or "unstated" if absent —\n` +
		`an unstated partition is itself a finding. Give each claim a stable id like "${doc.split('/').pop()}#1".\n` +
		`Do NOT verify anything here; just harvest faithfully and verbatim.`,
		{ label: `extract:${doc.split('/').pop()}`, phase: 'Extract', schema: CLAIMS_SCHEMA },
	),
)).then(rs => rs.filter(Boolean).flatMap(r => r.claims || []))

log(`extracted ${extracted.length} claims from ${DOCS.length} doc(s)`)
if (!extracted.length) return { error: 'no claims extracted — check the doc paths', docs: DOCS }

// Pipeline: each claim verifies, then (only if it survived) gets adversarially refuted.
// No barrier — a fast claim reaches Refute while a slow one is still in Verify.
const results = await pipeline(
	extracted,
	claim => agent(
		`Verify this claim against the database.\n\nCLAIM: ${claim.text}\n` +
		`LOCATION: ${claim.location}\nMETRIC: ${claim.metric}\nVALUE: ${claim.value || '(none parsed)'}\n` +
		`COHORT: ${claim.cohort || '(unstated)'}\nSTATED PARTITION: ${claim.partition || 'unstated'}\n\n` +
		`${RULES}\n` +
		`Query the DB for the actual value. Report what it says, whether the partition is the right one,\n` +
		`and how many runs stand behind it. Be exact; quote the row.`,
		{ label: `verify:${claim.id}`, phase: 'Verify', schema: VERDICT_SCHEMA, agentType: 'ids-security' },
	).then(v => ({ claim, verdict: v })),

	({ claim, verdict }) => {
		if (!verdict || verdict.status !== 'supported') return { claim, verdict, refutation: null }
		// Only survivors get attacked — refuting an already-failed claim is wasted work.
		return agent(
			`Try to REFUTE that this claim is safe to publish. Default to refuted=true if you are uncertain.\n\n` +
			`CLAIM: ${claim.text}\nVERIFIER SAID: ${verdict.reason} (db_value=${verdict.db_value}, n=${verdict.n})\n\n` +
			`${RULES}\n` +
			`Attack specifically: is this the held-out partition or a during-search number? Is N large enough to\n` +
			`support the comparison being made? Is the comparison cohort actually apples-to-apples? Is any gain a\n` +
			`base-rate artifact? Is it a best-of-N figure presented as typical?`,
			{ label: `refute:${claim.id}`, phase: 'Refute', schema: REFUTE_SCHEMA, agentType: 'experiment-design' },
		).then(r => ({ claim, verdict, refutation: r }))
	},
)

const rows = results.filter(Boolean)
const failed = rows.filter(r => r.verdict && r.verdict.status !== 'supported')
const refuted = rows.filter(r => r.refutation && r.refutation.refuted)
const clean = rows.filter(r => r.verdict && r.verdict.status === 'supported' && !(r.refutation && r.refutation.refuted))
log(`verified ${rows.length}: ${clean.length} clean, ${failed.length} failed verification, ${refuted.length} refuted on review`)

phase('Adjudicate')
const problems = [...failed, ...refuted]
const summary = await agent(
	`You are ruling on what is safe to publish. ${rows.length} claims were verified against the DB and the\n` +
	`survivors were adversarially reviewed.\n\n` +
	`PROBLEMS (${problems.length}):\n${JSON.stringify(problems.map(p => ({
		id: p.claim.id, loc: p.claim.location, text: p.claim.text,
		status: p.verdict && p.verdict.status, reason: p.verdict && p.verdict.reason,
		db: p.verdict && p.verdict.db_value, refuted: p.refutation && p.refutation.reason,
	})), null, 1)}\n\n` +
	`CLEAN: ${clean.length} claims passed both stages.\n\n` +
	`Produce a publication-readiness verdict. Group the problems by failure mode (leak, wrong partition,\n` +
	`non-comparable cohort, n too small, base-rate artifact, unverifiable). For each: the exact edit that\n` +
	`fixes it, or the smallest experiment that would settle it. Order by how badly it would hurt in review.\n` +
	`Do not invent numbers. If something needs a measurement that does not exist, say so plainly.`,
	{ label: 'adjudicate', phase: 'Adjudicate', agentType: 'experiment-design' },
)

return {
	docs: DOCS,
	total: rows.length,
	clean: clean.length,
	failed_verification: failed.length,
	refuted_on_review: refuted.length,
	problems: problems.map(p => ({ id: p.claim.id, location: p.claim.location, text: p.claim.text })),
	verdict: summary,
}
