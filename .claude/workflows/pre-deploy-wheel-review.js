export const meta = {
	name: 'pre-deploy-wheel-review',
	description: 'Multi-dimension review of accelerator changes before a wheel is built and swapped, with adversarial verification and a go/no-go verdict',
	whenToUse: 'Before `maturin develop --release` on a wheel that will be deployed — especially the WORKER wheel (ram_accelerator), which can only be swapped at worker-idle and takes the live IDS cohort down with it if it is wrong. Also worth running before a ram_core change, since that rebuilds both wheels.',
	phases: [
		{ title: 'Scope', detail: 'determine the diff and which wheels it forces to rebuild' },
		{ title: 'Review', detail: 'parallel dimension reviews over the changed code' },
		{ title: 'Verify', detail: 'adversarially refute each finding before it is reported' },
		{ title: 'Verdict', detail: 'go / no-go with the deploy sequence' },
	],
}

// args: { base?: string, wheel?: 'worker'|'controller'|'auto' }
const BASE = (args && args.base) || 'main'
const WHEEL = (args && args.wheel) || 'auto'

const ACCEL = 'src/wnn/ram/strategies/accelerator'

const SCOPE_SCHEMA = {
	type: 'object',
	required: ['files', 'wheels', 'summary'],
	properties: {
		files: { type: 'array', items: { type: 'string' } },
		wheels: { type: 'array', items: { type: 'string', enum: ['ram_accelerator', 'ram_controller', 'none'] } },
		touches_core: { type: 'boolean', description: 'true if ram_core changed — that rebuilds BOTH wheels' },
		touches_shaders: { type: 'boolean' },
		touches_abi: { type: 'boolean', description: 'PyO3 surface, ABI_VERSION, or a Python-side constructor signature' },
		summary: { type: 'string' },
	},
}

const FINDINGS_SCHEMA = {
	type: 'object',
	required: ['findings'],
	properties: {
		findings: {
			type: 'array',
			items: {
				type: 'object',
				required: ['title', 'file', 'severity', 'detail', 'failure_scenario'],
				properties: {
					title: { type: 'string' },
					file: { type: 'string' },
					line: { type: 'number' },
					severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
					detail: { type: 'string' },
					failure_scenario: { type: 'string', description: 'concrete inputs/state -> wrong output, crash, or OOM' },
				},
			},
		},
	},
}

const REFUTE_SCHEMA = {
	type: 'object',
	required: ['refuted', 'reason'],
	properties: {
		refuted: { type: 'boolean' },
		reason: { type: 'string' },
		severity_adjusted: { type: 'string', enum: ['blocker', 'major', 'minor', 'none'] },
	},
}

const CONTEXT = `
Repo facts you MUST hold while reviewing (${ACCEL}):
- Cargo WORKSPACE of three crates. ram_core (rlib: neuron_memory, packed_bits, sparse_memory, metal_sparse,
  cancel) is SHARED substrate — a change there rebuilds BOTH cdylibs. ram_accelerator = IDS/LM worker.
  ram_controller = drone controller hot path.
- The WORKER wheel can only be swapped at worker-idle (scripts/worker_swap.py); the CONTROLLER wheel
  installs anytime because the worker never imports it.
- Source/wheel skew is a real outage mode: Python + Rust constructor changes must land ATOMICALLY at
  driver-idle, or the next spawned cell crashes.
- QUAD_WEIGHTED (mode 2) is the default and must never silently become TERNARY. CPU cell semantics go
  through neuron_memory::cell_to_weight() ONLY — a hardcoded FALSE=>0.0/TRUE=>1.0 match shipped the
  inverted-QUAD multistage bug. Metal gets semantics from shaders/common.metal; never per-shader copies.
- CPU and GPU paths must agree bit-exactly (cargo test cpu_fallback_matches_gpu).
- Memory is SPARSE (used addresses only) — never dense n x 2^bits sizing.
`

phase('Scope')
const scope = await agent(
	`Determine the review scope for an accelerator change.\n` +
	`Run: git diff --stat ${BASE}...HEAD -- ${ACCEL} ; and git diff ${BASE}...HEAD --name-only.\n` +
	`Also check for uncommitted work: git status --short.\n\n` +
	`${CONTEXT}\n` +
	`Decide which wheel(s) this forces to rebuild (requested: ${WHEEL}). Flag whether ram_core, any Metal\n` +
	`shader, or the PyO3/ABI surface is touched. List the changed files under the accelerator.`,
	{ label: 'scope', phase: 'Scope', schema: SCOPE_SCHEMA },
)

if (!scope || !scope.files || !scope.files.length) {
	log('no accelerator changes found — nothing to review')
	return { verdict: 'NO CHANGES', base: BASE, scope }
}
log(`${scope.files.length} changed file(s); wheels=${(scope.wheels || []).join(',')}; core=${scope.touches_core}; shaders=${scope.touches_shaders}; abi=${scope.touches_abi}`)

const FILES = scope.files.join('\n')

// Dimensions are chosen by what the diff actually touches — reviewing shaders when no
// shader changed just burns tokens and dilutes the report.
const DIMENSIONS = [
	{
		key: 'correctness', agentType: 'rust-code',
		prompt: `Review these accelerator changes for CORRECTNESS bugs: indexing/bounds, integer width and\n` +
			`truncation (the u32 truncation bug bit n>=100,b>32 before), off-by-one in address computation,\n` +
			`rayon data races, unwrap/panic paths reachable from Python, and error handling that silently\n` +
			`swallows a failure.`,
	},
	{
		key: 'cell-semantics', agentType: 'wnn-specialist',
		prompt: `Review for MEMORY-MODE and cell-semantics violations: any hardcoded cell->weight mapping instead\n` +
			`of neuron_memory::cell_to_weight(), any path that defaults to TERNARY instead of QUAD_WEIGHTED,\n` +
			`empty-value handling, MSB-first bit order, and dense-vs-sparse sizing assumptions.`,
	},
	{
		key: 'memory-safety', agentType: 'rust-code',
		prompt: `Review for MEMORY/OOM risk: allocation sized by neurons x 2^bits or per-genome pools that scale\n` +
			`with dataset size (calculate_pool_size hardcoding 3000/neuron once produced a 63GB heap), buffers\n` +
			`held across batches, and anything that grows unbounded with cell count. This box runs the IDS\n` +
			`worker and a controller simultaneously — a regression here takes down live work.`,
	},
	...(scope.touches_shaders ? [{
		key: 'gpu-parity', agentType: 'rust-code',
		prompt: `Review the Metal shader changes for CPU/GPU PARITY: semantics must come from common.metal\n` +
			`(WNN_QUAD_WEIGHTS, wnn_compute_address*, wnn_cell_weight) and never be re-implemented per shader.\n` +
			`Check threadgroup sizing, masking of padded neurons in coalesced groups, and that\n` +
			`cpu_fallback_matches_gpu would still hold.`,
	}] : []),
	...(scope.touches_abi ? [{
		key: 'abi-skew', agentType: 'architect',
		prompt: `Review the PyO3/ABI surface for DEPLOY SKEW: is ABI_VERSION bumped when the surface changed? Do\n` +
			`the Python facades (wnn/accel.py for the worker, wnn/control/_accel.py for the controller) still\n` +
			`match? Would a stale wheel fail LOUDLY rather than silently misbehave? Are Python and Rust\n` +
			`constructor changes atomic, or could a half-deployed state crash the next spawned cell?`,
	}] : []),
]

log(`running ${DIMENSIONS.length} review dimension(s)`)

const reviewed = await pipeline(
	DIMENSIONS,
	d => agent(
		`${d.prompt}\n\nCHANGED FILES:\n${FILES}\n\nDIFF BASE: ${BASE}\n\n${CONTEXT}\n` +
		`Read the actual diff (git diff ${BASE}...HEAD -- <file>) and the surrounding code before judging.\n` +
		`Report ONLY defects you can name a concrete failure for. No style notes, no speculation.`,
		{ label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, agentType: d.agentType },
	),
	(review, d) => {
		const fs = (review && review.findings) || []
		if (!fs.length) return []
		return parallel(fs.map(f => () =>
			agent(
				`Try to REFUTE this finding. Default to refuted=true if you are uncertain — a false blocker that\n` +
				`delays a deploy costs real experiment time.\n\n` +
				`TITLE: ${f.title}\nFILE: ${f.file}${f.line ? ':' + f.line : ''}\nSEVERITY: ${f.severity}\n` +
				`DETAIL: ${f.detail}\nCLAIMED FAILURE: ${f.failure_scenario}\n\n${CONTEXT}\n` +
				`Read the code. Does the failure actually reproduce, or is it already prevented by a caller, a\n` +
				`guard, a type invariant, or an existing test? Confirm or adjust the severity.`,
				{ label: `verify:${d.key}:${f.file.split('/').pop()}`, phase: 'Verify', schema: REFUTE_SCHEMA, agentType: 'quality-assurance' },
			).then(v => ({ ...f, dimension: d.key, verdict: v })),
		))
	},
)

const all = reviewed.flat().filter(Boolean)
const confirmed = all.filter(f => f.verdict && !f.verdict.refuted)
const blockers = confirmed.filter(f => (f.verdict.severity_adjusted || f.severity) === 'blocker')
log(`${all.length} raw findings -> ${confirmed.length} confirmed, ${blockers.length} blocker(s)`)

phase('Verdict')
const verdict = await agent(
	`Give a GO / NO-GO verdict on building and deploying this wheel.\n\n` +
	`SCOPE: ${scope.summary}\nWHEELS: ${(scope.wheels || []).join(', ')} | core=${scope.touches_core} ` +
	`shaders=${scope.touches_shaders} abi=${scope.touches_abi}\n\n` +
	`CONFIRMED FINDINGS (${confirmed.length}):\n${JSON.stringify(confirmed.map(f => ({
		dim: f.dimension, title: f.title, file: f.file, line: f.line,
		severity: (f.verdict.severity_adjusted || f.severity), failure: f.failure_scenario,
	})), null, 1)}\n\n${CONTEXT}\n` +
	`Rule: any confirmed blocker is NO-GO. If GO, give the exact deploy sequence — which wheel(s) to build,\n` +
	`the exact maturin command, whether a worker swap at idle is required (worker wheel or ram_core) or\n` +
	`whether it installs anytime (controller-only), which tests to run first, and the verification command.\n` +
	`Remember: IDS is priority — a worker swap must wait for worker-idle, never interrupt a running flow.`,
	{ label: 'verdict', phase: 'Verdict', agentType: 'judge-orchestrator' },
)

return {
	base: BASE,
	files: scope.files.length,
	wheels: scope.wheels,
	raw_findings: all.length,
	confirmed: confirmed.length,
	blockers: blockers.length,
	go: blockers.length === 0,
	findings: confirmed.map(f => ({
		dimension: f.dimension, title: f.title, file: f.file, line: f.line,
		severity: (f.verdict.severity_adjusted || f.severity),
	})),
	verdict,
}
