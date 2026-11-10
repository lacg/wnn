/* QEMU TCG plugin: retired-instruction count PER REGION (CTRL-51, 10/11/2026).
 *
 * insncount.c counts a whole run, so a per-step cost is (I_N - I_0)/N: a MEAN. A
 * controller whose cost varies step to step (the MPCOF box QP) needs every step's
 * count to report a worst case. The guest brackets each region with two marker
 * instructions that compiled C never emits:
 *     mov r11, r11   (0x46DB)  region start
 *     mov r10, r10   (0x46D2)  region end
 * (bench_ctrl51.c MARK_BEGIN / MARK_END). At each END the plugin appends
 * (instructions retired since the matching START, excluding the START marker,
 * including the END marker) to a list, written one count per line to the file
 * named by the plugin argument out=<path> at exit. Counts are exact and
 * deterministic; they are INSTRUCTIONS, not cycles (see insncount.c).
 *
 * Each line is "insns soft libm vfp": besides the instruction count, how many
 * calls the region made into soft-double primitives (entry addresses passed as
 * soft=0x...; e.g. __aeabi_dadd/dmul/ddiv/dcmp*, conversions), into libm (lm=0x...;
 * atan2, asin, sqrt, ...) and how many VFP data-processing instructions
 * (hardware single-precision arithmetic: Thumb-2 0xEExx with coprocessor 10/11)
 * it retired. The function addresses come from the ELF's symbol table
 * (mcu/run_bench_ctrl51.sh).
 *
 * Build: as insncount.c (mcu/README.md), -o mcu/insnmark.dylib.
 * Run:   qemu-system-arm ... -plugin mcu/insnmark.dylib,out=/tmp/regions.txt -d plugin */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

enum { C_INSN, C_SOFT, C_LIBM, C_VFP, N_C };
static uint64_t tot[N_C], start[N_C];
static int in_region;
static uint64_t *counts;          /* N_C per region */
static size_t n_counts, cap_counts;
#define MAX_FN 256
static uint64_t soft_fn[MAX_FN], libm_fn[MAX_FN];
static int n_soft, n_libm;
static char out_path[512] = "insnmark_regions.txt";

static void vcpu_insn_exec(unsigned int cpu_index, void *udata) {
	tot[C_INSN]++;
}

/* udata = which counter (C_SOFT / C_LIBM / C_VFP) this instruction bumps. */
static void vcpu_class_exec(unsigned int cpu_index, void *udata) {
	tot[(uintptr_t)udata]++;
}

static void push_counts(void) {
	if (n_counts == cap_counts) {
		cap_counts = cap_counts ? 2 * cap_counts : 65536;
		counts = realloc(counts, cap_counts * N_C * sizeof *counts);
	}
	for (int c = 0; c < N_C; c++) counts[n_counts * N_C + c] = tot[c] - start[c];
	n_counts++;
}

static void mark_begin(unsigned int cpu_index, void *udata) {
	memcpy(start, tot, sizeof start);   /* the START marker itself already counted */
	in_region = 1;
}

static void mark_end(unsigned int cpu_index, void *udata) {
	if (in_region) push_counts();
	in_region = 0;
}

static int listed(const uint64_t *fn, int n, uint64_t va) {
	for (int i = 0; i < n; i++)
		if (fn[i] == va) return 1;
	return 0;
}

/* VFP data-processing: Thumb-2 first halfword 0xEExx, coprocessor (bits 11:8 of the
 * second halfword) 10 or 11. Loads/stores (0xEDxx) and moves to core are excluded
 * only in so far as they are not 0xEE-class; vmov core<->VFP is 0xEE-class and is
 * counted — read the column as "VFP arithmetic + register moves". */
static int is_vfp_dp(const uint8_t *b, size_t sz) {
	if (sz != 4) return 0;
	uint16_t hw1 = (uint16_t)(b[0] | (b[1] << 8)), hw2 = (uint16_t)(b[2] | (b[3] << 8));
	int cp = (hw2 >> 8) & 0xF;
	return (hw1 >> 8) == 0xEE && (cp == 10 || cp == 11);
}

static void classify(struct qemu_plugin_insn *insn) {
	uint64_t va = qemu_plugin_insn_vaddr(insn);
	uint8_t b[4] = {0, 0, 0, 0};
	size_t sz = qemu_plugin_insn_size(insn);
	qemu_plugin_insn_data(insn, b, sz < 4 ? sz : 4);
	uintptr_t c = listed(soft_fn, n_soft, va) ? C_SOFT : listed(libm_fn, n_libm, va) ? C_LIBM
	              : is_vfp_dp(b, sz) ? C_VFP : N_C;
	if (c != N_C)
		qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_class_exec, QEMU_PLUGIN_CB_NO_REGS, (void *)c);
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
	size_t n = qemu_plugin_tb_n_insns(tb);
	for (size_t i = 0; i < n; i++) {
		struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
		qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn_exec, QEMU_PLUGIN_CB_NO_REGS, NULL);
		classify(insn);
		uint8_t b[2] = {0, 0};
		if (qemu_plugin_insn_size(insn) != 2 || qemu_plugin_insn_data(insn, b, 2) != 2) continue;
		uint16_t op = (uint16_t)(b[0] | (b[1] << 8));
		/* Registered AFTER the counter, so it runs after this insn is counted. */
		if (op == 0x46DB)
			qemu_plugin_register_vcpu_insn_exec_cb(insn, mark_begin, QEMU_PLUGIN_CB_NO_REGS, NULL);
		else if (op == 0x46D2)
			qemu_plugin_register_vcpu_insn_exec_cb(insn, mark_end, QEMU_PLUGIN_CB_NO_REGS, NULL);
	}
}

static void plugin_exit(qemu_plugin_id_t id, void *p) {
	FILE *f = fopen(out_path, "w");
	for (size_t i = 0; f && i < n_counts; i++) {
		const uint64_t *r = &counts[i * N_C];
		fprintf(f, "%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", r[0], r[1], r[2], r[3]);
	}
	if (f) fclose(f);
	g_autofree char *s = g_strdup_printf("INSNS\t%" PRIu64 "\nREGIONS\t%zu\n", tot[C_INSN], n_counts);
	qemu_plugin_outs(s);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                                           int argc, char **argv) {
	for (int i = 0; i < argc; i++)
	{
		if (strncmp(argv[i], "out=", 4) == 0) snprintf(out_path, sizeof out_path, "%s", argv[i] + 4);
		if (strncmp(argv[i], "soft=", 5) == 0 && n_soft < MAX_FN) soft_fn[n_soft++] = strtoull(argv[i] + 5, 0, 16);
		if (strncmp(argv[i], "lm=", 3) == 0 && n_libm < MAX_FN) libm_fn[n_libm++] = strtoull(argv[i] + 3, 0, 16);
	}
	qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
	qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
	return 0;
}
