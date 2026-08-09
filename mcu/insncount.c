/* Minimal QEMU TCG plugin: count executed guest instructions.
 *
 * WHY THIS AND NOT DWT_CYCCNT. QEMU's mps2-an386 does not implement the DWT
 * cycle counter (it reads 0), so the register a real STM32F405 would use is
 * unavailable under emulation. Counting retired instructions is the honest
 * substitute: exact, deterministic, and reproducible. It is NOT cycles —
 * Cortex-M4 loads take ~2 cycles, taken branches ~2-3, and flash wait states
 * add more, so silicon cycles run above this count. Report it as INSTRUCTIONS.
 *
 * The guest brackets its region of interest by writing to a magic MMIO-ish
 * address; simpler here: we count everything and the guest runs exactly one
 * workload per invocation, so the delta between an empty-harness run and a
 * workload run isolates the workload.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static uint64_t insn_count;

static void vcpu_insn_exec(unsigned int cpu_index, void *udata) {
	insn_count++;
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
	size_t n = qemu_plugin_tb_n_insns(tb);
	for (size_t i = 0; i < n; i++) {
		struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
		qemu_plugin_register_vcpu_insn_exec_cb(insn, vcpu_insn_exec,
		                                       QEMU_PLUGIN_CB_NO_REGS, NULL);
	}
}

static void plugin_exit(qemu_plugin_id_t id, void *p) {
	g_autofree char *s = g_strdup_printf("INSNS\t%" PRIu64 "\n", insn_count);
	qemu_plugin_outs(s);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv) {
	qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
	qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
	return 0;
}
