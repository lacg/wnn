/* Freestanding Cortex-M4 benchmark support: vector table, semihosting output,
 * and the DWT cycle counter. No libc — nothing but the code under test runs
 * inside the measured region.
 *
 * Target: QEMU mps2-an386 (Cortex-M4). The DWT_CYCCNT read here is the SAME
 * register a real Crazyflie (STM32F405) would use, so the harness transfers to
 * hardware unchanged — but under emulation the count is instruction-paced, not
 * silicon-accurate. Every number this produces must be labelled EMULATED.
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>

#define DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define DEMCR      (*(volatile uint32_t *)0xE000EDFCu)

/* CP10/CP11 full access — the FPU is OFF out of reset; touching a VFP register
 * before this faults. Required on real M4F silicon too, not a QEMU quirk. */
static inline void fpu_enable(void) {
	*(volatile uint32_t *)0xE000ED88u |= (0xFu << 20);
	__asm__ volatile("dsb; isb");
}
static inline void cyccnt_init(void) {
	DEMCR |= (1u << 24);      /* TRCENA */
	DWT_CYCCNT = 0u;
	DWT_CTRL |= 1u;           /* CYCCNTENA */
}
static inline uint32_t cyccnt(void) { return DWT_CYCCNT; }

/* ARM semihosting: SYS_WRITE0 (0x04) writes a NUL-terminated string.
 * r0 must be "+r": the call RETURNS a value in r0, so declaring it input-only
 * lets GCC believe the operation code survives and the next call passes garbage. */
static void sh_write0(const char *s) {
	register int op __asm__("r0") = 0x04;
	register const char *p __asm__("r1") = s;
	__asm__ volatile("bkpt 0xAB" : "+r"(op) : "r"(p) : "memory");
}
static void sh_exit(void) {
	register int op __asm__("r0") = 0x18;
	register int c __asm__("r1") = 0x20026;
	__asm__ volatile("bkpt 0xAB" : "+r"(op) : "r"(c) : "memory");
}

static void put_str(const char *s) { sh_write0(s); }

static void put_u32(uint32_t v) {
	char b[12];
	int i = 11;
	b[i--] = 0;
	if (!v) b[i--] = '0';
	while (v) { b[i--] = (char)('0' + (v % 10u)); v /= 10u; }
	sh_write0(&b[i + 1]);
}

/* "label<TAB>value\n" — one row per measurement, easy to parse. */
static void report(const char *label, uint32_t v) {
	put_str(label); put_str("\t"); put_u32(v); put_str("\n");
}

/* Minimal startup. QEMU loads the ELF and honours the vector table. */
extern uint32_t _estack;
int main(void);
__attribute__((naked, noreturn)) static void Reset_Handler(void) {
	__asm__ volatile("bl main\n b .");
}
__attribute__((section(".isr_vector"), used))
static void *const vectors[] = { (void *)&_estack, (void *)Reset_Handler };

#endif /* BENCH_H */
