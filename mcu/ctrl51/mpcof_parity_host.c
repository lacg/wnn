/* CTRL-51 host parity: replay the Rust MPCOF teacher's recorded episodes
 * (ctrl51_export.rs traj_*.bin) through the C port (mcu/mpcof_cascade.c) and
 * compare, every tick, the held setpoint [p, q, r] and the plan's hidden inputs
 * (CtbrMpcOf::diag_hidden: measured ua, shadow I, shadow Fh, d-hat).
 * Prints one summary block; exit 1 on any double-build mismatch (bit equality).
 *   cc -O2 -std=c11 -ffp-contract=off -I mcu mcu/ctrl51/mpcof_parity_host.c \
 *      mcu/mpcof_cascade.c mcu/cascade_dhat.c mcu/wnn_ratefw.c -lm -o parity
 *   ./parity traj_nominal.bin [-v] [-d decisions.txt]
 * -d writes one line per decision tick: "qp_axes solves fallbacks" (the bench
 * summary joins it to the M4 per-call instruction counts, same decision order). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpcof_cascade.h"

typedef struct
{
	float gyro[3], applied[4], q[4], tgt[3];
	double label[3], hidden[12];
} tick_rec_t;

typedef struct
{
	long ticks, decisions, sp_mismatch, hid_mismatch;
	double sp_max_abs, hid_max_rel;
	long qp_decisions, qp_axes, solves, fallbacks, failed, max_solves;
} summary_t;

static int read_rec(FILE *f, tick_rec_t *r)
{
	return fread(r->gyro, 4, 14, f) == 14 && fread(r->label, 8, 15, f) == 15;
}

/* The C side of CtbrMpcOf::diag_hidden, read after label(). */
static void c_hidden(const mpcof_t *m, double h[12])
{
	cdhat_real ua[3] = {0, 0, 0};
	cdhat_measured_force(&m->obs, ua);
	for (int a = 0; a < 3; a++)
	{
		h[a] = (double)ua[a];
		h[3 + a] = m->shadow.ax[a].integ;
		h[6 + a] = m->shadow.held[a];
		h[9 + a] = (double)m->obs.dhat[a];
	}
}

static void compare(summary_t *s, const tick_rec_t *r, const double sp[3], const double h[12], long ep, long t, int verbose)
{
	int bad_sp = 0, bad_h = 0;
	for (int a = 0; a < 3; a++)
	{
		double d = fabs(sp[a] - r->label[a]);
		s->sp_max_abs = d > s->sp_max_abs ? d : s->sp_max_abs;
		bad_sp |= memcmp(&sp[a], &r->label[a], 8) != 0;
	}
	for (int i = 0; i < 12; i++)
	{
		double d = fabs(h[i] - r->hidden[i]) / (1e-12 + fabs(r->hidden[i]));
		s->hid_max_rel = d > s->hid_max_rel ? d : s->hid_max_rel;
		bad_h |= memcmp(&h[i], &r->hidden[i], 8) != 0;
	}
	if (bad_sp && verbose && s->sp_mismatch < 5)
		printf("  mismatch ep %ld tick %ld: c [%.17g %.17g %.17g] rust [%.17g %.17g %.17g]\n", ep, t,
		       sp[0], sp[1], sp[2], r->label[0], r->label[1], r->label[2]);
	s->sp_mismatch += bad_sp;
	s->hid_mismatch += bad_h;
}

static void count_decision(summary_t *s, const mpcof_stats_t *st)
{
	s->decisions++;
	s->qp_decisions += st->qp_axes > 0;
	s->qp_axes += st->qp_axes;
	s->solves += st->solves;
	s->fallbacks += st->fallbacks;
	s->failed += st->failed;
	s->max_solves = (long)st->solves > s->max_solves ? (long)st->solves : s->max_solves;
}

static FILE *g_dump;

static void replay_episode(FILE *f, mpcof_t *m, uint32_t steps, long ep, summary_t *s, int verbose)
{
	tick_rec_t r;
	mpcof_reset(m);
	for (uint32_t t = 0; t < steps && read_rec(f, &r); t++)
	{
		mpc_real spr[3];
		double sp[3], h[12];
		mpcof_observe(m, r.gyro, r.applied);
		int dec = mpcof_label(m, r.q, r.gyro, r.tgt, spr);
		for (int a = 0; a < 3; a++) sp[a] = (double)spr[a];
		c_hidden(m, h);
		compare(s, &r, sp, h, ep, (long)t, verbose);
		if (dec) count_decision(s, &m->last);
		if (dec && g_dump) fprintf(g_dump, "%u %u %u\n", m->last.qp_axes, m->last.solves, m->last.fallbacks);
		s->ticks++;
	}
}

static void print_summary(const char *path, const summary_t *s)
{
	printf("file %s\n", path);
	printf("ticks %ld decisions %ld\n", s->ticks, s->decisions);
	printf("setpoint_mismatch_ticks %ld max_abs_err %.3e\n", s->sp_mismatch, s->sp_max_abs);
	printf("hidden_mismatch_ticks %ld max_rel_err %.3e\n", s->hid_mismatch, s->hid_max_rel);
	printf("qp_decisions %ld qp_axes %ld solves %ld max_solves_per_decision %ld fallbacks %ld failed %ld\n",
	       s->qp_decisions, s->qp_axes, s->solves, s->max_solves, s->fallbacks, s->failed);
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: %s traj.bin [-v]\n", argv[0]); return 2; }
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 2; }
	int verbose = 0;
	for (int i = 2; i < argc; i++)
	{
		if (strcmp(argv[i], "-v") == 0) verbose = 1;
		if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) g_dump = fopen(argv[++i], "w");
	}
	static mpcof_t m;
	mpcof_init(&m);
#ifdef MPCOF_MUTANT
	m.obs.l_gain *= (cdhat_real)(1.0 + 1e-12);   /* must be caught */
#endif
	summary_t s;
	memset(&s, 0, sizeof s);
	uint32_t n_eps = 0, steps = 0;
	if (fread(&n_eps, 4, 1, f) != 1) return 2;
	for (uint32_t e = 0; e < n_eps && fread(&steps, 4, 1, f) == 1; e++)
		replay_episode(f, &m, steps, (long)e, &s, verbose);
	fclose(f);
	if (g_dump) fclose(g_dump);
	print_summary(argv[1], &s);
	return (s.sp_mismatch || s.hid_mismatch || s.failed) ? 1 : 0;
}
