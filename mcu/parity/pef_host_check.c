/* pef64 host exactness check: replays (neuron, addr, expected) uint32 triples
 * through the uint32 C lookup and the pef64 C lookup (pef_lookup_tu.c, linked
 * twice) and counts every disagreement. Driven by test_pef64_exactness.py.
 *
 * usage: pef_host_check <triples.u32>
 *   prints "C<TAB>checked N<TAB>u32_vs_pef M<TAB>u32_vs_expected E<TAB>pef_vs_expected P"
 *   exit 0 only when something was checked and all three counts are 0. */
#include <stdint.h>
#include <stdio.h>

int u32_present(uint32_t n, uint32_t addr);
int pef_present(uint32_t n, uint32_t addr);
uint32_t u32_neurons(void);
uint32_t pef_neurons(void);

typedef struct { unsigned long checked, u32_pef, u32_exp, pef_exp; } tally_t;

static void check_one(const uint32_t t[3], tally_t *c)
{
	int u = u32_present(t[0], t[1]), p = pef_present(t[0], t[1]), e = (int)t[2];
	c->checked++;
	c->u32_pef += (u != p);
	c->u32_exp += (u != e);
	c->pef_exp += (p != e);
}

static int replay(FILE *f, tally_t *c)
{
	uint32_t t[3];
	while (fread(t, sizeof t[0], 3, f) == 3)
	{
		if (t[0] >= u32_neurons()) { fprintf(stderr, "neuron %u out of range\n", (unsigned)t[0]); return 2; }
		check_one(t, c);
	}
	return 0;
}

int main(int argc, char **argv)
{
	FILE *f;
	tally_t c = {0, 0, 0, 0};
	int rc;
	if (argc != 2 || !(f = fopen(argv[1], "rb"))) { fprintf(stderr, "usage: %s <triples.u32>\n", argv[0]); return 2; }
	if (u32_neurons() != pef_neurons()) { fprintf(stderr, "headers disagree on WNN_NEURONS\n"); return 2; }
	rc = replay(f, &c);
	fclose(f);
	if (rc) return rc;
	printf("C\tchecked %lu\tu32_vs_pef %lu\tu32_vs_expected %lu\tpef_vs_expected %lu\n",
	       c.checked, c.u32_pef, c.u32_exp, c.pef_exp);
	return (c.checked && !c.u32_pef && !c.u32_exp && !c.pef_exp) ? 0 : 1;
}
