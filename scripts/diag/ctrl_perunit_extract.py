import re, sys, json
def per_unit(path):
	t = open(path, errors="replace").read()
	g = re.search(r"\[stage-select\] HEADLINE stage=\S+ genome=(\S+)", t).group(1)
	stage = g.split("#")[0]
	# runner-up headline: its own HEADLINE-<g> blocks; #0 winner: the stage's report blocks (pop[0] RESULT)
	pat = rf"HELD-OUT REPORT \[HEADLINE-{re.escape(g)}\][^\n]*FRESH seed (\d+)[^\n]*\n(?:[^\n]*\n)?[^\n]*RESULT[^\n]*stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?pos=([\d.]+)m"
	rows = re.findall(pat, t)
	if not rows:
		pat = rf"HELD-OUT REPORT \[{stage}\][^\n]*FRESH seed (\d+)[^\n]*\n(?:[^\n]*\n)?[^\n]*RESULT[^\n]*stable=([\d.]+)%\s+err=([\d.]+)°\s+steady=([\d.]+)°.*?pos=([\d.]+)m"
		rows = re.findall(pat, t)
	out = {}
	for s, st, e, sd, p in rows:
		out[int(s)] = (float(st), float(e), float(sd), float(p))
	return g, out
res = {}
for name, path in [("control", "logs/controller/ctrl30/rescore35/control.out"),
                   ("leak999", "logs/controller/ctrl30/C-S2-d_leak999_full_s2.out"),
                   ("rateI", "logs/controller/ctrl38/C-S2-d_rateI_full_s2.out"),
                   ("posx2", "logs/controller/ctrl30/rescore35/posx2.out")]:
	g, d = per_unit(path)
	res[name] = d
	print(name, g, len(d), "units")
json.dump({k: {str(s): v for s, v in d.items()} for k, d in res.items()}, open(sys.argv[1], "w"))
