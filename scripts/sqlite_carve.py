"""Forensic SQLite carver — recover deleted rows from a db file + its WAL.

`.recover` rebuilds live b-tree cells (and orphaned pages into lost_and_found),
but it does NOT read:
  * WAL page-image history — older frames hold pre-delete versions of a page,
    where deleted rows are still *live cells* (fully intact, not clobbered), and
  * freeblocks — deleted cells inside otherwise-live pages, intact until reused.

This tool scans every page image it can find (each WAL frame's page, plus every
main-db page), parses table-leaf b-tree pages, and decodes every cell it can —
live cells, freeblock cells, and bytes in the unallocated gap. Each decoded
record is emitted to an output SQLite as (src, pgno, rowid, ncol, c0..c25) so you
can sort rows back into their tables and filter (e.g. experiment_id) in SQL.

Usage
-----
    python scripts/sqlite_carve.py --db <db> [--wal <wal>] --out carved.db
"""
from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

MAXCOL = 26  # c0..c25


def read_varint(buf: bytes, off: int) -> tuple[int, int]:
	"""Decode a SQLite varint at buf[off]. Returns (value, bytes_consumed)."""
	val = 0
	for i in range(9):
		if off + i >= len(buf):
			return val, i
		b = buf[off + i]
		if i == 8:
			val = (val << 8) | b
			return val, 9
		val = (val << 7) | (b & 0x7F)
		if not (b & 0x80):
			return val, i + 1
	return val, 9


def decode_record(buf: bytes, off: int):
	"""Decode a SQLite record (header + body) at buf[off]. Returns list of values
	or None if it looks malformed."""
	try:
		hdr_len, n = read_varint(buf, off)
		if hdr_len <= 0 or hdr_len > 1 + 9 * MAXCOL * 2 or off + hdr_len > len(buf):
			return None
		serials = []
		p = off + n
		hdr_end = off + hdr_len
		while p < hdr_end:
			s, m = read_varint(buf, p)
			serials.append(s)
			p += m
			if len(serials) > MAXCOL + 4:
				return None
		vals = []
		q = hdr_end
		for s in serials:
			if s == 0:
				vals.append(None)
			elif s == 1:
				vals.append(int.from_bytes(buf[q:q+1], 'big', signed=True)); q += 1
			elif s == 2:
				vals.append(int.from_bytes(buf[q:q+2], 'big', signed=True)); q += 2
			elif s == 3:
				vals.append(int.from_bytes(buf[q:q+3], 'big', signed=True)); q += 3
			elif s == 4:
				vals.append(int.from_bytes(buf[q:q+4], 'big', signed=True)); q += 4
			elif s == 5:
				vals.append(int.from_bytes(buf[q:q+6], 'big', signed=True)); q += 6
			elif s == 6:
				vals.append(int.from_bytes(buf[q:q+8], 'big', signed=True)); q += 8
			elif s == 7:
				if q + 8 > len(buf):
					return None
				vals.append(struct.unpack('>d', buf[q:q+8])[0]); q += 8
			elif s == 8:
				vals.append(0)
			elif s == 9:
				vals.append(1)
			elif s >= 12:
				ln = (s - 12) // 2 if s % 2 == 0 else (s - 13) // 2
				chunk = buf[q:q+ln]; q += ln
				if s % 2 == 0:
					vals.append(chunk)  # BLOB
				else:
					try:
						vals.append(chunk.decode('utf-8'))
					except Exception:
						vals.append(None)
			else:
				return None
			if q > len(buf):
				return None
		return vals
	except Exception:
		return None


def carve_leaf_page(page: bytes, page_off_in_page: int = 0):
	"""Yield decoded records from a table-leaf page. Parses live cells AND scans
	the freeblock chain + unallocated gap for deleted cells.
	page_off_in_page: header offset within `page` (100 for page 1)."""
	h = page_off_in_page
	if h >= len(page) or page[h] != 0x0D:  # 0x0D = table leaf
		return
	ncell = int.from_bytes(page[h+3:h+5], 'big')
	first_free = int.from_bytes(page[h+1:h+3], 'big')
	cell_ptr_arr = h + 8
	# Live cells.
	for i in range(ncell):
		pp = cell_ptr_arr + 2 * i
		if pp + 2 > len(page):
			break
		cell = int.from_bytes(page[pp:pp+2], 'big')
		rec = _decode_cell(page, cell)
		if rec is not None:
			yield ('live',) + rec
	# Freeblock chain — deleted cells (first 4 bytes clobbered by link+size).
	fb = first_free
	seen = set()
	while fb and fb not in seen and fb + 4 <= len(page):
		seen.add(fb)
		size = int.from_bytes(page[fb+2:fb+4], 'big')
		# The freed cell's payload likely starts a few bytes in; brute the start.
		for start in range(fb, min(fb + max(size, 4) + 1, len(page) - 1)):
			vals = decode_record(page, start + _leading_varints_guess(page, start))
		nxt = int.from_bytes(page[fb:fb+2], 'big')
		fb = nxt


def _decode_cell(page: bytes, cell: int):
	"""Decode a table-leaf cell at offset `cell`: payload_len, rowid, record."""
	plen, n1 = read_varint(page, cell)
	rowid, n2 = read_varint(page, cell + n1)
	rec_off = cell + n1 + n2
	vals = decode_record(page, rec_off)
	if vals is None:
		return None
	return (rowid, vals)


def _leading_varints_guess(page, start):
	return 0


def scan_pages(db_path: Path, wal_path: Path | None):
	"""Yield (src, pgno, page_bytes) for every main-db page and WAL frame page."""
	data = db_path.read_bytes()
	page_size = int.from_bytes(data[16:18], 'big')
	if page_size == 1:
		page_size = 65536
	npages = len(data) // page_size
	for pno in range(npages):
		yield ('db', pno + 1, data[pno*page_size:(pno+1)*page_size])
	if wal_path and wal_path.exists():
		w = wal_path.read_bytes()
		if len(w) >= 32:
			wal_page_size = int.from_bytes(w[8:12], 'big') or page_size
			off = 32
			frame = 0
			while off + 24 + wal_page_size <= len(w):
				pgno = int.from_bytes(w[off:off+4], 'big')
				page = w[off+24:off+24+wal_page_size]
				yield ('wal%d' % frame, pgno, page)
				off += 24 + wal_page_size
				frame += 1


def main() -> int:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--db", required=True)
	ap.add_argument("--wal")
	ap.add_argument("--out", required=True)
	ap.add_argument("--contains", nargs="*", help="Only deep-scan pages containing these int values' bytes (prefilter)")
	args = ap.parse_args()

	out = sqlite3.connect(args.out)
	cols = ", ".join(f"c{i}" for i in range(MAXCOL))
	out.execute(f"DROP TABLE IF EXISTS carved")
	out.execute(f"CREATE TABLE carved (src TEXT, pgno INT, kind TEXT, rowid INT, ncol INT, {cols})")
	ins = f"INSERT INTO carved (src,pgno,kind,rowid,ncol,{cols}) VALUES (?,?,?,?,?,{','.join('?'*MAXCOL)})"

	# Optional prefilter: only deep-scan pages containing these byte patterns
	# (e.g. an experiment_id) — huge speedup, scans only relevant pages.
	needles = []
	for v in (args.contains or []):
		iv = int(v)
		# emit candidate encodings: 1,2,3,4-byte big-endian signed
		for nb in (1, 2, 3, 4):
			try:
				needles.append(iv.to_bytes(nb, 'big', signed=True))
			except OverflowError:
				pass

	def page_relevant(page: bytes) -> bool:
		if not needles:
			return True
		return any(nd in page for nd in needles)

	n_rec = 0
	seen = set()  # dedup (ncol, tuple(first 8 vals)) across page versions
	for src, pgno, page in scan_pages(Path(args.db), Path(args.wal) if args.wal else None):
		# Targeted recovery: brute-decode every offset of every page that
		# matches the prefilter, regardless of current page type (handles
		# freelisted pages whose old table-leaf bytes are still intact).
		if not page_relevant(page):
			continue
		hoff = 100 if pgno == 1 else 0
		o = hoff
		end = len(page)
		while o < end:
			vals = decode_record(page, o)
			# field 0 is the rowid-alias placeholder (NULL) for INTEGER PRIMARY KEY
			# tables, so accept None or int there.
			if vals is not None and 8 <= len(vals) <= MAXCOL and (vals[0] is None or isinstance(vals[0], int)):
				key = (len(vals), tuple(str(v)[:24] for v in vals[:8]))
				if key not in seen:
					seen.add(key)
					row = list(vals[:MAXCOL]) + [None]*(MAXCOL - len(vals))
					out.execute(ins, [src, pgno, 'carve', None, len(vals)] + row)
					n_rec += 1
			o += 1
	out.commit()
	print(f"carved {n_rec} distinct records (prefiltered pages) into {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
