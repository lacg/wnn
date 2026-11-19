"""gen_digest: a new optional metrics field left at None must not change the digest (19/11/2026,
the ABI 58 vs 59 gate false-FAIL), while any real value change still does."""
import dataclasses
from typing import Optional

from wnn.control.gen_digest import population_digest


@dataclasses.dataclass
class _Old:
	err: float = 1.0
	steady: Optional[float] = 0.5


@dataclasses.dataclass
class _New(_Old):
	tail: Optional[float] = None


class _G:
	def fingerprint(self):
		return (1, 2, 3)


def test_unset_new_field_keeps_digest():
	assert population_digest([(_G(), _Old())]) == population_digest([(_G(), _New())])


def test_set_new_field_changes_digest():
	assert population_digest([(_G(), _Old())]) != population_digest([(_G(), _New(tail=2.0))])


def test_one_ulp_still_changes_digest():
	import math
	a = population_digest([(_G(), _Old(err=1.0))])
	b = population_digest([(_G(), _Old(err=math.nextafter(1.0, 2.0)))])
	assert a != b
