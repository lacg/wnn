//! Unit tests for the receptive-field projection (receptive_fields.rs).

use super::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// 100 bits: family 0 = 40 bits, 1 = 30, 2 = 20, 3 = 10 (contiguous runs).
fn map_4() -> Vec<u8>
{
	[(0u8, 40usize), (1, 30), (2, 20), (3, 10)].iter().flat_map(|&(f, n)| std::iter::repeat(f).take(n)).collect()
}

fn uniform_genome(widths: &[usize], tib: usize, rng: &mut StdRng) -> Vec<i64>
{
	widths.iter().flat_map(|&w| (0..w).map(|_| rng.gen_range(0..tib as i64)).collect::<Vec<_>>()).collect()
}

fn slices<'a>(widths: &'a [usize], conns: &'a [i64]) -> impl Iterator<Item = &'a [i64]> + 'a
{
	let mut start = 0;
	widths.iter().map(move |&w| {
		let s = &conns[start..start + w];
		start += w;
		s
	})
}

#[test]
fn refuses_bad_cross_and_empty_map()
{
	assert!(ReceptiveFields::new(map_4(), 0.5).is_err(), "0.5 breaks the majority rule");
	assert!(ReceptiveFields::new(map_4(), -0.1).is_err());
	assert!(ReceptiveFields::new(vec![], 0.2).is_err());
	assert!(ReceptiveFields::new(map_4(), 0.0).is_ok());
}

#[test]
fn budget_is_floor_of_cross_times_width()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	assert_eq!([4, 5, 9, 10, 20, 34].map(|w| rf.budget(w)), [0, 1, 1, 2, 4, 6]);
	let rf = ReceptiveFields::new(map_4(), 0.29).unwrap();
	assert_eq!(rf.budget(100), 29, "float slop must not lose a slot");
}

#[test]
fn every_neuron_satisfies_the_budget_after_projection()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut rng = StdRng::seed_from_u64(1);
	let widths: Vec<usize> = (0..400).map(|i| 4 + i % 31).collect();
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	let changed = rf.project(&widths, &mut conns, &mut rng).unwrap();
	assert!(changed > 300, "fresh uniform neurons almost all violate (changed {changed})");
	for s in slices(&widths, &conns)
	{
		let f = rf.field(s);
		assert!(f.exempt || (f.family.is_some() && s.len() - f.in_family <= f.budget), "{s:?} -> {f:?}");
	}
}

#[test]
fn projection_is_idempotent_and_draws_nothing_on_a_valid_genome()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut rng = StdRng::seed_from_u64(2);
	let widths: Vec<usize> = (0..200).map(|i| 6 + i % 25).collect();
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	let before = conns.clone();
	let mut a = StdRng::seed_from_u64(99);
	let b = a.clone();
	assert_eq!(rf.project(&widths, &mut conns, &mut a).unwrap(), 0);
	assert_eq!(conns, before, "a valid genome is a fixed point");
	assert_eq!(a.next_u64(), b.clone().next_u64(), "a valid genome must not consume RNG");
}

#[test]
fn same_seed_same_projection()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let widths = vec![12usize; 50];
	let base = uniform_genome(&widths, 100, &mut StdRng::seed_from_u64(3));
	let (mut x, mut y) = (base.clone(), base);
	rf.project(&widths, &mut x, &mut StdRng::seed_from_u64(4)).unwrap();
	rf.project(&widths, &mut y, &mut StdRng::seed_from_u64(4)).unwrap();
	assert_eq!(x, y);
}

#[test]
fn budget_zero_is_pure_family()
{
	let rf = ReceptiveFields::new(map_4(), 0.0).unwrap();
	let mut rng = StdRng::seed_from_u64(5);
	let widths = vec![8usize; 100];
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	for s in slices(&widths, &conns)
	{
		let f = rf.field(s);
		assert_eq!(f.in_family, 8, "{s:?}");
	}
}

#[test]
fn small_family_is_never_a_home_at_a_width_it_cannot_fill()
{
	// width 20, budget 4: a home needs 16 bits -> family 3 (10 bits) is ineligible.
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut rng = StdRng::seed_from_u64(6);
	let widths = vec![20usize; 300];
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	let homes: Vec<u8> = slices(&widths, &conns).map(|s| rf.field(s).family.unwrap()).collect();
	assert!(!homes.contains(&3));
	// A neuron wider than every family + budget is exempt and untouched.
	let wide = vec![60usize];
	let mut c = uniform_genome(&wide, 100, &mut rng);
	let before = c.clone();
	assert_eq!(rf.project(&wide, &mut c, &mut rng).unwrap(), 0);
	assert_eq!(c, before);
	assert!(rf.field(&c).exempt);
}

#[test]
fn fresh_neurons_pick_families_by_their_bit_share()
{
	// Narrow neurons (w=4, budget 0): every family is eligible, shares 40/30/20/10.
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut rng = StdRng::seed_from_u64(7);
	let widths = vec![4usize; 20_000];
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	let mut hist = [0usize; 4];
	slices(&widths, &conns).for_each(|s| hist[rf.field(s).family.unwrap() as usize] += 1);
	let share: Vec<f64> = hist.iter().map(|&h| h as f64 / 20_000.0).collect();
	for (got, want) in share.iter().zip([0.4, 0.3, 0.2, 0.1])
	{
		assert!((got - want).abs() < 0.03, "share {share:?}");
	}
}

#[test]
fn held_family_survives_small_perturbations()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut rng = StdRng::seed_from_u64(8);
	let widths = vec![20usize; 200];
	let mut conns = uniform_genome(&widths, 100, &mut rng);
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	let homes: Vec<Option<u8>> = slices(&widths, &conns).map(|s| rf.field(s).family).collect();
	// Two connections of every neuron jump to random bits (a mutation), re-project.
	for n in 0..200
	{
		conns[n * 20] = rng.gen_range(0..100);
		conns[n * 20 + 7] = rng.gen_range(0..100);
	}
	rf.project(&widths, &mut conns, &mut rng).unwrap();
	let after: Vec<Option<u8>> = slices(&widths, &conns).map(|s| rf.field(s).family).collect();
	assert_eq!(homes, after, "a 2-connection drift must not move a neuron to another family");
}

#[test]
fn length_mismatch_is_refused()
{
	let rf = ReceptiveFields::new(map_4(), 0.2).unwrap();
	let mut conns = vec![0i64; 5];
	assert!(rf.project(&[3, 3], &mut conns, &mut StdRng::seed_from_u64(0)).is_err());
}
