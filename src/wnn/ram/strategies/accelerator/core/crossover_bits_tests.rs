//! Unit tests for the `ga_crossover_bits` rules.

use super::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const TIB: usize = 200;

/// Genome with uniform width `b` per cluster; connections are distinct per neuron.
fn uniform(widths: &[usize], counts: &[usize], base: i64) -> Parent
{
	let mut bits = Vec::new();
	let mut conns = Vec::new();
	let mut next = base;
	for (&b, &n) in widths.iter().zip(counts)
	{
		for _ in 0..n
		{
			bits.push(b);
			for _ in 0..b
			{
				conns.push(next % TIB as i64);
				next += 1;
			}
		}
	}
	(bits, counts.to_vec(), conns, 0.0)
}

/// Child that copies p1 at even positions and p2 at odd ones (a mixed-width child).
fn alternate(p1: &Parent, p2: &Parent) -> Child
{
	let (v1, v2) = (View::new(p1), View::new(p2));
	let (mut bits, mut conns) = (Vec::new(), Vec::new());
	for (c, &n) in p1.1.iter().enumerate()
	{
		for i in 0..n
		{
			let v = if i % 2 == 0 || v2.bits_at(c, i).is_none() { &v1 } else { &v2 };
			bits.push(v.bits_at(c, i).unwrap());
			conns.extend_from_slice(v.conns_at(c, i));
		}
	}
	(bits, p1.1.clone(), conns)
}

fn consistent(child: &Child)
{
	assert_eq!(child.0.len(), child.1.iter().sum::<usize>());
	assert_eq!(child.2.len(), child.0.iter().sum::<usize>());
	assert!(child.2.iter().all(|&x| x >= 0 && (x as usize) < TIB));
}

#[test]
fn inherit_is_identity_and_draws_nothing()
{
	let (p1, p2) = (uniform(&[34, 34], &[30, 20], 0), uniform(&[32, 32], &[25, 25], 7));
	let child = alternate(&p1, &p2);
	let mut a = StdRng::seed_from_u64(1);
	let out = apply(CrossoverBits::Inherit, child.clone(), &p1, &p2, TIB, &mut a);
	assert_eq!(out, child);
	let mut b = StdRng::seed_from_u64(1);
	assert_eq!(a.gen::<u64>(), b.gen::<u64>(), "Inherit must not consume RNG");
}

#[test]
fn blend_neuron_widths_stay_between_parents_and_reach_the_middle()
{
	let (p1, p2) = (uniform(&[34, 34], &[300, 200], 0), uniform(&[32, 32], &[250, 250], 11));
	let mut rng = StdRng::seed_from_u64(2);
	let out = apply(CrossoverBits::BlendNeuron, alternate(&p1, &p2), &p1, &p2, TIB, &mut rng);
	consistent(&out);
	assert!(out.0.iter().all(|&b| (32..=34).contains(&b)));
	assert!(out.0.contains(&33), "BLX-0 should reach the in-between width");
	// cluster 0 positions 250..300 exist only in p1: they keep 34
	assert!(out.0[250..300].iter().all(|&b| b == 34));
}

#[test]
fn blend_cluster_gives_one_width_per_cluster()
{
	let (p1, p2) = (uniform(&[34, 20], &[40, 10], 0), uniform(&[32, 24], &[30, 15], 5));
	for seed in 0..50
	{
		let mut rng = StdRng::seed_from_u64(seed);
		let out = apply(CrossoverBits::BlendCluster, alternate(&p1, &p2), &p1, &p2, TIB, &mut rng);
		consistent(&out);
		let (c0, c1) = (&out.0[..40], &out.0[40..]);
		assert!(c0.iter().all(|&b| b == c0[0]) && (32..=34).contains(&c0[0]));
		assert!(c1.iter().all(|&b| b == c1[0]) && (20..=24).contains(&c1[0]));
	}
}

#[test]
fn grown_neuron_keeps_own_connections_then_takes_partners()
{
	let own = [1i64, 2, 3];
	let partners: [&[i64]; 2] = [&[1, 2, 3], &[3, 9, 8, 7]];
	let mut rng = StdRng::seed_from_u64(3);
	assert_eq!(refit(&own, 5, &partners, TIB, &mut rng), vec![1, 2, 3, 9, 8]);
}

#[test]
fn shrunk_neuron_keeps_an_ordered_subset()
{
	let own = [10i64, 20, 30, 40, 50];
	let mut rng = StdRng::seed_from_u64(4);
	let kept = keep_subset(&own, 3, &mut rng);
	assert_eq!(kept.len(), 3);
	assert!(kept.windows(2).all(|w| w[0] < w[1]) && kept.iter().all(|k| own.contains(k)));
}

#[test]
fn parse_rejects_unknown()
{
	assert_eq!(CrossoverBits::parse("blend_cluster"), Ok(CrossoverBits::BlendCluster));
	assert!(CrossoverBits::parse("blend").is_err());
}
