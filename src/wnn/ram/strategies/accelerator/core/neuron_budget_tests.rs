//! Unit tests for the whole-genome neuron budget.

use super::*;

#[test]
fn within_budget_is_unchanged()
{
	assert_eq!(budget_counts(&[200, 250], 500, 1), vec![200, 250]);
	assert_eq!(budget_counts(&[500], 500, 1), vec![500]);
}

#[test]
fn over_budget_shrinks_proportionally_to_exactly_total()
{
	let out = budget_counts(&[500, 500], 500, 1);
	assert_eq!(out, vec![250, 250]);
	let out = budget_counts(&[600, 300], 500, 1);
	assert_eq!(out.iter().sum::<usize>(), 500);
	assert_eq!(out, vec![333, 167]); // 333.3 / 166.7 → the larger remainder gets the spare
	let out = budget_counts(&[281, 219, 317, 181].map(|x| x * 2), 997, 1);
	assert_eq!(out.iter().sum::<usize>(), 997);
}

#[test]
fn floors_hold_and_excess_comes_from_the_largest()
{
	// 10 x 1000/1010 = 9.9 → floored to min 10; the excess comes off the big cluster.
	let out = budget_counts(&[1000, 10], 500, 10);
	assert_eq!(out, vec![490, 10]);
	// A cluster already below min is left alone.
	assert_eq!(budget_counts(&[1000, 3], 500, 10), vec![497, 3]);
}

#[test]
fn infeasible_budget_returns_floors()
{
	assert_eq!(budget_counts(&[100, 100, 100], 20, 10), vec![10, 10, 10]);
}

#[test]
fn fit_genome_truncates_each_cluster_from_its_end()
{
	// cluster 0: 4 neurons (widths 2,3,2,2), cluster 1: 4 neurons width 1
	let bits = vec![2, 3, 2, 2, 1, 1, 1, 1];
	let neurons = vec![4, 4];
	let conns: Vec<i64> = (0..13).collect();
	let (b, n, c) = fit_genome(&bits, &neurons, &conns, 4, 1);
	assert_eq!(n, vec![2, 2]);
	assert_eq!(b, vec![2, 3, 1, 1]);
	assert_eq!(c, vec![0, 1, 2, 3, 4, 9, 10]);
}

#[test]
fn fit_genome_without_connections()
{
	let (b, n, c) = fit_genome(&[5, 5, 5, 5], &[3, 1], &[], 2, 1);
	assert_eq!((b, n, c), (vec![5, 5], vec![1, 1], vec![]));
}
