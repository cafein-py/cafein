use super::*;

fn costs(values: &[Option<f64>]) -> Vec<Option<f64>> {
    values.to_vec()
}

#[test]
fn step_weight_is_one_inside_and_zero_beyond_the_budget() {
    let decay = Decay::Step;
    assert_eq!(decay.weight(0.0, 1800.0), 1.0);
    assert_eq!(decay.weight(1800.0, 1800.0), 1.0);
    assert_eq!(decay.weight(1800.1, 1800.0), 0.0);
}

#[test]
fn linear_weight_ramps_around_the_budget() {
    let decay = Decay::Linear { width: 600.0 };
    // Ramp spans [b - width/2, b + width/2] but is truncated at b.
    assert_eq!(decay.weight(0.0, 1800.0), 1.0);
    assert_eq!(decay.weight(1500.0, 1800.0), 1.0);
    assert!((decay.weight(1650.0, 1800.0) - 0.75).abs() < 1e-12);
    assert!((decay.weight(1800.0, 1800.0) - 0.5).abs() < 1e-12);
    assert_eq!(decay.weight(1801.0, 1800.0), 0.0);
}

#[test]
fn exponential_weight_halves_at_the_half_life() {
    let decay = Decay::Exponential { half_life: 900.0 };
    assert_eq!(decay.weight(0.0, 3600.0), 1.0);
    assert!((decay.weight(900.0, 3600.0) - 0.5).abs() < 1e-12);
    assert!((decay.weight(1800.0, 3600.0) - 0.25).abs() < 1e-12);
    assert_eq!(decay.weight(3601.0, 3600.0), 0.0);
}

#[test]
fn logistic_weight_is_half_at_the_budget() {
    let decay = Decay::Logistic { scale: 120.0 };
    assert!((decay.weight(1800.0, 1800.0) - 0.5).abs() < 1e-12);
    assert!(decay.weight(1500.0, 1800.0) > 0.9);
    // Hard truncation beats the sigmoid's tail.
    assert_eq!(decay.weight(1900.0, 1800.0), 0.0);
}

#[test]
fn opportunity_sums_count_reachable_mass_per_budget_and_field() {
    // Three destinations, two fields (say population, jobs).
    let costs = costs(&[Some(600.0), Some(1200.0), None]);
    let opportunities = [10.0, 1.0, 20.0, 2.0, 40.0, 4.0];
    let sums = opportunity_sums(&costs, &opportunities, 2, &[900.0, 1800.0], &Decay::Step);
    // Budget 900: only the first destination. Budget 1800: first two.
    assert_eq!(sums, vec![10.0, 1.0, 30.0, 3.0]);
}

#[test]
fn opportunity_sums_apply_decay_weights() {
    let costs = costs(&[Some(900.0)]);
    let sums = opportunity_sums(
        &costs,
        &[8.0],
        1,
        &[3600.0],
        &Decay::Exponential { half_life: 900.0 },
    );
    assert!((sums[0] - 4.0).abs() < 1e-12);
}

#[test]
fn opportunity_sums_with_no_destinations_are_zero() {
    let sums = opportunity_sums(&[], &[], 3, &[1800.0], &Decay::Step);
    assert_eq!(sums, vec![0.0, 0.0, 0.0]);
}

#[test]
fn nearest_ranks_by_cost_then_index_and_respects_the_horizon() {
    let costs = costs(&[Some(300.0), Some(100.0), None, Some(100.0), Some(9000.0)]);
    // Tie between destinations 1 and 3 breaks by index.
    assert_eq!(
        nearest(&costs, 3, 7200.0),
        vec![(1, 100.0), (3, 100.0), (0, 300.0)]
    );
    // The horizon excludes destination 4 even with room in k.
    assert_eq!(
        nearest(&costs, 5, 7200.0),
        vec![(1, 100.0), (3, 100.0), (0, 300.0)]
    );
    // Fewer than k reachable: fewer pairs, never padding.
    assert_eq!(nearest(&costs, 5, 150.0), vec![(1, 100.0), (3, 100.0)]);
    assert_eq!(nearest(&costs, 0, 7200.0), vec![]);
}

#[test]
fn nearest_keeps_the_k_smallest_across_a_long_stream() {
    let costs: Vec<Option<f64>> = (0..1000).rev().map(|c| Some(c as f64)).collect();
    let best = nearest(&costs, 2, f64::MAX);
    assert_eq!(best, vec![(999, 0.0), (998, 1.0)]);
}

#[test]
fn nearest_survives_a_huge_k() {
    let costs = costs(&[Some(2.0), Some(1.0)]);
    // k far beyond the destination count must neither allocate for k
    // nor overflow k + 1.
    assert_eq!(nearest(&costs, usize::MAX, 10.0), vec![(1, 1.0), (0, 2.0)]);
}

#[test]
fn reached_returns_in_index_order_within_the_budget() {
    let costs = costs(&[Some(1800.0), None, Some(1.0), Some(1801.0)]);
    assert_eq!(reached(&costs, 1800.0), vec![0, 2]);
    assert_eq!(reached(&costs, 0.5), Vec::<usize>::new());
}
