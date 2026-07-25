use super::{auto_mc_tbtr, auto_time_tbtr, factor_fingerprint, same_factors};

#[test]
fn auto_time_requires_matching_cached_date() {
    assert!(auto_time_tbtr(Some("2022-02-22"), "2022-02-22", false));
    assert!(!auto_time_tbtr(Some("2022-02-21"), "2022-02-22", false));
    assert!(!auto_time_tbtr(None, "2022-02-22", false));
    // Exclusions force RAPTOR even over a matching cache.
    assert!(!auto_time_tbtr(Some("2022-02-22"), "2022-02-22", true));
}

#[test]
fn auto_mc_requires_matching_cache_and_supported_query() {
    let factors = [f64::NAN, 74.0, 101.0];
    let cached = Some(("2022-02-22", &factors[..]));
    assert!(auto_mc_tbtr(cached, "2022-02-22", &factors, false));
    assert!(!auto_mc_tbtr(cached, "2022-02-23", &factors, false));
    let other = [f64::NAN, 74.0, 88.0];
    assert!(!auto_mc_tbtr(cached, "2022-02-22", &other, false));
    assert!(!auto_mc_tbtr(cached, "2022-02-22", &factors, true));
    assert!(!auto_mc_tbtr(None, "2022-02-22", &factors, false));
}

#[test]
fn factor_equality_is_bitwise_and_exact() {
    // NaN-padded identical vectors are the same configuration (float `==`
    // would say no), and a prefix is not.
    assert!(same_factors(&[f64::NAN, 74.0], &[f64::NAN, 74.0]));
    assert!(!same_factors(&[f64::NAN, 74.0], &[f64::NAN]));

    // A crafted FNV collision: solve the second element with the modular
    // inverse of the FNV prime so both vectors hash identically. Under
    // fingerprint equality a set built for `a` would serve a query
    // resolving to `b`; the exact comparison refuses it.
    const PRIME: u64 = 0x100000001b3;
    let mut inverse = PRIME;
    for _ in 0..6 {
        // Newton's iteration doubles the correct low bits each round.
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(PRIME.wrapping_mul(inverse)));
    }
    assert_eq!(PRIME.wrapping_mul(inverse), 1);
    let offset = 0xcbf29ce484222325u64;
    let a = [74.0f64, 101.0];
    let after_first = (offset ^ a[0].to_bits()).wrapping_mul(PRIME);
    let target = (after_first ^ a[1].to_bits()).wrapping_mul(PRIME);
    let b_first = 75.0f64;
    let after_b_first = (offset ^ b_first.to_bits()).wrapping_mul(PRIME);
    let b = [
        b_first,
        f64::from_bits(target.wrapping_mul(inverse) ^ after_b_first),
    ];
    assert_eq!(factor_fingerprint(&a), factor_fingerprint(&b));
    assert!(!same_factors(&a, &b));
    assert!(!auto_mc_tbtr(
        Some(("2022-02-22", &a[..])),
        "2022-02-22",
        &b,
        false
    ));
}
