use std::hash::{Hash, Hasher};

use ahash::AHasher;
use float16::Bf16;

pub struct SizeSketch <const N: usize> {
    /** A sum-distinct sketch for estimating the size of a multiset */
    /* The key observation that powers this sketch is that if we have exponential random variables
     * with parameters lambda_1, lambda_2, ..., lambda_n, then the minimum of these random variables is an exponential
     * with parameter lambda_1 + lambda_2 + ... + lambda_n.
     * 
     * At a high level, it works by, every time a new value is added, deterministically generating N exponential random variables with
     * parameters equal to the value being added, comparing them with the values already there, and taking the minimum.
     * We then estimate the size of the multiset by taking the third quartile of the values in the sketch (the third quartile gave the
     * best results of all the estimators we tried), and deriving an estimate for lambda from that.
     * 
     * We do one key thing differently though, as an optimisation. We derive the estimate for lambda _before_ adding the new value.
     * We estimate lambda as -log(1 - p) / x, so doing this reverses the order of comparison (so we actually take maxima rather
     * than minima, and take the first quartile), but means that we can store the estimate
     * in a relatively low-precision format (we use 16-bit floats), which significantly reduces the memory usage of the sketch.
     */
    values: [Bf16; N],
}

impl <const N: usize> SizeSketch<N> {
    pub fn new() -> Self {
        Self {
            values: [Bf16::from_f64(0.0); N],
        }
    }

    pub fn add<I: Hash>(&mut self, id: I, value: f64) {
        let log_one_minus_p = (0.25f64).ln();
        for i in 0..N {
            let mut hasher = AHasher::default();
            hasher.write_usize(i);
            id.hash(&mut hasher);
            let hash = hasher.finish();
            let u = hash as f64 / (u64::MAX as f64);
            // # Code commented out deliberately, to show derivation of estimator more clearly.
            // let exp_variate = -(u).ln() / value;
            // let estimator = -log_one_minus_p / exp_variate;
            let estimator = value * log_one_minus_p / u.ln();
            let estimator: Bf16 = Bf16::from_f64(estimator);
            if estimator > self.values[i] {
                self.values[i] = estimator;
            }
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for i in 0..N {
            result.values[i] = self.values[i].max(other.values[i]);
        }
        result
    }

    pub fn estimate(&self) -> f64 {
        let mut values: [Bf16; N] = self.values;
        let q1 = values.as_mut_slice().select_nth_unstable_by((N + 1) / 4 - 1, Bf16::total_cmp).1;
        (*q1).into()
    }
}

pub type LowPrecisionSizeSketch = SizeSketch<7>;
pub type MediumPrecisionSizeSketch = SizeSketch<31>;
pub type HighPrecisionSizeSketch = SizeSketch<127>;

#[cfg(test)]
mod test {
    use std::array;

use super::*;

    macro_rules! make_test {
        ($($testname:ident $typename:ident $precision:literal),*) => {
            $(
                #[test]
                fn $testname() {
                    let mut sum = 0.0;
                    let mut sum_err = 0.0;
                    let mut sum_sq_err = 0.0;
                    let sketches: [$typename; 1000] = array::from_fn(|i| {
                        let mut sketch = $typename::new();
                        sketch.add(10 * i + 1, 10.0);
                        sketch.add(10 * i + 2, 20.0);
                        sketch.add(10 * i + 3, 30.0);
                        sketch.add(10 * i + 4, 40.0);
                        sketch.add(10 * i + 4, 40.0); // Duplicate value, should not affect the sketch.
                        
                        sketch
                    });
                    let estimates = sketches.map(|s| {
                        let est = s.estimate();
                        sum += est;
                        let err = (est - 100.0).abs();
                        sum_err += err;
                        sum_sq_err += err * err;
                        est
                    });
                    // Correct answer is 100, but we allow anything within the given percentage
                    let correct_estimates = estimates.iter().filter(|&&e| e > (100.0 - $precision) && e < (100.0 + $precision)).count();
                    println!(
                        "{}: Average estimate: {}, average error: {}, RMSE: {}, correct estimates within {}%: {} out of 1000",
                        stringify!($typename), sum / 1000.0, sum_err / 1000.0, (sum_sq_err / 1000.0).sqrt(), $precision, correct_estimates);
                    assert!(correct_estimates > 800, "Too many incorrect estimates: {} out of 1000", 1000 - correct_estimates);
                }                
            )*
        };
    }

    make_test!(
        test_low_precision_sketch LowPrecisionSizeSketch 60.0,
        test_medium_precision_sketch MediumPrecisionSizeSketch 30.0,
        test_high_precision_sketch HighPrecisionSizeSketch 15.0
    );

    #[test]
    fn test_union() {
        let mut sketch1 = HighPrecisionSizeSketch::new();
        sketch1.add(1, 10.0);
        sketch1.add(2, 20.0);
        sketch1.add(3, 30.0);
        let mut sketch2 = HighPrecisionSizeSketch::new();
        sketch2.add(3, 30.0);
        sketch2.add(4, 40.0);
        sketch2.add(5, 50.0);
        let union_sketch = sketch1.union(&sketch2);
        let estimate = union_sketch.estimate();
        assert!(estimate > 130.0 && estimate < 170.0, "Union estimate is way off: {}", estimate);
    }

}