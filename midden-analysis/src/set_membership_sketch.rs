use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, PartialEq, Eq)]
pub struct SetMembershipSketch<const N: usize> {
    registers: [u32; N],
}

impl<const N: usize> SetMembershipSketch<N> {
    pub fn new() -> Self {
        Self {
            registers: [u32::MAX; N],
        }
    }

    pub fn add<H: Hash>(&mut self, item: &H) {
        for i in 0..N {
            let mut hasher = DefaultHasher::new();
            hasher.write_usize(i);
            item.hash(&mut hasher);
            let h = hasher.finish() as u32;
            if h < self.registers[i] {
                self.registers[i] = h;
            }
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for i in 0..N {
            result.registers[i] = self.registers[i].min(other.registers[i]);
        }
        result
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        for i in 0..N {
            if other.registers[i] > self.registers[i] {
                return false;
            }
        }
        true
    }

    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&r| r == u32::MAX)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(N * std::mem::size_of::<u32>());
        for &r in &self.registers {
            result.extend_from_slice(&r.to_le_bytes());
        }
        result
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), N * std::mem::size_of::<u32>());
        let mut registers = [0u32; N];
        for i in 0..N {
            let start = i * std::mem::size_of::<u32>();
            let end = start + std::mem::size_of::<u32>();
            registers[i] = u32::from_le_bytes(bytes[start..end].try_into().unwrap());
        }
        Self { registers }
    }
}

pub type DefaultMembershipSketch = SetMembershipSketch<8>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_membership_sketch() {
        let mut sketch = SetMembershipSketch::<4>::new();
        assert!(sketch.is_empty());

        sketch.add(&"item1");
        assert!(!sketch.is_empty());

        let bytes = sketch.to_bytes();
        let sketch2 = SetMembershipSketch::<4>::from_bytes(bytes.as_slice());
        assert_eq!(sketch, sketch2);

        let mut sketch3 = SetMembershipSketch::<4>::new();
        sketch3.add(&"item1");
        sketch3.add(&"item2");
        assert!(sketch.is_subset_of(&sketch3));
        assert!(!sketch3.is_subset_of(&sketch));
        let sketch4 = sketch.union(&sketch3);
        assert_eq!(sketch4, sketch3);
    }
}
