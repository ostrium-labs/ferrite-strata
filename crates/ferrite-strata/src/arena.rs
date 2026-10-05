//! The host's value arena: what an executable reads and writes.
//!
//! # Why this is in the core crate and not the runtime
//!
//! Because an executable that cannot be executed is not a useful abstraction. The
//! first draft of this design had `Executable` with `label`, `graph` and `outputs`
//! and no `run`, on the reasoning that buffer ownership is a plugin-ABI question for
//! B2. That reasoning was wrong: it made `compile` return an object with no way to
//! use it, and pushed the problem into a type-erased dead end.
//!
//! So the minimal arena lives here, next to the trait that needs it, and B2 widens
//! it. That is cheaper than a second round of redesign, and it is the same argument
//! as [`crate::id::ValueId`]: the arena is how values are named, and every crate in
//! the system needs to agree on that naming.
//!
//! # What is deliberately minimal
//!
//! Two entry kinds and one element type. [`ArenaEntry::F32`] is what the CPU path
//! computes in; [`ArenaEntry::Opaque`] is a placeholder for a device buffer whose
//! bytes Strata will not interpret. There is no `DType`-parameterised storage here,
//! because a `bf16` tensor's *bytes* are meaningless to the host — converting them
//! is the backend's job, at its own boundary, which is exactly where it should be.
//!
//! A backend that computes in `bf16` writes `Opaque`, and a host that wants the
//! numbers asks the backend to read it back. The host never guesses.

use std::collections::HashMap;

use crate::id::ValueId;

/// One value's storage.
#[derive(Clone, Debug, PartialEq)]
pub enum ArenaEntry {
    /// Row-major `f32`, which the host can read and compute on directly.
    F32(Vec<f32>),
    /// Bytes the host will not interpret: a device buffer, or an encoded weight.
    ///
    /// Named rather than `Device` because nothing here guarantees a device — a
    /// plugin may hand back a host-memory blob it wants to keep opaque. The host's
    /// only obligation is to pass it back unchanged.
    Opaque(Vec<u8>),
}

impl ArenaEntry {
    /// The `f32` data, if this entry is host-readable.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(data) => Some(data),
            Self::Opaque(_) => None,
        }
    }

    /// The `f32` data, if this entry is host-readable.
    pub fn as_f32_mut(&mut self) -> Option<&mut Vec<f32>> {
        match self {
            Self::F32(data) => Some(data),
            Self::Opaque(_) => None,
        }
    }
}

/// Every live value, keyed by [`ValueId`].
///
/// Not a `Vec`, because ids come from one graph and a subgraph's ids are a subset of
/// it — a dense vector indexed by the *original* graph's ids would carry holes, and
/// two subgraphs could not share an arena without a merge step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Arena {
    values: HashMap<ValueId, ArenaEntry>,
}

impl Arena {
    /// An empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a value, replacing anything already there.
    pub fn insert(&mut self, id: ValueId, entry: ArenaEntry) {
        self.values.insert(id, entry);
    }

    /// Store `f32` data.
    pub fn insert_f32(&mut self, id: ValueId, data: Vec<f32>) {
        self.insert(id, ArenaEntry::F32(data));
    }

    /// One value.
    #[must_use]
    pub fn get(&self, id: ValueId) -> Option<&ArenaEntry> {
        self.values.get(&id)
    }

    /// One value, mutably.
    #[must_use]
    pub fn get_mut(&mut self, id: ValueId) -> Option<&mut ArenaEntry> {
        self.values.get_mut(&id)
    }

    /// One value's `f32` data.
    #[must_use]
    pub fn f32_at(&self, id: ValueId) -> Option<&[f32]> {
        self.get(id).and_then(ArenaEntry::as_f32)
    }

    /// Whether the arena holds `id`.
    #[must_use]
    pub fn contains(&self, id: ValueId) -> bool {
        self.values.contains_key(&id)
    }

    /// How many values the arena holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the arena is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The ids present, in ascending order.
    ///
    /// Sorted so two runs over the same graph produce the same sequence, which is
    /// what makes a run's trace comparable to another run's.
    #[must_use]
    pub fn ids(&self) -> Vec<ValueId> {
        let mut ids: Vec<ValueId> = self.values.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Drop every value.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Copy every value from `other` into this arena, overwriting on conflict.
    ///
    /// How two subgraphs hand their boundary values back to a shared arena. Overwrite
    /// rather than error because a partition's result *is* authoritative for the ids
    /// it owns, and re-running a graph must be idempotent.
    pub fn extend_from(&mut self, other: &Self) {
        self.values
            .extend(other.values.iter().map(|(k, v)| (*k, v.clone())));
    }
}

#[cfg(test)]
mod tests {
    use super::{Arena, ArenaEntry};
    use crate::id::ValueId;

    #[test]
    fn an_arena_starts_empty() {
        let arena = Arena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
        assert!(arena.ids().is_empty());
        assert!(!arena.contains(ValueId(0)));
    }

    #[test]
    fn f32_data_round_trips() {
        let mut arena = Arena::new();
        arena.insert_f32(ValueId(1), vec![1.0, 2.0, 3.0]);
        assert_eq!(arena.f32_at(ValueId(1)), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn an_opaque_entry_is_not_readable_by_the_host() {
        // The host must not guess at bytes it does not understand; a backend that
        // computed in bf16 owns the conversion.
        let mut arena = Arena::new();
        arena.insert(ValueId(2), ArenaEntry::Opaque(vec![0xde, 0xad]));
        assert_eq!(arena.f32_at(ValueId(2)), None);
        assert!(arena.contains(ValueId(2)));
    }

    #[test]
    fn ids_are_reported_in_order_so_two_runs_comparably() {
        let mut arena = Arena::new();
        arena.insert_f32(ValueId(5), vec![0.0]);
        arena.insert_f32(ValueId(1), vec![0.0]);
        arena.insert_f32(ValueId(3), vec![0.0]);
        assert_eq!(arena.ids(), [ValueId(1), ValueId(3), ValueId(5)]);
    }

    #[test]
    fn storing_the_same_id_twice_replaces_it() {
        let mut arena = Arena::new();
        arena.insert_f32(ValueId(1), vec![1.0]);
        arena.insert_f32(ValueId(1), vec![2.0, 3.0]);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.f32_at(ValueId(1)), Some([2.0, 3.0].as_slice()));
    }

    #[test]
    fn extend_from_overwrites_so_a_rerun_is_idempotent() {
        let mut arena = Arena::new();
        arena.insert_f32(ValueId(1), vec![1.0]);
        arena.insert_f32(ValueId(2), vec![9.0]);

        let mut produced = Arena::new();
        produced.insert_f32(ValueId(1), vec![7.0]);
        produced.insert_f32(ValueId(3), vec![3.0]);

        arena.extend_from(&produced);
        assert_eq!(arena.f32_at(ValueId(1)), Some([7.0].as_slice()));
        assert_eq!(arena.f32_at(ValueId(2)), Some([9.0].as_slice()));
        assert_eq!(arena.f32_at(ValueId(3)), Some([3.0].as_slice()));
    }

    #[test]
    fn clear_empties_it() {
        let mut arena = Arena::new();
        arena.insert_f32(ValueId(1), vec![1.0]);
        arena.clear();
        assert!(arena.is_empty());
    }
}
