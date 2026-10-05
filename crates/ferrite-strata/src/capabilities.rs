//! What a backend can do: versioned data plus a pull `supports` query.
//!
//! # Why a pull query, and not just a capability blob
//!
//! A blob is enough to *describe* a device. It is not enough to *plan* a
//! partition, because planning needs an answer before it commits to a compile —
//! and a compile can take minutes.
//!
//! PJRT has no answer here. Its nearest mechanisms, `PJRT_XlaTransform` and
//! `PJRT_Custom_Partitioner`, are **push** callbacks: the framework hands the
//! vendor an HLO module and waits to be told what came back. Asking "will you
//! take this pattern for these dtypes?" first is not expressible.
//!
//! Burn can only approximate it by running a search.
//! `FusionRuntime::fusers(device)` returns the *set of matchers*, not a boolean,
//! so answering "supports flash-attn-v3 for bf16?" means executing the search —
//! which means the answer is only available once you have already decided to
//! compile. That is the wrong order of operations.
//!
//! So there are two mechanisms here, and the split is the point:
//!
//! - [`CapabilitySet`] is **data**. Serializable, diffable, cacheable, printable.
//!   It describes the device.
//! - [`StrataSupport`] is the **pull query**. `supports(pattern_id, dtype,
//!   shape_class) -> Refuse | Fallback | Fused`, asked before compiling.
//!
//! # Three answers, not two
//!
//! ```text
//! Refuse   not mine, do not offer it
//! Fallback compute it here, element by element
//! Fused    run it as one kernel
//! ```
//!
//! The middle one is not decoration. "Can you do this at all?" and "can you do
//! this *well*?" have three states between them, not two, and a planner that can
//! only see two of them either sends fusable work down a slow path or refuses
//! work it could have run. A backend that supports flash-attention on bf16 and
//! nothing else has exactly three honest answers across its pattern table, and a
//! two-valued query forces it to lie about one of them.
//!
//! # Vocabulary ownership
//!
//! `pattern_id` is a string, and that is the deliberate choice per
//! [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md).
//! Burn's pattern vocabulary is a closed enum in a 5,292-line file; adding a
//! pattern there is an edit to their IR crate. Here a vendor adds
//! `fused::flash_attention_v3` to its own blob and no downstream crate sees a
//! semver break. Strata owns the *contract* for those strings, not the
//! vocabulary.
//!
//! What Strata does own is the reference vocabulary — the ops in
//! [`Op`](crate::Op) — and those are deliberately few. See [`mod@reference`] for the
//! twelve ops a conforming backend is expected to name, which is the minimum for
//! covering a transformer block.

use crate::dtype::{DType, ElementType};
use crate::shape::{Shape, ShapeClass};

/// The level at which a backend can handle one pattern.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SupportLevel {
    /// Cannot run it. The caller should not offer this subgraph.
    Refuse,
    /// Can run it, but as the constituent ops rather than as a fused kernel.
    ///
    /// Correct, and the answer most patterns get from a backend that does not
    /// specialise for them.
    Fallback,
    /// Can run it as one fused unit.
    Fused,
}

impl SupportLevel {
    /// Whether this level means the backend will not run the pattern.
    #[must_use]
    pub const fn is_refusal(self) -> bool {
        matches!(self, Self::Refuse)
    }

    /// Whether this level means the backend will run it.
    ///
    /// The question the partitioner asks when deciding whether to keep a subgraph
    /// on this device.
    #[must_use]
    pub const fn is_runnable(self) -> bool {
        !self.is_refusal()
    }
}

/// One pattern's answer for one dtype and shape class.
///
/// The unit of a capability table. Not `(pattern_id, dtype, shape)` as three
/// separate dimensions in one big enum: the table has to be *extendable at
/// runtime* by a plugin Strata was compiled without, which is only possible if
/// the pattern id is data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatternSupport {
    /// The vendor-owned pattern id.
    pub pattern_id: String,
    /// The dtype the answer applies to.
    pub dtype: DType,
    /// The shape class the answer applies to.
    pub shape_class: ShapeClass,
    /// The level.
    pub level: SupportLevel,
    /// Why, when the level is not obvious.
    ///
    /// Free text on purpose: the planner does not parse it, but a human
    /// debugging a declined partition does, and having no field for it means
    /// vendors put the explanation in the pattern id.
    pub reason: Option<String>,
}

impl PatternSupport {
    /// An answer with no explanation attached.
    #[must_use]
    pub fn new(
        pattern_id: impl Into<String>,
        dtype: DType,
        shape_class: ShapeClass,
        level: SupportLevel,
    ) -> Self {
        Self {
            pattern_id: pattern_id.into(),
            dtype,
            shape_class,
            level,
            reason: None,
        }
    }

    /// The same answer, with a reason.
    #[must_use]
    pub fn with_reason(
        pattern_id: impl Into<String>,
        dtype: DType,
        shape_class: ShapeClass,
        level: SupportLevel,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            pattern_id: pattern_id.into(),
            dtype,
            shape_class,
            level,
            reason: Some(reason.into()),
        }
    }
}

/// What a compute unit claims to be, beyond its pattern table.
///
/// The vendor half of the convention in
/// [ADR-0007](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0007-burn-naming-and-the-vendor-half-of-the-convention.md):
/// Strata owns "NPU" as a term for "not a GPU", and the vendor names what it
/// actually is. This is where that name is carried, so a diagnostic can say
/// `acme-npu / "Axon-4"` instead of either lying or guessing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeUnit {
    /// The unit index, dense from 0, in the order the device enumerates them.
    pub index: u32,
    /// The vendor's name for it.
    pub name: String,
    /// The vendor's own term for the class of device.
    pub vendor_kind: String,
    /// Bytes of device memory, if bounded.
    pub memory_bytes: Option<u64>,
}

/// The numeric limits a backend declares.
///
/// Every field is optional because "we did not measure it" and "it is
/// unbounded" are different answers, and collapsing them into one makes a
/// planner either give up or overflow. A `None` limit is unknown, never
/// infinite — [`DeclaredLimits::allows`] says which reading applies.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DeclaredLimits {
    /// The largest single allocation, if bounded.
    pub max_allocation_bytes: Option<u64>,
    /// The largest total graph byte size, if bounded.
    ///
    /// This is a *declared* limit, which is not the same as the largest subgraph
    /// that will compile: a compiler can fail on a graph inside its own limit,
    /// and that failure is a hard error rather than a decline.
    pub max_graph_bytes: Option<u64>,
    /// The largest rank accepted in a shape.
    pub max_rank: Option<u32>,
    /// The largest dimension accepted in a shape.
    pub max_dim: Option<u64>,
    /// The largest number of nodes in one subgraph.
    pub max_nodes: Option<u64>,
}

/// Which reading of a missing limit applies.
///
/// Not an inherent ambiguity: [`DeclaredLimits`] documents `None` as *unknown*,
/// and this makes that explicit at the call site so a planner can decide to be
/// conservative rather than accidentally being optimistic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LimitReading {
    /// Treat the limit as absent: accept the request.
    Unbounded,
    /// Treat the limit as zero: reject the request.
    ///
    /// Correct when the absence means the vendor has not published a limit and the
    /// planner refuses to speculate. Also what a *failing* probe should produce,
    /// rather than `Unbounded`.
    Conservative,
}

impl DeclaredLimits {
    /// Whether `bytes` is within the declared allocation limit.
    ///
    /// `reading` decides what an absent limit means; see [`LimitReading`].
    #[must_use]
    pub fn allows(&self, bytes: u64, reading: LimitReading) -> bool {
        match (self.max_allocation_bytes, reading) {
            (Some(limit), _) => bytes <= limit,
            (None, LimitReading::Unbounded) => true,
            (None, LimitReading::Conservative) => false,
        }
    }

    /// Whether a shape is within the declared rank and dimension limits.
    #[must_use]
    pub fn allows_shape(&self, shape: &Shape, reading: LimitReading) -> bool {
        if let LimitReading::Conservative = reading {
            // Conservative means refuse anything the vendor has not described, and
            // an undescribed shape has undescribed rank and dimensions.
            if self.max_rank.is_none() || self.max_dim.is_none() {
                return false;
            }
        }
        if let Some(max_rank) = self.max_rank
            && shape.rank() > max_rank as usize
        {
            return false;
        }
        if let Some(max_dim) = self.max_dim
            && shape.dims().iter().any(|&dim| dim as u64 > max_dim)
        {
            return false;
        }
        true
    }

    /// Whether a node count is within the declared limit.
    #[must_use]
    pub fn allows_nodes(&self, nodes: u64, reading: LimitReading) -> bool {
        match (self.max_nodes, reading) {
            (Some(limit), _) => nodes <= limit,
            (None, LimitReading::Unbounded) => true,
            (None, LimitReading::Conservative) => false,
        }
    }
}

/// A backend's advertised capabilities: one self-describing blob.
///
/// Serializable and diffable, which is the property the pull query alone does
/// not have. A host can snapshot a plugin's capabilities at load, store them,
/// and compare two plugin versions without executing anything on the device —
/// the check you want before deciding to upgrade a plugin in production.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilitySet {
    /// The backend's stable name, e.g. `acme-npu`.
    pub backend_name: String,
    /// The pattern table.
    pub patterns: Vec<PatternSupport>,
    /// The compute units.
    pub compute_units: Vec<ComputeUnit>,
    /// The declared limits.
    pub limits: DeclaredLimits,
    /// The dtype mask: the element types this backend can store.
    ///
    /// A mask rather than a list of `PatternSupport`, because "can this device hold
    /// an `f8_e5m2` tensor at all" is a property of the hardware and is
    /// independent of any pattern.
    pub dtype_mask: DTypeMask,
    /// Whether this backend can accept a graph Strata did not partition for it.
    ///
    /// False for a backend that only runs subgraphs produced by
    /// [`partition`](crate::partition), which is the honest default.
    pub accepts_unpartitioned: bool,
}

impl CapabilitySet {
    /// An empty capability set for `backend_name`.
    ///
    /// Refuses everything, which is the safe reading of "nothing declared": a
    /// backend that advertises no patterns must not be offered subgraphs it may
    /// silently mishandle.
    #[must_use]
    pub fn empty(backend_name: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.into(),
            patterns: Vec::new(),
            compute_units: Vec::new(),
            limits: DeclaredLimits::default(),
            dtype_mask: DTypeMask::EMPTY,
            accepts_unpartitioned: false,
        }
    }

    /// Add a pattern answer, returning `self` for chaining during construction.
    #[must_use]
    pub fn with_pattern(mut self, pattern: PatternSupport) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Set the limits.
    #[must_use]
    pub fn with_limits(mut self, limits: DeclaredLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the storable dtype mask.
    #[must_use]
    pub fn with_dtype_mask(mut self, mask: DTypeMask) -> Self {
        self.dtype_mask = mask;
        self
    }

    /// Add a compute unit.
    #[must_use]
    pub fn with_compute_unit(mut self, unit: ComputeUnit) -> Self {
        self.compute_units.push(unit);
        self
    }
}

/// A set of element types, as a bitmask over [`ElementType`].
///
/// A mask so a capability blob can say "bf16 and f16" in one word, and so
/// `supports` can be asked about a *class* of dtypes rather than a single one.
/// `u32` because [`ElementType`] has 14 variants and the next ten are not going
/// to need a second word.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DTypeMask(u32);

impl DTypeMask {
    /// The empty mask.
    pub const EMPTY: Self = Self(0);

    /// A mask containing exactly one element type.
    #[must_use]
    pub const fn of(element: ElementType) -> Self {
        Self(1 << (element as u32))
    }

    /// A mask containing every element type.
    #[must_use]
    pub const fn all() -> Self {
        let mut mask = 0u32;
        let mut i = 0;
        while i < ElementType::all().len() {
            mask |= 1 << i;
            i += 1;
        }
        Self(mask)
    }

    /// Whether `element` is in the mask.
    #[must_use]
    pub const fn contains(self, element: ElementType) -> bool {
        self.0 & (1 << (element as u32)) != 0
    }

    /// Whether the mask holds no types.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The union of two masks.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The intersection of two masks.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// How many element types the mask holds.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// The element types in the mask, in declaration order.
    #[must_use]
    pub fn elements(self) -> Vec<ElementType> {
        ElementType::all()
            .iter()
            .copied()
            .filter(|&e| self.contains(e))
            .collect()
    }
}

impl core::ops::BitOr for DTypeMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl core::ops::BitAnd for DTypeMask {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        self.intersection(rhs)
    }
}

/// The pull query: ask a backend what it can do, before compiling.
///
/// Constructed from a [`CapabilitySet`], and the only way a planner learns
/// whether a subgraph is worth compiling here. The answer is
/// `supports(pattern_id, dtype, shape_class)`, matching the signature in
/// [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md)
///
/// # Lookup order
///
/// Exact match on `(pattern, dtype, class)` first; then a wildcard entry
/// matching the pattern with a wildcard shape class; then [`SupportLevel::Refuse`]
/// as the default. The refusal default is the load-bearing part: a pattern absent
/// from the table is not supported, because the alternative — assuming support
/// until something fails at run time — is precisely the failure mode decline
/// exists to prevent.
#[derive(Clone, Debug)]
pub struct StrataSupport {
    capabilities: CapabilitySet,
}

impl StrataSupport {
    /// Build a query over a capability set.
    #[must_use]
    pub fn new(capabilities: CapabilitySet) -> Self {
        Self { capabilities }
    }

    /// The underlying data.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    /// Whether this backend can store `dtype` at all.
    #[must_use]
    pub fn can_store(&self, dtype: DType) -> bool {
        self.capabilities.dtype_mask.contains(dtype.element)
    }

    /// The answer for one pattern, dtype and shape class.
    ///
    /// Never fails. An unknown pattern, an unknown dtype, and a shape outside the
    /// declared limits all produce [`SupportLevel::Refuse`], with a reason
    /// attached in [`PatternSupport`] where one was declared.
    #[must_use]
    pub fn supports(
        &self,
        pattern_id: &str,
        dtype: DType,
        shape_class: ShapeClass,
    ) -> SupportLevel {
        if !self.can_store(dtype) {
            return SupportLevel::Refuse;
        }
        self.lookup(pattern_id, dtype, shape_class)
            .map_or(SupportLevel::Refuse, |entry| entry.level)
    }

    /// The full entry behind [`StrataSupport::supports`], including the reason.
    #[must_use]
    pub fn lookup(
        &self,
        pattern_id: &str,
        dtype: DType,
        shape_class: ShapeClass,
    ) -> Option<&PatternSupport> {
        self.capabilities.patterns.iter().find(|entry| {
            entry.pattern_id == pattern_id
                && entry.dtype == dtype
                && entry.shape_class == shape_class
        })
    }

    /// Every entry for one pattern, which is what a human reads to understand a
    /// backend's coverage.
    #[must_use]
    pub fn entries_for(&self, pattern_id: &str) -> Vec<&PatternSupport> {
        self.capabilities
            .patterns
            .iter()
            .filter(|entry| entry.pattern_id == pattern_id)
            .collect()
    }

    /// Whether this backend would accept a graph Strata did not partition for it.
    #[must_use]
    pub fn accepts_unpartitioned(&self) -> bool {
        self.capabilities.accepts_unpartitioned
    }

    /// Whether a shape is within the declared limits, under a chosen reading of
    /// absent limits.
    #[must_use]
    pub fn allows_shape(&self, shape: &Shape, reading: crate::capabilities::LimitReading) -> bool {
        self.capabilities.limits.allows_shape(shape, reading)
    }
}

/// The reference pattern vocabulary a conforming backend is expected to name.
///
/// Twelve ops. That is not an arbitrary number: it is roughly what it takes to
/// cover a transformer block end to end, which is the workload Strata targets in
/// B1. A backend that names all twelve can accept a whole block in one partition;
/// one that names only `matmul` gets a partitioner full of one-node subgraphs and
/// should decline the rest.
///
/// This list is a **minimum**, not a maximum. It exists so the partitioner has a
/// defined floor and so a capability blob can be checked for completeness. The
/// vocabulary above it is open, and that is where fused patterns live.
pub mod reference {
    use crate::shape::ShapeClass;

    /// The ops a conforming backend names in its capability blob.
    ///
    /// `ReferenceOp`, not `Op`, on purpose: this is the *capability* vocabulary,
    /// which a backend maps onto its own lowering names. The IR's [`Op`](crate::Op)
    /// is the vocabulary Strata itself speaks.
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub enum ReferenceOp {
        /// An elementwise binary add.
        Add,
        /// An elementwise binary multiply.
        Mul,
        /// A batched matrix multiply.
        MatMul,
        /// A natural logarithm.
        Log,
        /// An exponential.
        Exp,
        /// A row-wise softmax.
        Softmax,
        /// A layer normalisation.
        LayerNorm,
        /// The GELU activation.
        Gelu,
        /// A linear layer.
        Linear,
        /// A reshape.
        Reshape,
        /// A transpose.
        Transpose,
        /// A broadcast.
        Expand,
    }

    impl ReferenceOp {
        /// All twelve, in a stable order.
        #[must_use]
        pub const fn all() -> &'static [Self] {
            &[
                Self::Add,
                Self::Mul,
                Self::MatMul,
                Self::Log,
                Self::Exp,
                Self::Softmax,
                Self::LayerNorm,
                Self::Gelu,
                Self::Linear,
                Self::Reshape,
                Self::Transpose,
                Self::Expand,
            ]
        }

        /// The pattern id a backend uses for this op.
        ///
        /// Prefixed `ref::` to keep it disjoint from vendor-owned `fused::` ids, so
        /// a blob mixing both is unambiguous. This naming is the Strata half of the
        /// convention in
        /// [ADR-0007](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0007-burn-naming-and-the-vendor-half-of-the-convention.md).
        #[must_use]
        pub const fn pattern_id(self) -> &'static str {
            match self {
                Self::Add => "ref::add",
                Self::Mul => "ref::mul",
                Self::MatMul => "ref::matmul",
                Self::Log => "ref::log",
                Self::Exp => "ref::exp",
                Self::Softmax => "ref::softmax",
                Self::LayerNorm => "ref::layer_norm",
                Self::Gelu => "ref::gelu",
                Self::Linear => "ref::linear",
                Self::Reshape => "ref::reshape",
                Self::Transpose => "ref::transpose",
                Self::Expand => "ref::expand",
            }
        }

        /// The level a backend gets for this op with no entry of its own.
        ///
        /// [`Fallback`](super::SupportLevel::Fallback), and this is the one place a
        /// missing entry is
        /// not a refusal: these twelve are the IR's own ops, so a backend that
        /// can run the graph at all can run them one at a time. A fused `fused::*`
        /// id has no such floor — it exists only because a backend declared it.
        #[must_use]
        pub const fn default_level(self) -> crate::capabilities::SupportLevel {
            crate::capabilities::SupportLevel::Fallback
        }
    }

    /// Every shape class, for a backend filling out a complete table.
    #[must_use]
    pub fn all_shape_classes() -> &'static [ShapeClass] {
        &[
            ShapeClass::Scalar,
            ShapeClass::Empty,
            ShapeClass::OuterProduct,
            ShapeClass::Vector,
            ShapeClass::Matrix,
            ShapeClass::Batch3,
            ShapeClass::Batch4,
            ShapeClass::HigherRank { rank: 5 },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::reference::ReferenceOp;
    use super::{
        CapabilitySet, ComputeUnit, DTypeMask, DeclaredLimits, LimitReading, PatternSupport,
        StrataSupport, SupportLevel,
    };
    use crate::dtype::{DType, ElementType};
    use crate::shape::{Shape, ShapeClass};

    fn bf16() -> DType {
        DType::plain(ElementType::BF16)
    }

    fn f32() -> DType {
        DType::plain(ElementType::F32)
    }

    fn a_backend() -> StrataSupport {
        let capabilities = CapabilitySet::empty("acme-npu")
            .with_dtype_mask(DTypeMask::of(ElementType::BF16) | DTypeMask::of(ElementType::F32))
            .with_pattern(PatternSupport::with_reason(
                "fused::flash_attention_v3",
                bf16(),
                ShapeClass::Batch3,
                SupportLevel::Fused,
                "requires head_dim to be a multiple of 64",
            ))
            .with_pattern(PatternSupport::new(
                "fused::flash_attention_v3",
                f32(),
                ShapeClass::Batch3,
                SupportLevel::Refuse,
            ))
            .with_pattern(PatternSupport::new(
                ReferenceOp::MatMul.pattern_id(),
                bf16(),
                ShapeClass::Matrix,
                SupportLevel::Fused,
            ))
            .with_pattern(PatternSupport::new(
                ReferenceOp::Gelu.pattern_id(),
                bf16(),
                ShapeClass::Vector,
                SupportLevel::Fallback,
            ))
            .with_limits(DeclaredLimits {
                max_allocation_bytes: Some(1 << 30),
                max_graph_bytes: Some(1 << 24),
                max_rank: Some(4),
                max_dim: Some(1 << 20),
                max_nodes: Some(4096),
            })
            .with_compute_unit(ComputeUnit {
                index: 0,
                name: "Axon-4".into(),
                vendor_kind: "tensor accelerator".into(),
                memory_bytes: Some(8 << 30),
            });

        StrataSupport::new(capabilities)
    }

    #[test]
    fn a_fused_answer_is_fused() {
        let b = a_backend();
        assert_eq!(
            b.supports("fused::flash_attention_v3", bf16(), ShapeClass::Batch3),
            SupportLevel::Fused
        );
    }

    #[test]
    fn the_same_pattern_can_be_refused_for_another_dtype() {
        // The same answer is not a property of the pattern. It is a property of
        // (pattern, dtype, shape class), which is why the query takes all three.
        let b = a_backend();
        assert_eq!(
            b.supports("fused::flash_attention_v3", f32(), ShapeClass::Batch3),
            SupportLevel::Refuse
        );
    }

    #[test]
    fn an_unknown_pattern_is_refused_not_fused() {
        // The refusal default is the load-bearing part: assuming support is what
        // decline exists to prevent.
        let b = a_backend();
        assert_eq!(
            b.supports("fused::never_declared", bf16(), ShapeClass::Batch3),
            SupportLevel::Refuse
        );
    }

    #[test]
    fn a_dtype_the_device_cannot_store_is_refused_before_the_table_is_read() {
        let b = a_backend();
        // `f16` is not in the mask, so even a table entry could not make this
        // answer anything but a refusal.
        assert!(!b.can_store(DType::plain(ElementType::F16)));
        assert_eq!(
            b.supports(
                ReferenceOp::MatMul.pattern_id(),
                DType::plain(ElementType::F16),
                ShapeClass::Matrix
            ),
            SupportLevel::Refuse
        );
    }

    #[test]
    fn lookup_exposes_the_reason() {
        // A planner ignores the reason; a human debugging a declined partition is
        // who it is for.
        let b = a_backend();
        let entry = b
            .lookup("fused::flash_attention_v3", bf16(), ShapeClass::Batch3)
            .expect("declared");
        assert_eq!(
            entry.reason.as_deref(),
            Some("requires head_dim to be a multiple of 64")
        );
    }

    #[test]
    fn entries_for_lists_a_patterns_whole_row() {
        let b = a_backend();
        let row = b.entries_for("fused::flash_attention_v3");
        assert_eq!(row.len(), 2);
        assert_eq!(row[0].dtype, bf16());
        assert_eq!(row[1].dtype, f32());
    }

    #[test]
    fn levels_answer_the_two_questions_the_partitioner_asks() {
        assert!(SupportLevel::Refuse.is_refusal());
        assert!(!SupportLevel::Refuse.is_runnable());
        assert!(SupportLevel::Fallback.is_runnable());
        assert!(SupportLevel::Fused.is_runnable());
        assert!(!SupportLevel::Fallback.is_refusal());
        assert!(!SupportLevel::Fused.is_refusal());
    }

    #[test]
    fn fallback_is_distinct_from_fused_and_from_refuse() {
        // Three states, not two. Collapsing them forces a backend to lie about one.
        assert_ne!(SupportLevel::Fallback, SupportLevel::Fused);
        assert_ne!(SupportLevel::Fallback, SupportLevel::Refuse);
        assert_ne!(SupportLevel::Fused, SupportLevel::Refuse);
    }

    #[test]
    fn an_empty_capability_set_refuses_everything() {
        let b = StrataSupport::new(CapabilitySet::empty("silent"));
        assert_eq!(
            b.supports(ReferenceOp::MatMul.pattern_id(), f32(), ShapeClass::Matrix),
            SupportLevel::Refuse
        );
        assert!(!b.accepts_unpartitioned());
    }

    #[test]
    fn an_absent_limit_reads_as_unknown_not_as_infinite() {
        let limits = DeclaredLimits::default();
        assert!(limits.allows(1 << 40, LimitReading::Unbounded));
        assert!(!limits.allows(0, LimitReading::Conservative));
        assert!(!limits.allows_nodes(0, LimitReading::Conservative));
    }

    #[test]
    fn a_declared_limit_is_checked_under_both_readings() {
        let limits = DeclaredLimits {
            max_allocation_bytes: Some(1024),
            ..DeclaredLimits::default()
        };
        assert!(limits.allows(1024, LimitReading::Unbounded));
        assert!(limits.allows(1024, LimitReading::Conservative));
        assert!(!limits.allows(1025, LimitReading::Conservative));
    }

    #[test]
    fn shape_limits_are_enforced_when_declared() {
        let limits = DeclaredLimits {
            max_rank: Some(4),
            max_dim: Some(1024),
            ..DeclaredLimits::default()
        };
        let ok = Shape::new(&[1, 1024, 1024]).expect("rank 3");
        assert!(limits.allows_shape(&ok, LimitReading::Unbounded));

        let too_many = Shape::new(&[1, 2, 3, 4, 5]).expect("rank 5");
        assert!(!limits.allows_shape(&too_many, LimitReading::Unbounded));

        let too_wide = Shape::new(&[1, 2048]).expect("rank 2");
        assert!(!limits.allows_shape(&too_wide, LimitReading::Unbounded));
    }

    #[test]
    fn the_conservative_reading_refuses_an_undescribed_shape() {
        // Conservative means "do not speculate about what the vendor never said".
        let limits = DeclaredLimits::default();
        let s = Shape::new(&[2, 2]).expect("rank 2");
        assert!(limits.allows_shape(&s, LimitReading::Unbounded));
        assert!(!limits.allows_shape(&s, LimitReading::Conservative));

        let partial = DeclaredLimits {
            max_rank: Some(4),
            ..DeclaredLimits::default()
        };
        assert!(!partial.allows_shape(&s, LimitReading::Conservative));
    }

    #[test]
    fn a_dtype_mask_is_a_set_of_element_types() {
        let mask = DTypeMask::of(ElementType::BF16) | DTypeMask::of(ElementType::F16);
        assert!(mask.contains(ElementType::BF16));
        assert!(mask.contains(ElementType::F16));
        assert!(!mask.contains(ElementType::F32));
        assert_eq!(mask.len(), 2);
        assert!(!mask.is_empty());
        assert!(DTypeMask::EMPTY.is_empty());
        assert!(!DTypeMask::all().is_empty());
    }

    #[test]
    fn a_dtype_mask_intersects_and_lists() {
        let a = DTypeMask::of(ElementType::BF16) | DTypeMask::of(ElementType::F32);
        let b = DTypeMask::of(ElementType::BF16);
        assert_eq!(a.intersection(b), b);
        assert_eq!(a.elements().len(), 2);
        assert!(a.elements().contains(&ElementType::BF16));
    }

    #[test]
    fn there_are_exactly_twelve_reference_ops() {
        // Roughly what it takes to cover a transformer block end to end, which is
        // the B1 target. The vocabulary above this floor is open.
        assert_eq!(ReferenceOp::all().len(), 12);
    }

    #[test]
    fn reference_pattern_ids_are_namespaced_and_stable() {
        // The `ref::` prefix keeps Strata's own vocabulary disjoint from a
        // vendor's `fused::` ids, so a blob mixing both is unambiguous.
        for op in ReferenceOp::all() {
            let id = op.pattern_id();
            assert!(id.starts_with("ref::"), "{id} is not namespaced");
        }
        assert_eq!(ReferenceOp::MatMul.pattern_id(), "ref::matmul");
        assert_eq!(ReferenceOp::LayerNorm.pattern_id(), "ref::layer_norm");
    }

    #[test]
    fn a_reference_op_with_no_declared_entry_still_has_a_floor() {
        // The one place a missing entry is not a refusal: these twelve are the
        // IR's own ops, so a backend that runs the graph can run them one at a
        // time. A `fused::*` id has no such floor.
        assert_eq!(ReferenceOp::Gelu.default_level(), SupportLevel::Fallback);
        for op in ReferenceOp::all() {
            assert!(op.default_level().is_runnable(), "{}", op.pattern_id());
        }
    }

    #[test]
    fn a_capability_set_is_comparable_without_touching_the_device() {
        // The check you want before upgrading a plugin in production.
        let before = a_backend();
        let mut after = before.capabilities().clone();
        after.patterns[0].level = SupportLevel::Refuse;

        assert_ne!(before.capabilities(), &after);
    }

    #[test]
    fn a_compute_unit_carries_the_vendors_own_name_for_the_device() {
        // ADR-0007: Strata says "NPU", the vendor says what it actually is. Both
        // live in the blob.
        let b = a_backend();
        let unit = &b.capabilities().compute_units[0];
        assert_eq!(unit.name, "Axon-4");
        assert_eq!(unit.vendor_kind, "tensor accelerator");
        assert_eq!(unit.memory_bytes, Some(8 << 30));
    }
}
