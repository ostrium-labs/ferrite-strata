//! The traits a plugin implements: [`Backend`], [`Executable`], and the tri-state
//! compile that sits between them.
//!
//! # Compile is three-valued
//!
//! ```text
//! Compiled(executable)   mine, here it is
//! Declined(reason)       not mine, send it elsewhere
//! Err(error)             the call is wrong, or the device broke
//! ```
//!
//! Three outcomes, not two, and the middle one is the design. See
//! [ADR-0006](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0006-compile-returns-a-tri-state-with-decline-as-a-first-class-outcome.md)
//! for why both prior arts had to retrofit it — PyTorch/XLA with a tri-state
//! lowering step, Burn with `is_refusal()` alongside `is_device_poisoned()` on an
//! error type that had no room for it.
//!
//! The failure mode this prevents is specific and common: a vendor that has no
//! way to decline accepts every subgraph it is handed, then fails at run time on
//! the ones it should have sent elsewhere. The trait makes declining cheaper than
//! pretending, and [`Backend::supports`] means a planner can
//! avoid the question entirely.
//!
//! # Objects and lifetimes
//!
//! [`Executable`] is an opaque, owned, type-erased handle. Not a borrowed graph,
//! not a `dyn Trait` object the caller downcasts — a plugin may need to hold
//! device resources whose lifetime Strata cannot see, so the handle owns them and
//! releases on `Drop`.
//!
//! `Buffer` and `Stream` are deliberately absent from this module's required
//! surface. A backend takes its input as an owned `Buffer` it allocates itself
//! from the shape and dtype in the graph; the ABI's buffer-sharing entry points
//! arrive in B2 with the rest of the plugin surface. Naming them here would mean
//! designing ownership across a boundary that is not specified yet.

use crate::capabilities::{StrataSupport, SupportLevel};
use crate::error::Error;
use crate::graph::Graph;
use crate::id::ValueId;

/// The Strata IR version this build speaks.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StrataVersion {
    /// Major: a change that an older plugin cannot read at all.
    pub major: u16,
    /// Minor: a change an older plugin can read and safely ignore.
    pub minor: u16,
}

impl StrataVersion {
    /// This build's version, from the crate version.
    #[must_use]
    pub const fn current() -> Self {
        Self { major: 0, minor: 1 }
    }

    /// Whether a plugin at `self` can read a subgraph produced at `required`.
    ///
    /// Same major, and `self.minor` at least `required.minor`. This is the whole
    /// compatibility rule for the IR wire format, and it is checked at plugin load
    /// rather than left to fail mid-compile.
    #[must_use]
    pub const fn can_read(self, required: Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl core::fmt::Display for StrataVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// A plugin's own version, which is independent of the Strata version it targets.
///
/// Independent on purpose: a vendor ships plugin 4.2.0 to support Strata 0.1, and
/// a host must be able to say "plugin 4.1.0 does not have the flash-attention
/// pattern" without conflating that with "this plugin is for a different Strata".
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PluginVersion {
    /// The vendor's version, semantic.
    pub major: u16,
    /// The vendor's minor.
    pub minor: u16,
    /// The vendor's patch.
    pub patch: u16,
    /// The Strata version this plugin was built against.
    pub targets: StrataVersion,
}

impl PluginVersion {
    /// A plugin version.
    #[must_use]
    pub const fn new(major: u16, minor: u16, patch: u16, targets: StrataVersion) -> Self {
        Self {
            major,
            minor,
            patch,
            targets,
        }
    }
}

impl core::fmt::Display for PluginVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}.{}.{} (strata {})",
            self.major, self.minor, self.patch, self.targets
        )
    }
}

/// An opaque compiled subgraph.
///
/// Deliberately not `Debug`-printable with its contents, and deliberately not
/// cloneable: two handles to the same compiled artefact would mean two owners of
/// device memory with no defined release order. The handle is move-only and
/// releases on drop.
pub trait Executable: Send {
    /// The backend's name for this artefact, for logs and cache keys.
    ///
    /// Distinct from the graph's name and distinct from a *launch sequence* — see
    /// [ADR-0014](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0014-two-distinct-artefacts-get-two-distinct-names.md).
    /// A compiled subgraph is what the backend produced; a launch sequence is what
    /// the device recorded, and conflating them is how a cache ends up keyed on the
    /// wrong artefact.
    fn label(&self) -> &str;

    /// The graph this executable was compiled from.
    ///
    /// Borrowed, not owned: the plugin may not keep the IR around, and the host
    /// needs it to map outputs back to buffers.
    fn graph(&self) -> &Graph;

    /// The outputs this executable produces, matching [`Graph::outputs`] order.
    fn outputs(&self) -> &[ValueId];

    /// Execute against the host's value arena.
    ///
    /// Reads its operands from `arena` and writes its results back into it,
    /// replacing whatever was there. Mutating one arena rather than taking inputs
    /// and returning outputs is deliberate: a partition's boundary values are shared
    /// with its neighbours, and copying them out and back in per step is where a
    /// multi-partition run quietly picks up a stale boundary value.
    ///
    /// # Errors
    ///
    /// With [`ErrorCode::InvalidArgument`](crate::ErrorCode::InvalidArgument) if a
    /// required operand is missing or
    /// unreadable by this backend. Not with
    /// [`ErrorCode::Declined`](crate::ErrorCode::Declined): declining is
    /// what [`Backend::compile`] does, and a compiled executable that declines at run
    /// time has already wasted the work.
    fn run(&self, arena: &mut crate::arena::Arena) -> Result<(), Error>;
}

/// An owned, type-erased handle to an [`Executable`].
///
/// The host stores these without knowing the plugin's concrete type, which is
/// what makes a plugin loadable at run time rather than linked at build time.
pub struct ExecutableId {
    inner: Box<dyn Executable>,
    backend: String,
}

impl ExecutableId {
    /// Take ownership of a concrete executable from a backend.
    #[must_use]
    pub fn new<E: Executable + 'static>(backend: impl Into<String>, inner: E) -> Self {
        Self {
            inner: Box::new(inner),
            backend: backend.into(),
        }
    }

    /// The backend that produced this.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// The executable's label.
    #[must_use]
    pub fn label(&self) -> &str {
        self.inner.label()
    }

    /// The graph it was compiled from.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        self.inner.graph()
    }

    /// Its outputs.
    #[must_use]
    pub fn outputs(&self) -> &[ValueId] {
        self.inner.outputs()
    }

    /// Execute it against the host arena.
    ///
    /// Forwarded rather than left to a downcast: the whole point of the handle is that
    /// the host does not know the plugin's concrete type, and a `run` that required
    /// one would defeat it.
    ///
    /// # Errors
    ///
    /// Whatever the backend reports. See [`Executable::run`].
    pub fn run(&self, arena: &mut crate::arena::Arena) -> Result<(), Error> {
        self.inner.run(arena)
    }
}

impl core::fmt::Debug for ExecutableId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutableId")
            .field("backend", &self.backend)
            .field("label", &self.inner.label())
            .finish_non_exhaustive()
    }
}

/// The outcome of a compile attempt.
///
/// Not a `Result<ExecutableId, Error>` alias, because the *successful* and the
/// *declined* arms carry different things and the declined arm is not an error at
/// all. A `Result` would make `Declined` an `Err(Error)` with a magic code, and
/// every caller would grow a `match` that special-cases it — which is the same
/// thing PyTorch and Burn ended up doing with their error enums.
#[derive(Debug)]
pub enum CompileOutcome {
    /// The backend took it.
    Compiled(Box<ExecutableId>),
    /// The backend will not take it, and says why.
    Declined {
        /// The backend's name, for the host's attribution.
        backend: String,
        /// The human-readable reason. Never machine-parsed; see
        /// [`PatternSupport::reason`](crate::capabilities::PatternSupport::reason).
        reason: String,
    },
}

impl CompileOutcome {
    /// Whether this is a compiled executable.
    #[must_use]
    pub const fn is_compiled(&self) -> bool {
        matches!(self, Self::Compiled(_))
    }

    /// Whether this is a decline.
    #[must_use]
    pub const fn is_declined(&self) -> bool {
        matches!(self, Self::Declined { .. })
    }

    /// The executable, if it compiled.
    #[must_use]
    pub fn compiled(self) -> Option<Box<ExecutableId>> {
        match self {
            Self::Compiled(inner) => Some(inner),
            Self::Declined { .. } => None,
        }
    }

    /// The decline's backend and reason, if it declined.
    #[must_use]
    pub fn declined(self) -> Option<(String, String)> {
        match self {
            Self::Declined { backend, reason } => Some((backend, reason)),
            Self::Compiled(_) => None,
        }
    }

    /// Whether the partitioner should try a different backend after this outcome.
    ///
    /// True for a decline; false for a compiled executable. A hard error is *not*
    /// in this type — it comes back as `Err` from [`Backend::compile`] — so this
    /// question has only one interesting answer.
    #[must_use]
    pub const fn should_try_elsewhere(&self) -> bool {
        matches!(self, Self::Declined { .. })
    }
}

/// The pull query result, as returned by [`Backend::supports`].
///
/// [`SupportLevel`] carried by name so a plugin author sees, in the trait
/// definition, that this is the capability vocabulary rather than a bare integer.
pub type Support = SupportLevel;

/// What a vendor plugin implements.
///
/// Four required methods, no more. Everything else a backend might want to expose
/// is either in the capability blob (data) or is not in B1. Keeping the surface
/// this small is what makes a plugin implementable, and what makes the boundary
/// stable enough to write a version-0 ABI against.
pub trait Backend: Send + Sync {
    /// The backend's stable name, matching `CapabilitySet::backend_name`.
    fn name(&self) -> &str;

    /// The plugin's version.
    fn plugin_version(&self) -> PluginVersion;

    /// The capabilities, as data.
    fn capabilities(&self) -> &StrataSupport;

    /// The pull query: can this backend handle this pattern, for this dtype and
    /// shape class?
    ///
    /// The default forwards to the capability blob, which is what makes
    /// [`StrataSupport::supports`] and the two descriptions of a backend agree.
    /// An implementation overrides this only when the answer depends on state the
    /// blob cannot hold — which is the extension
    /// [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md)
    /// anticipates, and overriding it loses diffability, so it needs a reason.
    fn supports(
        &self,
        pattern_id: &str,
        dtype: crate::dtype::DType,
        shape_class: crate::shape::ShapeClass,
    ) -> Support {
        self.capabilities().supports(pattern_id, dtype, shape_class)
    }

    /// Compile a subgraph.
    ///
    /// The contract, in full:
    ///
    /// - `Ok(Compiled(..))` — this backend owns the artefact; it must be able to
    ///   run it.
    /// - `Ok(Declined { .. })` — this backend understood the request and will not
    ///   take it. It must not have partially compiled anything, and it must leave
    ///   the device usable. Declining after allocating and freeing is fine;
    ///   declining and leaving the device wedged is a bug.
    /// - `Err(_)` with [`ErrorCode::Declined`](crate::ErrorCode::Declined) — **wrong**.
    ///   That code arrives inside
    ///   the `Declined` arm; putting it in an `Err` is the mistake this design
    ///   exists to make impossible.
    /// - `Err(_)` otherwise — the call is malformed, a limit was hit, or the
    ///   device broke. The caller surfaces it rather than repartitioning, except for
    ///   [`ErrorCode::ResourceExhausted`](crate::ErrorCode::ResourceExhausted), which a
    ///   planner may legitimately retry
    ///   on a smaller subgraph.
    fn compile(&self, subgraph: &Graph, pattern_id: &str) -> Result<CompileOutcome, Error>;
}

#[cfg(test)]
mod tests {
    use super::{
        Backend, CompileOutcome, Executable, ExecutableId, PluginVersion, StrataSupport,
        StrataVersion, SupportLevel,
    };
    use crate::capabilities::{CapabilitySet, DTypeMask, DeclaredLimits, PatternSupport};
    use crate::dtype::{DType, ElementType};
    use crate::error::{Error, ErrorCode};
    use crate::graph::{Graph, GraphBuilder};
    use crate::id::ValueId;
    use crate::shape::{Shape, ShapeClass};

    /// A backend that compiles everything, for the happy-path tests.
    struct AcceptAll {
        support: StrataSupport,
    }

    impl AcceptAll {
        fn new() -> Self {
            let capabilities = CapabilitySet::empty("accept-all")
                .with_dtype_mask(DTypeMask::all())
                .with_pattern(PatternSupport::new(
                    "ref::matmul",
                    DType::plain(ElementType::F32),
                    ShapeClass::Matrix,
                    SupportLevel::Fused,
                ))
                .with_limits(DeclaredLimits {
                    max_rank: Some(8),
                    ..DeclaredLimits::default()
                });
            Self {
                support: StrataSupport::new(capabilities),
            }
        }
    }

    struct Fake {
        graph: Graph,
    }

    impl Executable for Fake {
        fn label(&self) -> &str {
            "accept-all::0"
        }

        fn graph(&self) -> &Graph {
            &self.graph
        }

        fn outputs(&self) -> &[ValueId] {
            self.graph.outputs()
        }

        fn run(&self, _arena: &mut crate::arena::Arena) -> Result<(), Error> {
            Ok(())
        }
    }

    impl Backend for AcceptAll {
        fn name(&self) -> &str {
            "accept-all"
        }

        fn plugin_version(&self) -> PluginVersion {
            PluginVersion::new(1, 2, 3, StrataVersion::current())
        }

        fn capabilities(&self) -> &StrataSupport {
            &self.support
        }

        fn compile(&self, subgraph: &Graph, _pattern_id: &str) -> Result<CompileOutcome, Error> {
            Ok(CompileOutcome::Compiled(Box::new(ExecutableId::new(
                self.name(),
                Fake {
                    graph: subgraph.clone(),
                },
            ))))
        }
    }

    /// A backend that declines everything, for the decline tests.
    struct DeclineAll;

    impl Backend for DeclineAll {
        fn name(&self) -> &str {
            "decline-all"
        }

        fn plugin_version(&self) -> PluginVersion {
            PluginVersion::new(0, 1, 0, StrataVersion::current())
        }

        fn capabilities(&self) -> &StrataSupport {
            // Constructed inline: a decline-only backend genuinely has no
            // capabilities worth caching.
            static EMPTY: std::sync::OnceLock<StrataSupport> = std::sync::OnceLock::new();
            EMPTY.get_or_init(|| StrataSupport::new(CapabilitySet::empty("decline-all")))
        }

        fn compile(&self, _subgraph: &Graph, _pattern_id: &str) -> Result<CompileOutcome, Error> {
            Ok(CompileOutcome::Declined {
                backend: self.name().to_string(),
                reason: "no fp32 kernels in this build".into(),
            })
        }
    }

    fn a_graph() -> Graph {
        // Built through the builder rather than by poking the arenas, so the test
        // graph is one a caller could actually have produced.
        let mut b = GraphBuilder::new("test");
        let x = b
            .input(
                DType::plain(ElementType::F32),
                Shape::new(&[2, 2]).expect("rank 2"),
                "x",
            )
            .expect("x is free");
        b.output(x).expect("not already an output");
        b.build()
    }

    #[test]
    fn the_current_version_reads_itself() {
        let current = StrataVersion::current();
        assert!(current.can_read(current));
        assert_eq!(current.to_string(), "0.1");
    }

    #[test]
    fn a_minor_mismatch_is_unreadable_in_one_direction_only() {
        let v0_1 = StrataVersion { major: 0, minor: 1 };
        let v0_2 = StrataVersion { major: 0, minor: 2 };
        let v1_0 = StrataVersion { major: 1, minor: 0 };

        // A newer plugin can read an older IR; an older plugin cannot read a
        // newer one.
        assert!(v0_2.can_read(v0_1));
        assert!(!v0_1.can_read(v0_2));
        // A major bump is a hard boundary in both directions.
        assert!(!v1_0.can_read(v0_1));
        assert!(!v0_1.can_read(v1_0));
    }

    #[test]
    fn a_plugin_version_names_both_versions() {
        let v = PluginVersion::new(4, 2, 0, StrataVersion::current());
        assert_eq!(v.to_string(), "4.2.0 (strata 0.1)");
        assert_eq!(v.targets, StrataVersion::current());
    }

    #[test]
    fn compile_can_return_a_compiled_executable() {
        let backend = AcceptAll::new();
        let g = a_graph();
        let outcome = backend.compile(&g, "ref::matmul").expect("no hard error");
        assert!(outcome.is_compiled());
        assert!(!outcome.is_declined());
        assert!(!outcome.should_try_elsewhere());

        let executable = outcome.compiled().expect("compiled");
        assert_eq!(executable.backend(), "accept-all");
        assert_eq!(executable.label(), "accept-all::0");
        assert_eq!(executable.graph().node_count(), g.node_count());
    }

    #[test]
    fn compile_can_decline_and_the_outcome_is_not_an_error() {
        // The whole design in one test: a decline comes back inside `Ok`, not as
        // an `Err`. No error code to inspect, no `is_refusal()` to reimplement.
        let backend = DeclineAll;
        let g = a_graph();
        let outcome = backend.compile(&g, "ref::matmul").expect("not an error");
        assert!(outcome.is_declined());
        assert!(!outcome.is_compiled());
        assert!(outcome.should_try_elsewhere());

        let (name, reason) = outcome.declined().expect("declined");
        assert_eq!(name, "decline-all");
        assert_eq!(reason, "no fp32 kernels in this build");
    }

    #[test]
    fn a_decline_is_distinguishable_from_an_err() {
        let declined = DeclineAll.compile(&a_graph(), "ref::matmul");
        assert!(declined.is_ok(), "a decline is inside Ok");

        let failed: Result<CompileOutcome, Error> = Err(Error::from_backend(
            ErrorCode::Internal,
            "decline-all",
            "device fell off the bus",
        ));
        assert!(failed.is_err());
        // And the difference is not cosmetic: only the decline means "try
        // elsewhere", while the error poisons the device.
        assert!(ErrorCode::Internal.is_device_poisoning());
        assert!(!ErrorCode::Declined.is_device_poisoning());
    }

    #[test]
    fn a_declining_backend_stays_declining() {
        let backend = DeclineAll;
        assert!(backend.compile(&a_graph(), "x").expect("ok").is_declined());
        assert!(backend.compile(&a_graph(), "y").expect("ok").is_declined());
    }

    #[test]
    fn supports_defaults_to_the_capability_blob() {
        // The two descriptions of a backend must agree, so the default forwards
        // rather than duplicating the lookup.
        let backend = AcceptAll::new();
        let dtype = DType::plain(ElementType::F32);
        assert_eq!(
            backend.supports("ref::matmul", dtype, ShapeClass::Matrix),
            SupportLevel::Fused
        );
        assert_eq!(
            backend.supports("fused::unknown", dtype, ShapeClass::Matrix),
            SupportLevel::Refuse
        );
    }

    #[test]
    fn a_backend_reports_its_own_name_and_version() {
        let backend = AcceptAll::new();
        assert_eq!(backend.name(), "accept-all");
        assert_eq!(backend.plugin_version().to_string(), "1.2.3 (strata 0.1)");
        assert_eq!(
            backend.capabilities().capabilities().backend_name,
            "accept-all"
        );
    }

    #[test]
    fn an_executable_handle_is_debuggable_without_its_contents() {
        // Debug exists for diagnostics; the artefact inside is opaque and must
        // not be printed.
        let outcome = AcceptAll::new()
            .compile(&a_graph(), "ref::matmul")
            .expect("ok");
        let executable = outcome.compiled().expect("compiled");
        let text = format!("{executable:?}");
        assert!(text.contains("accept-all"), "{text}");
        assert!(
            text.contains(".."),
            "expected finish_non_exhaustive: {text}"
        );
    }

    #[test]
    fn a_decline_carries_the_backend_name_for_attribution() {
        // A multi-backend host needs to say *which* one declined without tracking
        // it separately.
        let (name, _) = DeclineAll
            .compile(&a_graph(), "ref::matmul")
            .expect("ok")
            .declined()
            .expect("declined");
        assert!(!name.is_empty());
    }
}
