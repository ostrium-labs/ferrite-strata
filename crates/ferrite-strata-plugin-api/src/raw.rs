//! The C ABI, as `#[repr(C)]`.
//!
//! Every type here is what crosses the `dlopen` boundary, and a vendor in any
//! language sees only this — the mirror of `include/strata_plugin.h`. Nothing in
//! this module is a Rust trait object, a generic, or a `String`; there is no
//! layout Rust gets to choose.
//!
//! # Discriminants are assigned here, not inherited
//!
//! [`StrataElementType`], [`StrataShapeClass`], [`StrataSupportLevel`] and
//! [`StrataOpKind`] declare their own `#[repr(u32)]` values rather than mirroring
//! a Rust enum from `ferrite-strata`. Rust's enum layout is explicitly unstable,
//! and an ABI that inherits it is an ABI that changes when the enum grows a
//! variant — which would break every plugin built against the previous release.
//! The mapping is one function per type, in [`crate::wire`], and it is
//! fallible in the direction that matters: a newer plugin can send a tag this
//! build has never heard of, and the host reports that rather than guessing.
//!
//! # `struct_size` on every extensible struct
//!
//! Each struct that may grow carries its own size. A reader compares it against
//! its own `size_of` and refuses anything shorter; that is what lets a v1 host
//! read a v2 plugin's `StrataCompileArgs` while ignoring the fields it does not
//! know. The rule is: **unknown trailing fields are ignorable, unknown
//! leading fields are not.** Every struct here puts the size first for that
//! reason.

use core::ffi::c_void;

/// The ABI version a plugin speaks.
///
/// Split major/minor rather than one integer, because the two halves have
/// different failure modes: a major mismatch means the plugin cannot be read at
/// all and must be re-`dlopen`ed against a different host, while a minor
/// mismatch means every field the host knows about is still at the same offset
/// and the plugin simply has more.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StrataAbiVersion {
    /// Incompatible change. Must match exactly.
    pub major: u32,
    /// Additive change. A host accepts a plugin with a *higher* minor.
    pub minor: u32,
}

impl StrataAbiVersion {
    /// The ABI this build implements.
    pub const CURRENT: Self = Self { major: 0, minor: 1 };
}

/// Kind tag on an extension in the chain.
///
/// An open `u32` rather than a closed enum: an extension type is defined by
/// whoever adds it, and a host that meets one it does not know skips it rather
/// than failing the load. See [`StrataExtBase`].
pub type StrataExtType = u32;

/// The header every extension in the chain begins with.
///
/// PJRT's extension chain, copied because it is the one piece of PJRT that is
/// genuinely reusable: a linked list of variable-sized structs where the reader
/// checks `struct_size` before touching the rest.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataExtBase {
    /// Size of *this* struct including the header. A reader that does not
    /// recognise `ty` skips exactly this many bytes.
    pub struct_size: usize,
    /// The extension's kind. Unrecognised values are skipped, not fatal.
    pub ty: StrataExtType,
    /// The next extension, or null.
    pub next: *mut StrataExtBase,
}

/// Sentinel for the end of an extension chain.
pub const STRATA_EXT_NONE: StrataExtType = 0;

/// The extension carrying the vendor's own plugin version.
///
/// An extension rather than a field on [`StrataPluginV1`], because the vtable is
/// read once at load by a host that may be older than the plugin, and a field
/// appended to the vtable would be past the end of what that host knows the
/// struct's size to be. The chain is walked by declared length for exactly this
/// reason.
pub const STRATA_EXT_PLUGIN_VERSION: StrataExtType = 1;

/// The plugin version extension's payload.
/// `u16` for each component because a vendor version past 65535 is a different
/// problem, and two extra bytes saved per field buys room for two more fields
/// before the extension needs its own size bump.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataPluginVersionExt {
    /// The chain header. `ty` must be [`STRATA_EXT_PLUGIN_VERSION`].
    pub base: StrataExtBase,
    /// The vendor's major version.
    pub major: u16,
    /// The vendor's minor version.
    pub minor: u16,
    /// The vendor's patch version.
    pub patch: u16,
    /// The Strata ABI version this plugin was built against.
    pub targets: StrataAbiVersion,
}

/// The extension carrying a human-readable message.
///
/// The only payload in ABI v1, and the reason [`StrataError`] is opaque: the
/// message is reached through
/// [`StrataPluginV1::error_payload`]`'s visitor, so the plugin owns the bytes and
/// can grow the payload without the host's layout changing.
pub const STRATA_EXT_MESSAGE: StrataExtType = 2;

/// The message payload's body.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataMessageExt {
    /// The chain header. `ty` must be [`STRATA_EXT_MESSAGE`].
    pub base: StrataExtBase,
    /// A NUL-terminated UTF-8 message, owned by whoever produced the error.
    pub message: *const core::ffi::c_char,
}

/// An error, opaque to the host.
///
/// Deliberately not a struct the host can read fields out of. PJRT's error object
/// is opaque with accessors, and the reason is not aesthetics: an error is the
/// one thing a plugin hands back when it has already failed, so the host must
/// not depend on its layout, and the plugin must be free to grow it. The
/// message is reached through [`StrataPluginV1::error_payload`] and freed
/// through [`StrataPluginV1::error_free`], so the plugin owns its own storage.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataError {
    _opaque: [u8; 0],
}

/// Callback that receives one payload.
///
/// Modelled on PJRT's `PJRT_Error_ForEachPayload`, in the smaller form: a visitor
/// rather than an array, because the payload set is open and the allocation
/// belongs to whoever built the error.
///
/// Returns non-zero to stop iteration.
pub type StrataPayloadVisitor =
    Option<unsafe extern "C" fn(ctx: *mut c_void, payload: *mut StrataExtBase) -> i32>;

/// Element storage type, as a wire tag.
///
/// Values are assigned by this ABI, not by the `ElementType` enum's layout. The
/// `..=` gap at 1 is intentional: it was an 8-bit float variant removed before
/// the ABI was frozen, and leaving the hole stops a retired tag being reused by
/// accident.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrataElementType {
    /// 8-bit IEEE 754, 5 exponent bits.
    F8E5M2 = 0,
    /// 8-bit IEEE 754, 4 exponent bits.
    F8E4M3 = 2,
    /// IEEE binary16.
    F16 = 3,
    /// IEEE binary32.
    F32 = 4,
    /// IEEE binary64.
    F64 = 5,
    /// Signed 8-bit integer.
    I8 = 6,
    /// Signed 16-bit integer.
    I16 = 7,
    /// Signed 32-bit integer.
    I32 = 8,
    /// Signed 64-bit integer.
    I64 = 9,
    /// Unsigned 8-bit integer.
    U8 = 10,
    /// Boolean.
    Bool = 11,
    /// 8-bit microscaling exponent scale.
    F8E8M0 = 12,
    /// 4-bit float, 2 exponent bits, 1 mantissa bit.
    F4E2M1 = 13,
    /// Brain floating point, the same exponent range as `F32` with 8 mantissa bits.
    BF16 = 14,
}

/// How a tensor's elements are scaled.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrataQuantKind {
    /// The bits are the value.
    None = 0,
    /// Per-block affine: one scale and one zero-point per block.
    Block = 1,
    /// Per-block `E8M0` exponent, no zero-point.
    Microscaling = 2,
}

/// The scaling convention attached to a dtype.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StrataQuantSpec {
    /// Which convention.
    pub kind: StrataQuantKind,
    /// Elements per block. Meaningless unless `kind` is not
    /// [`StrataQuantKind::None`].
    pub block_size: u32,
    /// Whether the zero-point is fixed at zero. Only meaningful for
    /// [`StrataQuantKind::Block`].
    pub symmetric: u8,
}

/// A complete element description: storage bits plus scaling.
///
/// One struct rather than a bare element tag because the ABI has no room for the
/// distinction PJRT also lacks: a bare `i8` tensor and a symmetrically quantised
/// `i8` tensor are the same eight bits and completely different numbers, and a
/// vendor that cannot tell them apart will scale a weight twice. Carrying
/// `quant` here is what makes the boundary honest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StrataDType {
    /// The storage bits.
    pub element: StrataElementType,
    /// The scaling convention.
    pub quant: StrataQuantSpec,
}

/// The kind half of a [`StrataShapeClass`].
///
/// Mirrors `ferrite_strata::ShapeClass` variant for variant. The split into a
/// kind plus a rank is because the core enum has one data-carrying variant, and
/// an ABI that flattened it into five tags would have to invent tags for ranks it
/// does not know.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrataShapeClassKind {
    /// Rank 0.
    Scalar = 0,
    /// Some dimension is zero.
    Empty = 1,
    /// Leading dimension 1, at any rank: a matmul operand, a bias, one token.
    OuterProduct = 2,
    /// Rank 1 with a leading dimension above 1.
    Vector = 3,
    /// Rank 2.
    Matrix = 4,
    /// Rank 3.
    Batch3 = 5,
    /// Rank 4, the transformer-activation shape.
    Batch4 = 6,
    /// Rank 5 or more, with the rank carried alongside.
    HigherRank = 7,
}

/// The coarse shape classification capability queries are answered over.
///
/// A vendor never sees a concrete shape in [`StrataPluginV1::supports`]. It sees
/// the class, because "can you do a fused attention over 8 heads" is a question
/// about shapes as a family, and answering it for one shape would force a
/// planner to probe every shape it might ever meet.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StrataShapeClass {
    /// Which family.
    pub kind: StrataShapeClassKind,
    /// The rank, for [`StrataShapeClassKind::HigherRank`] only; zero otherwise.
    pub rank: u8,
    /// Reserved, must be zero. Keeps the struct two words so the C header's
    /// layout does not depend on the host compiler's padding choices.
    pub reserved: [u8; 2],
}

/// How well a backend can run a pattern.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum StrataSupportLevel {
    /// Will not run it.
    Refuse = 0,
    /// Runs it as its constituent ops.
    Fallback = 1,
    /// Runs it as one fused unit.
    Fused = 2,
}

/// The wire tag for an op.
///
/// Extensible by tag rather than by a Rust enum variant, for the reason
/// [`StrataElementType`] assigns its own values: a plugin built against an
/// older header must keep working when Strata learns a new reference op. The
/// tags below 16 are the reference ops `ferrite-strata` understands;
/// [`StrataOpKind::Custom`] is the escape hatch that carries a vendor-owned
/// pattern id as data.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StrataOpKind {
    /// A graph input. Consumes nothing.
    Input = 0,
    /// Elementwise add.
    Add = 1,
    /// Elementwise multiply.
    Mul = 2,
    /// Batched matrix multiply.
    MatMul = 3,
    /// Natural log.
    Log = 4,
    /// Exponential.
    Exp = 5,
    /// Row-wise softmax.
    Softmax = 6,
    /// Layer normalisation.
    LayerNorm = 7,
    /// GELU, exact form.
    Gelu = 8,
    /// `x @ w + b`.
    Linear = 9,
    /// Reshape to a carried target shape.
    Reshape = 10,
    /// Transpose by an explicit permutation.
    Transpose = 11,
    /// Broadcast along new leading axes.
    Expand = 12,
    /// Slice along an axis.
    Slice = 13,
    /// A vendor-owned op, named by `pattern_id`.
    ///
    /// Not a hole in the type system: it round-trips through the wire format,
    /// it is what a fused subgraph looks like coming back from a vendor, and its
    /// id is the same string the capability table is keyed by.
    Custom = 14,
}

impl StrataOpKind {
    /// Decode a tag, or `None` for one this build does not know.
    ///
    /// `None` is a hard failure on the way *in*: an unknown op tag means the
    /// blob is not readable, because unlike an unknown *section* there is no
    /// length field to skip past it with.
    #[must_use]
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Input),
            1 => Some(Self::Add),
            2 => Some(Self::Mul),
            3 => Some(Self::MatMul),
            4 => Some(Self::Log),
            5 => Some(Self::Exp),
            6 => Some(Self::Softmax),
            7 => Some(Self::LayerNorm),
            8 => Some(Self::Gelu),
            9 => Some(Self::Linear),
            10 => Some(Self::Reshape),
            11 => Some(Self::Transpose),
            12 => Some(Self::Expand),
            13 => Some(Self::Slice),
            14 => Some(Self::Custom),
            _ => None,
        }
    }
}

/// An owned byte string handed to the host.
///
/// Length-carrying and NUL-terminated, because a plugin that returns one of
/// these across the boundary cannot know whether the host is reading it by
/// `strlen` or by `len`, and `str::from_utf8_unchecked` on the wrong answer is
/// how a display name becomes a heap overread. Both are present so neither
/// question is load-bearing.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataBlob {
    /// Payload length in bytes, excluding the terminator.
    pub len: usize,
    /// Payload. Owned by whoever produced this; released via
    /// [`StrataPluginV1::free_blob`].
    pub data: *const u8,
    /// NUL terminator, so the payload can also be read as a C string when it is
    /// known to be text.
    pub nul: u8,
}

/// A capability query, as the host sends it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataPatternQuery {
    /// Size of this struct, for the forward-compatibility rule.
    pub struct_size: usize,
    /// The pattern id being asked about. Vendor-owned extensible string.
    pub pattern_id: *const u8,
    /// Length of `pattern_id`, excluding a NUL terminator.
    pub pattern_id_len: usize,
    /// The dtype the answer applies to.
    pub dtype: StrataDType,
    /// The shape class the answer applies to.
    pub shape_class: StrataShapeClass,
}

/// The versioned wire encoding of a subgraph.
///
/// Not `const uint8_t*` and length: a pointer into a plugin's address space
/// says nothing about whether the host may outlive it, and the host must be able
/// to cache a subgraph across runs. This header is what makes it
/// self-describing — see [`crate::wire`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataWireBlob {
    /// Magic, [`crate::raw::MAGIC`].
    pub magic: u64,
    /// The wire format major. Must match the host's.
    pub format_major: u32,
    /// The wire format minor. A host accepts a blob whose minor it does not
    /// know, and rejects a major it does not.
    pub format_minor: u32,
    /// Payload length in bytes, following this header.
    pub payload_len: u64,
    /// The payload, owned by whoever produced this blob.
    ///
    /// A plugin's capability blob lives exactly as long as the host says it does:
    /// the host copies it and calls `free_blob` immediately. That is what stops a
    /// plugin from handing back a pointer into memory it is about to reuse.
    pub data: *const u8,
    /// A NUL byte, so a payload known to be text can be read as a C string without
    /// a second length field. Always present and always zero, and never counted in
    /// `payload_len`.
    pub nul: u8,
}

/// Arguments to [`StrataPluginV1::compile`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataCompileArgs {
    /// Size of this struct, for the forward-compatibility rule.
    pub struct_size: usize,
    /// The subgraph, in the wire format.
    pub subgraph: StrataWireBlob,
    /// The pattern id the host thinks this subgraph is, or null if the host is
    /// offering an unpartitioned graph. Advisory: the host has already asked
    /// via [`StrataPluginV1::supports`], and a plugin may decline anyway.
    pub pattern_id: *const u8,
    /// Length of `pattern_id`.
    pub pattern_id_len: usize,
    /// The capability blob the host advertised to itself, so the plugin can
    /// compile against the shape of the host it was handed.
    pub capabilities: StrataWireBlob,
    /// Do everything a real compile does except dispatch.
    ///
    /// CubeCL's `LaunchMode::Skip` discipline, and it is what makes a partition
    /// costable: the planner can ask "would this actually work, and how long
    /// would it take" without a device allocation or a launch. A plugin that
    /// ignores this flag still works, and only loses the ability to be planned
    /// rather than merely tried.
    pub probe: u8,
    /// Reserved, must be zero. Present so a host can be sure it did not leave
    /// garbage in a field the plugin will read.
    pub reserved: [u8; 7],
}

/// One buffer operand or result in a [`StrataRunArgs`] entry.
///
/// `StrataBuffer` is a handle the plugin allocated via
/// [`StrataPluginV1::alloc`], never a raw pointer into host memory. That is a
/// real restriction rather than an accident: the whole premise is that the
/// vendor's memory may be device memory the host cannot address, and a boundary
/// that accepted a raw pointer would make that premise false by construction.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataBufferRef {
    /// The buffer handle, or null to mean "no buffer", which is how an absent
    /// optional operand (a `Linear` with no bias) is expressed.
    pub buffer: *mut StrataBuffer,
    /// Which value this is, as an index into the executable's value arena. The
    /// host and the plugin agree on the arena because the plugin decoded the
    /// subgraph it was compiled from.
    pub value_index: u32,
    /// Reserved, must be zero.
    pub reserved: u32,
}

/// Arguments to [`StrataPluginV1::run`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataRunArgs {
    /// Size of this struct, for the forward-compatibility rule.
    pub struct_size: usize,
    /// The executable to run.
    pub executable: *const StrataExecutable,
    /// Number of entries in `args`.
    pub arg_count: u32,
    /// Operands and destinations, indexed by `value_index`.
    pub args: *const StrataBufferRef,
    /// The arena to write results into, for a host that owns the memory.
    ///
    /// Null when every result is a plugin-allocated buffer, which is the normal
    /// case. Non-null when the host has buffers the plugin should fill, which is
    /// how a plugin's device memory is read back.
    pub host_arena: *mut StrataArenaView,
}

/// A read/write view of host-owned memory, for a plugin to fill.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataArenaView {
    /// Number of entries in `entries`.
    pub entry_count: u32,
    /// The entries.
    pub entries: *mut StrataArenaEntryView,
}

/// One value in a [`StrataArenaView`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataArenaEntryView {
    /// Element count, so the plugin can bounds-check before writing.
    pub element_count: u64,
    /// The dtype, which says how many bytes per element.
    pub dtype: StrataDType,
    /// Pointer to the first element, host-owned and valid for the call.
    pub data: *mut u8,
}

/// A buffer the plugin allocated. Opaque to the host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataBuffer {
    _opaque: [u8; 0],
}

/// How to allocate a buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataAllocDesc {
    /// Size of this struct, for the forward-compatibility rule.
    pub struct_size: usize,
    /// The dtype, which fixes the element width.
    pub dtype: StrataDType,
    /// Dimension count.
    pub rank: u32,
    /// Reserved, must be zero.
    pub reserved: u32,
    /// The dimensions.
    pub dims: *const u64,
}

/// A synchronisation event. Opaque to the host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataEvent {
    _opaque: [u8; 0],
}

/// A compiled subgraph. Opaque to the host.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataExecutable {
    _opaque: [u8; 0],
}

/// The one symbol a plugin exports.
///
/// Everything else is reached through this struct, so a plugin has exactly one
/// name to get right and a host has exactly one lookup to fail.
pub const STRATA_PLUGIN_ENTRY_SYMBOL: &str = "strata_plugin_v1";

/// `b"STRATAG1"` as a little-endian `u64`.
///
/// First so a blob handed to the wrong function fails on four bytes of compare
/// rather than on a nonsense length, and second because it is what
/// [`crate::wire::Parsed::parse`] checks before trusting anything else in the
/// header.
pub const MAGIC: u64 = u64::from_le_bytes(*b"STRATAG1");

/// A `StrataWireBlob`'s header length in bytes.
pub const WIRE_HEADER_LEN: usize = 8 + 4 + 4 + 8;

/// The plugin vtable.
///
/// Every function pointer is `Option`, so a v1 host reading a v2 plugin skips
/// the entries it does not know instead of calling through a null pointer. The
/// first eight entries are mandatory and a load fails if any of them is null;
/// the rest are optional and a missing one is a feature the plugin does not
/// have.
///
/// # Ownership, in one paragraph
///
/// The host owns every pointer it passes in and frees them with the
/// `free_*` entry points that produced them. The plugin owns every pointer it
/// returns and the host frees those with the matching `free_*` entry. Nothing
/// is freed across the boundary by anything other than its allocator, because a
/// plugin linked against a different allocator than the host will otherwise
/// corrupt the heap in a way that looks like a memory bug in somebody's kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StrataPluginV1 {
    /// Size of this struct. A host that reads a shorter vtable stops after the
    /// last entry that fits, which is how a v1 host survives a v2 plugin that
    /// appended to the end.
    pub struct_size: usize,
    /// The extension chain, or null.
    pub extension_start: *mut StrataExtBase,
    /// The ABI version this plugin speaks.
    pub abi_version: StrataAbiVersion,

    // ---- error handling, mandatory ----
    /// Release an error the plugin produced. Never null.
    pub error_free: Option<unsafe extern "C" fn(error: *mut StrataError)>,
    /// The error's code as a [`StrataSupportLevel`]-shaped tag; see
    /// [`crate::status`] for the value set. Never null.
    pub error_code: Option<unsafe extern "C" fn(error: *const StrataError) -> u32>,
    /// Visit the error's payloads. Never null, though it may visit nothing.
    pub error_payload: Option<
        unsafe extern "C" fn(
            error: *const StrataError,
            visitor: StrataPayloadVisitor,
            ctx: *mut c_void,
        ),
    >,

    // ---- capability, mandatory ----
    /// The capability blob, as data. Never null.
    ///
    /// Data rather than a pull query alone, so a host can snapshot, cache, diff
    /// and print it without executing anything on the device. The pointer stays
    /// valid until `free_blob` is called on it.
    pub capabilities:
        Option<unsafe extern "C" fn(ctx: *mut c_void, out: *mut StrataWireBlob) -> i32>,

    /// The pull query: can this plugin run this pattern, for this dtype and shape
    /// class. Never null.
    ///
    /// The blob above answers "what can you do"; this answers "what can you do
    /// *right now*", which is the question a planner asks before committing a
    /// partition. A plugin that is out of memory may answer differently.
    ///
    /// The answer is the raw [`StrataSupportLevel`] *tag*, not the Rust enum, for
    /// the same reason the enum exists at all: a plugin built against a newer
    /// header may answer with a level this build has never heard of, and the
    /// host has to be able to see the tag and refuse it rather than have the
    /// value materialised as an invalid enum discriminant on the way in.
    pub supports:
        Option<unsafe extern "C" fn(ctx: *mut c_void, query: *const StrataPatternQuery) -> u32>,

    // ---- compilation, mandatory ----
    /// Compile a subgraph. Never null.
    ///
    /// Three outcomes, and the middle one is the design: [`crate::status::OK`]
    /// with a `StrataExecutable*` out-param, [`crate::status::DECLINED`] with
    /// nothing allocated, or a hard error with a `StrataError*` out-param. A
    /// plugin that returns a hard error for "not mine" has made the mistake this
    /// ABI exists to make impossible.
    pub compile: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            args: *const StrataCompileArgs,
            out_executable: *mut *mut StrataExecutable,
            out_error: *mut *mut StrataError,
        ) -> i32,
    >,

    // ---- buffers and events ----
    /// Allocate a buffer. Optional; a plugin that rejects `StrataAllocDesc`
    /// with [`crate::status::UNIMPLEMENTED`] is behaving correctly.
    pub alloc: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            desc: *const StrataAllocDesc,
            out_buffer: *mut *mut StrataBuffer,
            out_error: *mut *mut StrataError,
        ) -> i32,
    >,
    /// Release a buffer. Optional; required if `alloc` is present.
    pub buffer_release:
        Option<unsafe extern "C" fn(ctx: *mut c_void, buffer: *mut StrataBuffer) -> i32>,

    /// Copy host bytes into a plugin buffer, blocking until they land.
    ///
    /// `offset` and `len` are in **bytes**, not elements, because a plugin
    /// buffer is not required to be densely element-addressed — a block-scaled
    /// `i8` tensor interleaves its scales beside the data, and an
    /// element-indexed copy would have to know the block size to be correct. The
    /// host computes the byte length from the dtype in the graph, which is the
    /// authority, so the plugin does not get to disagree about a tensor's size.
    ///
    /// Blocking is a deliberate simplification against PJRT, which hands back a
    /// host pointer plus an event and lets the caller overlap the copy. That is
    /// the right end state and the wrong first version: an overlapped path needs
    /// the host to own lifetime across an async boundary, which is the part of
    /// buffer sharing this ABI has not yet settled.
    pub buffer_write: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            buffer: *mut StrataBuffer,
            offset: u64,
            data: *const u8,
            len: u64,
        ) -> i32,
    >,
    /// Copy bytes out of a plugin buffer, blocking until they are valid.
    ///
    /// Same byte-offset contract as [`StrataPluginV1::buffer_write`], and the
    /// same simplification.
    pub buffer_read: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            buffer: *const StrataBuffer,
            offset: u64,
            data: *mut u8,
            len: u64,
        ) -> i32,
    >,
    /// An event recording when a buffer's last use completed. Optional.
    pub buffer_event: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            buffer: *const StrataBuffer,
            out_event: *mut *mut StrataEvent,
        ) -> i32,
    >,
    /// Whether the event has completed. Optional.
    pub event_ready:
        Option<unsafe extern "C" fn(ctx: *mut c_void, event: *const StrataEvent) -> i32>,
    /// Block until the event completes. Optional.
    pub event_await:
        Option<unsafe extern "C" fn(ctx: *mut c_void, event: *const StrataEvent) -> i32>,
    /// Free an event. Optional.
    pub event_free: Option<unsafe extern "C" fn(ctx: *mut c_void, event: *mut StrataEvent) -> i32>,

    // ---- execution, mandatory ----
    /// Run a compiled executable. Never null.
    pub run: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            args: *const StrataRunArgs,
            out_error: *mut *mut StrataError,
        ) -> i32,
    >,

    // ---- teardown, mandatory ----
    /// Release a compiled executable. Never null.
    pub free_executable:
        Option<unsafe extern "C" fn(ctx: *mut c_void, executable: *mut StrataExecutable)>,
    /// Release any blob the plugin produced via `capabilities`. Never null.
    pub free_blob: Option<unsafe extern "C" fn(ctx: *mut c_void, blob: *mut StrataWireBlob)>,
    /// Create the plugin's context, returning it, or null for a stateless plugin.
    ///
    /// Optional, and the reason the entry symbol needs no arguments: PJRT takes a
    /// `PJRT_Client*` from the embedder, which only works because PJRT is linked
    /// into the same process and can agree an allocator. A `dlopen`ed plugin has
    /// no such guarantee, so it makes its own context and hands it over.
    ///
    /// Called exactly once, immediately after the ABI is negotiated. May open a
    /// device, which is why it exists at all rather than the plugin doing that in
    /// a constructor nobody can see the failure of.
    pub plugin_init: Option<unsafe extern "C" fn() -> *mut c_void>,
    /// Release the plugin's `ctx`. Required if `plugin_init` returned non-null.
    ///
    /// Called exactly once, at `dlclose` time or just before.
    pub plugin_free: Option<unsafe extern "C" fn(ctx: *mut c_void)>,

    // ---- ahead-of-time artefacts ----
    /// Serialise a compiled executable to an opaque blob. Optional.
    ///
    /// PJRT's own header says a blob it returns is portable only to a compatible
    /// build and "the serialization is not guaranteed to be stable over time";
    /// that caveat is inherited deliberately rather than papered over, which is
    /// why `abi_compatible` exists as a separate entry rather than being implied
    /// by the blob.
    pub serialize: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            executable: *const StrataExecutable,
            out_blob: *mut StrataWireBlob,
        ) -> i32,
    >,
    /// Load a blob produced by `serialize` on a compatible build. Optional.
    pub deserialize_load: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            blob: *const StrataWireBlob,
            out_executable: *mut *mut StrataExecutable,
        ) -> i32,
    >,
    /// Whether this plugin can load `blob`. Optional.
    ///
    /// Separate from `deserialize_load` because "can you read this" is a
    /// question a cache asks *before* deciding to keep the entry, and answering
    /// it must not allocate device memory.
    pub abi_compatible:
        Option<unsafe extern "C" fn(ctx: *mut c_void, blob: *const StrataWireBlob) -> i32>,
}
