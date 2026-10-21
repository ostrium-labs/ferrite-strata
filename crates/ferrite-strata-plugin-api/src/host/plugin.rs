//! The host side: a loaded plugin presented as a [`Backend`].
//!
//! Everything that touches a raw pointer lives in this module, and every call has
//! the same three steps in the same order: check the plugin offered the entry,
//! call it, translate the `int` it returned. The middle step is where the `unsafe`
//! is. The third is where a decline stays a decline.
//!
//! # What the host owns and what the plugin owns
//!
//! Every `StrataWireBlob` a plugin hands back is copied into a `Vec` and the
//! plugin's copy is freed immediately through `free_blob`. Not fastidiousness —
//! it is the only way to be correct against a plugin that reuses one static buffer
//! between calls, which is a reasonable thing for a plugin to do and which the
//! ABI cannot forbid without mandating that every call allocate.
//!
//! Every executable holds an [`Arc`] of the plugin's inner state, so the `.so`
//! cannot be `dlclose`d while an executable that will call back into it is alive.

use core::ffi::{CStr, c_void};
use std::ffi::CString;
use std::sync::Arc;

use ferrite_strata::arena::{Arena, ArenaEntry};
use ferrite_strata::backend::{
    Backend, CompileOutcome, Executable, ExecutableId, PluginVersion, StrataVersion,
};
use ferrite_strata::capabilities::StrataSupport;
use ferrite_strata::error::{Error, ErrorCode};
use ferrite_strata::graph::Graph;
use ferrite_strata::id::ValueId;

use crate::raw::{
    STRATA_EXT_MESSAGE, StrataAllocDesc, StrataBuffer, StrataBufferRef, StrataCompileArgs,
    StrataError, StrataExecutable, StrataExtBase, StrataMessageExt, StrataPatternQuery,
    StrataPluginV1, StrataRunArgs, StrataWireBlob,
};
use crate::status;
use crate::wire::{self, SectionKind, WireError};

use super::PluginError;
use super::loader::{Library, Loaded, PluginVersionTriple};

/// A loaded vendor plugin, usable as a [`Backend`].
///
/// Cheap to share and safe to use from several threads, because the ABI has no
/// locking and therefore requires a plugin's context to be thread-safe. That
/// requirement is on the plugin; the wrapper adds no mutex, because a mutex here
/// would serialise every vendor in the host and would hide a plugin that is not
/// thread-safe rather than failing to load it.
#[derive(Debug)]
pub struct Plugin {
    inner: Arc<PluginInner>,
}

#[derive(Debug)]
struct PluginInner {
    library: Library,
    vtable: StrataPluginV1,
    ctx: *mut c_void,
    support: StrataSupport,
    triple: Option<PluginVersionTriple>,
}

// SAFETY: `ctx` is plugin-owned and the ABI requires it to be usable from any
// thread; the vtable is an immutable copy taken at load; `StrataSupport` is
// immutable after load; `Library` is `Send + Sync` on the same basis.
unsafe impl Send for PluginInner {}
unsafe impl Sync for PluginInner {}

impl Plugin {
    /// `dlopen` a plugin, negotiate its ABI, and decode its capabilities.
    ///
    /// Everything that can fail before the plugin is useful fails here, so a
    /// `Plugin` that exists is one whose capabilities decoded and whose version
    /// was acceptable. A partially-usable plugin is not a thing this API can
    /// express, which is the point: a host that has to check whether the plugin
    /// is usable checks in the wrong place.
    ///
    /// # Errors
    ///
    /// [`PluginError`], covering the whole load sequence: open, symbol, vtable,
    /// ABI negotiation, or a capability blob that would not decode.
    pub fn open(path: impl Into<String>) -> Result<Self, PluginError> {
        let library = Library::open(path)?;
        let Loaded {
            vtable,
            plugin_version,
        } = super::loader::read_vtable(&library)?;

        // SAFETY: no arguments, no state; the ABI is already negotiated, which is
        // what makes it safe to let a plugin open a device from here.
        let ctx = match vtable.plugin_init {
            Some(init) => unsafe { init() },
            None => core::ptr::null_mut(),
        };

        let support = read_capabilities(&vtable, ctx, library.path())?;

        Ok(Self {
            inner: Arc::new(PluginInner {
                library,
                vtable,
                ctx,
                support,
                triple: plugin_version,
            }),
        })
    }

    /// The plugin's stable backend name, from its own capability blob.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        &self.inner.support.capabilities().backend_name
    }

    /// The Strata ABI version this plugin speaks.
    #[must_use]
    pub fn strata_version(&self) -> StrataVersion {
        let raw = self.inner.vtable.abi_version;
        StrataVersion {
            major: raw.major as u16,
            minor: raw.minor as u16,
        }
    }

    /// The capabilities the plugin advertised, decoded once at load.
    #[must_use]
    pub fn support(&self) -> &StrataSupport {
        &self.inner.support
    }

    /// Every pattern id the plugin advertised, sorted and deduplicated.
    ///
    /// Deduplicated because a capability table has one row per
    /// pattern/dtype/shape triple, so the id necessarily repeats, and a list with
    /// repeats reads as though the plugin could run the pattern several times.
    #[must_use]
    pub fn pattern_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self
            .inner
            .support
            .capabilities()
            .patterns
            .iter()
            .map(|p| p.pattern_id.as_str())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Ask the plugin what it can run *right now*.
    ///
    /// Distinct from reading [`Self::support`], and the reason both exist. The
    /// capability blob answers "what can you do"; this answers "what can you do
    /// at this moment", which is the question a planner asks before committing a
    /// partition. A plugin that is out of memory, or whose device has been reset,
    /// may answer differently from its own declaration, and a host that only ever
    /// read the blob would hand it work it cannot take — producing exactly the
    /// run-time failure that the tri-state compile was designed to prevent.
    ///
    /// Returns [`ferrite_strata::capabilities::SupportLevel::Refuse`] for a
    /// plugin that has no `supports` entry, which is the safe direction: an
    /// unknown is never offered a subgraph.
    #[must_use]
    pub fn ask(
        &self,
        pattern_id: &str,
        dtype: ferrite_strata::dtype::DType,
        shape_class: ferrite_strata::shape::ShapeClass,
    ) -> ferrite_strata::capabilities::SupportLevel {
        let Some(query_fn) = self.inner.vtable.supports else {
            return ferrite_strata::capabilities::SupportLevel::Refuse;
        };

        let Ok(c_pattern) = CString::new(pattern_id) else {
            return ferrite_strata::capabilities::SupportLevel::Refuse;
        };

        let query = StrataPatternQuery {
            struct_size: core::mem::size_of::<StrataPatternQuery>(),
            pattern_id: c_pattern.as_ptr().cast::<u8>(),
            pattern_id_len: pattern_id.len(),
            dtype: wire::graph::dtype_to_raw(dtype),
            shape_class: wire::graph::shape_class_to_raw(shape_class),
        };

        // SAFETY: the vtable was validated at load, `query` matches its declared
        // struct size, and `c_pattern` outlives the call. The plugin reads the
        // pattern id and nothing else out of it.
        let raw = unsafe { query_fn(self.inner.ctx, &query) };
        match wire::caps::level_from_raw(raw) {
            Some(ferrite_strata::capabilities::SupportLevel::Fallback) => {
                ferrite_strata::capabilities::SupportLevel::Fallback
            }
            Some(ferrite_strata::capabilities::SupportLevel::Fused) => {
                ferrite_strata::capabilities::SupportLevel::Fused
            }
            // Anything else, including a level from a newer plugin, is a refusal.
            // Guessing `Fallback` for an unrecognised level would make the host
            // offer a subgraph it has no evidence the plugin can run.
            _ => ferrite_strata::capabilities::SupportLevel::Refuse,
        }
    }

    /// The path this plugin was loaded from.
    #[must_use]
    pub fn path(&self) -> &str {
        self.inner.library.path()
    }

    /// The plugin's own context, for callers reaching an entry point directly.
    #[must_use]
    pub fn context(&self) -> *mut c_void {
        self.inner.ctx
    }

    /// The vtable, for callers reaching an entry point directly.
    #[must_use]
    pub fn vtable(&self) -> &StrataPluginV1 {
        &self.inner.vtable
    }
}

impl Drop for PluginInner {
    fn drop(&mut self) {
        if let Some(free) = self.vtable.plugin_free {
            // SAFETY: `ctx` came from this plugin's `plugin_init` and is handed
            // back exactly once. `Library::drop` runs *after* this, because field
            // drop order is declaration order and `library` is declared first, so
            // the code is still mapped when `plugin_free` runs.
            unsafe { free(self.ctx) };
        }
    }
}

impl Backend for Plugin {
    fn name(&self) -> &str {
        self.backend_name()
    }

    fn plugin_version(&self) -> PluginVersion {
        match self.inner.triple {
            Some(t) => PluginVersion::new(t.major, t.minor, t.patch, self.strata_version()),
            // A plugin with no version extension is not a load failure: the
            // extension is additive and a v1 plugin predates it. Reporting 0.0.0
            // is honest about "we were not told", which matters because a host
            // that reported something else would be inventing a version.
            None => PluginVersion::new(0, 0, 0, self.strata_version()),
        }
    }

    fn capabilities(&self) -> &StrataSupport {
        &self.inner.support
    }

    fn compile(&self, subgraph: &Graph, pattern_id: &str) -> Result<CompileOutcome, Error> {
        let inner = Arc::clone(&self.inner);

        let outcome = compile_remote(&inner, subgraph, pattern_id);

        match outcome {
            Ok(Some(handle)) => Ok(CompileOutcome::Compiled(Box::new(ExecutableId::new(
                self.backend_name(),
                RemoteExecutable {
                    plugin: Arc::clone(&inner),
                    handle,
                    label: format!("{}::{}", self.backend_name(), pattern_id),
                    graph: subgraph.clone(),
                    outputs: subgraph.outputs().to_vec(),
                },
            )))),
            // A decline comes back inside `Ok`, never as an `Err`. This arm is
            // where ADR-0006 survives the FFI boundary: the transport is an
            // `Error`, and the tri-state is reconstructed from its code.
            Ok(None) => Ok(CompileOutcome::Declined {
                backend: self.backend_name().to_string(),
                reason: format!(
                    "the plugin declined `{pattern_id}`, and declined it through the \
                     status code rather than with a message"
                ),
            }),
            Err(err) => Err(err),
        }
    }
}

/// Compile through the vtable. `Ok(None)` means declined.
fn compile_remote(
    inner: &Arc<PluginInner>,
    subgraph: &Graph,
    pattern_id: &str,
) -> Result<Option<*mut StrataExecutable>, Error> {
    let vt = &inner.vtable;
    let ctx = inner.ctx;

    let mut blob = wire::Blob::new();
    blob.push(SectionKind::Graph, &wire::graph::encode(subgraph));
    let (subgraph_header, _subgraph_bytes) = blob.finish_borrowed();

    let capability_bytes = wire::caps::encode(inner.support.capabilities());
    let capabilities = StrataWireBlob {
        magic: crate::MAGIC,
        format_major: wire::FORMAT_MAJOR,
        format_minor: wire::FORMAT_MINOR,
        payload_len: capability_bytes.len() as u64,
        data: capability_bytes.as_ptr(),
        nul: 0,
    };

    let c_pattern = CString::new(pattern_id)
        .map_err(|_| Error::invalid_argument("the pattern id contains a NUL byte"))?;

    let args = StrataCompileArgs {
        struct_size: core::mem::size_of::<StrataCompileArgs>(),
        subgraph: subgraph_header,
        pattern_id: c_pattern.as_ptr().cast::<u8>(),
        pattern_id_len: pattern_id.len(),
        capabilities,
        // The host has already asked `supports` and been told this is runnable,
        // so this is a real compile. A planner that wants a probe sets the flag
        // itself; a plugin that ignores it still returns a correct answer, it
        // just cannot be costed before being tried.
        probe: 0,
        reserved: [0; 7],
    };

    let mut out_executable: *mut StrataExecutable = core::ptr::null_mut();
    let mut out_error: *mut StrataError = core::ptr::null_mut();
    // SAFETY: the vtable was validated at load, both blobs and the pattern string
    // outlive the call, and each out-param is written only on the matching status.
    let rc = unsafe {
        (vt.compile.expect("validated at load"))(ctx, &args, &mut out_executable, &mut out_error)
    };

    match status_of(rc) {
        Status::Ok => Ok(Some(out_executable)),
        Status::Declined => {
            // A declining plugin must not have allocated an executable, but if it
            // did, freeing it here is the difference between a decline and a leak
            // that grows with every repartition.
            if !out_executable.is_null() {
                unsafe { (vt.free_executable.expect("validated at load"))(ctx, out_executable) };
            }
            free_error(vt, out_error);
            Ok(None)
        }
        Status::Failed(code) => Err(take_error(vt, out_error).unwrap_or_else(|| {
            Error::new(code, format!("compile failed with `{}`", status::name(rc)))
        })),
    }
}

/// An executable that lives in a plugin's address space.
struct RemoteExecutable {
    plugin: Arc<PluginInner>,
    handle: *mut StrataExecutable,
    label: String,
    graph: Graph,
    outputs: Vec<ValueId>,
}

// SAFETY: the ABI has no locking and requires a plugin's context and executables
// to be usable from any thread, so this handle is exactly as thread-safe as the
// plugin is. The host additionally promises never to call `run` on one executable
// concurrently, which is the one guarantee it adds.
unsafe impl Send for RemoteExecutable {}

impl Executable for RemoteExecutable {
    fn label(&self) -> &str {
        &self.label
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    fn run(&self, arena: &mut Arena) -> Result<(), Error> {
        let vt = &self.plugin.vtable;

        let alloc = vt
            .alloc
            .ok_or_else(|| Error::unimplemented("this plugin exposes no buffer allocator"))?;
        let write = vt
            .buffer_write
            .ok_or_else(|| Error::unimplemented("this plugin exposes no host-to-buffer copy"))?;
        let read = vt
            .buffer_read
            .ok_or_else(|| Error::unimplemented("this plugin exposes no buffer-to-host copy"))?;
        let release = vt
            .buffer_release
            .ok_or_else(|| Error::unimplemented("alloc was offered without a matching release"))?;

        let mut buffers: Vec<*mut StrataBuffer> = Vec::new();
        let result = self.run_inner(arena, alloc, write, read, &mut buffers);

        // Released on every path out, including the error paths. A leak on an
        // error return is how a long-running host ends up blaming the device for
        // the plugin's memory.
        for buffer in &buffers {
            unsafe { release(self.plugin.ctx, *buffer) };
        }
        result
    }
}

/// `alloc`'s signature, named so `run_inner`'s is readable.
type AllocFn = unsafe extern "C" fn(
    *mut c_void,
    *const StrataAllocDesc,
    *mut *mut StrataBuffer,
    *mut *mut StrataError,
) -> i32;

/// `buffer_write`'s signature: host bytes into a plugin buffer.
type WriteFn = unsafe extern "C" fn(*mut c_void, *mut StrataBuffer, u64, *const u8, u64) -> i32;

/// `buffer_read`'s signature: plugin bytes out to the host.
type ReadFn = unsafe extern "C" fn(*mut c_void, *const StrataBuffer, u64, *mut u8, u64) -> i32;

impl RemoteExecutable {
    fn run_inner(
        &self,
        arena: &mut Arena,
        alloc: AllocFn,
        write: WriteFn,
        read: ReadFn,
        buffers: &mut Vec<*mut StrataBuffer>,
    ) -> Result<(), Error> {
        let vt = &self.plugin.vtable;
        let ctx = self.plugin.ctx;

        let mut refs: Vec<StrataBufferRef> = Vec::with_capacity(self.graph.value_count());

        for (index, value) in self.graph.values().iter().enumerate() {
            let bytes = value.byte_len();
            let dims: Vec<u64> = value.shape.dims().iter().map(|d| *d as u64).collect();
            let desc = StrataAllocDesc {
                struct_size: core::mem::size_of::<StrataAllocDesc>(),
                dtype: wire::graph::dtype_to_raw(value.dtype),
                rank: value.shape.rank() as u32,
                reserved: 0,
                dims: dims.as_ptr(),
            };

            let mut handle: *mut StrataBuffer = core::ptr::null_mut();
            let mut out_error: *mut StrataError = core::ptr::null_mut();
            // SAFETY: `desc` and the `dims` it points at outlive the call, and both
            // out-params are written only on the matching status.
            let rc = unsafe { alloc(ctx, &desc, &mut handle, &mut out_error) };
            if status_of(rc) != Status::Ok || handle.is_null() {
                return Err(take_error(vt, out_error).unwrap_or_else(|| {
                    Error::new(
                        code_of(rc),
                        format!("the plugin could not allocate a buffer for {}", value.name),
                    )
                }));
            }
            buffers.push(handle);

            // Only values the arena already holds are copied in. A subgraph value
            // the host never populated is one the plugin is expected to produce,
            // and writing zeros over it would destroy a fusion's accumulator.
            if let Some(entry) = arena.get(ValueId(index as u32))
                && let Some(data) = encode_host(entry, value.dtype, bytes)?
            {
                // SAFETY: `data` outlives the call, and `bytes` is what the graph's
                // dtype and shape imply, which is the same desc the plugin
                // allocated from.
                let rc = unsafe { write(ctx, handle, 0, data.as_ptr(), bytes as u64) };
                if status_of(rc) != Status::Ok {
                    return Err(Error::new(
                        code_of(rc),
                        format!("copying {} into the plugin failed", value.name),
                    ));
                }
            }

            refs.push(StrataBufferRef {
                buffer: handle,
                value_index: index as u32,
                reserved: 0,
            });
        }

        let args = StrataRunArgs {
            struct_size: core::mem::size_of::<StrataRunArgs>(),
            executable: self.handle,
            arg_count: refs.len() as u32,
            args: refs.as_ptr(),
            host_arena: core::ptr::null_mut(),
        };
        let mut out_error: *mut StrataError = core::ptr::null_mut();
        // SAFETY: `refs` outlives the call, `args` matches its declared struct size,
        // and `host_arena` is null so the plugin writes only its own buffers.
        let rc = unsafe { (vt.run.expect("validated at load"))(ctx, &args, &mut out_error) };
        if status_of(rc) != Status::Ok {
            return Err(take_error(vt, out_error)
                .unwrap_or_else(|| Error::new(code_of(rc), "the plugin's run failed")));
        }

        // Only the declared outputs come back. A subgraph's internal values are
        // the plugin's business once the step is over, and copying them out would
        // read device memory the host has no reason to have.
        for (index, value) in self.graph.values().iter().enumerate() {
            let id = ValueId(index as u32);
            if !self.graph.is_output(id) {
                continue;
            }
            let Some(handle) = buffers.get(index) else {
                continue;
            };
            let bytes = value.byte_len();
            let mut out = vec![0u8; bytes];
            // SAFETY: `out` is exactly `bytes` long and the plugin allocated this
            // buffer from the same dtype and shape.
            let rc = unsafe { read(ctx, *handle, 0, out.as_mut_ptr(), bytes as u64) };
            if status_of(rc) != Status::Ok {
                return Err(Error::new(
                    code_of(rc),
                    format!("reading {id} back from the plugin failed"),
                ));
            }
            if let Some(entry) = decode_host(out, value.dtype) {
                arena.insert(id, entry);
            }
        }
        Ok(())
    }
}

impl Drop for RemoteExecutable {
    fn drop(&mut self) {
        if let Some(free) = self.plugin.vtable.free_executable {
            // SAFETY: the handle came from this plugin's `compile` and is freed
            // exactly once, before the plugin's context is released.
            unsafe { free(self.plugin.ctx, self.handle) };
        }
    }
}

/// The host's bytes for an arena entry, or `None` to leave the buffer alone.
///
/// `Opaque` returns `None` rather than copying the bytes: the entry exists so the
/// host can hand a device pointer back unchanged, and reading it as bytes would
/// be exactly the dereference the indirection was added to prevent.
fn encode_host(
    entry: &ArenaEntry,
    dtype: ferrite_strata::dtype::DType,
    bytes: usize,
) -> Result<Option<Vec<u8>>, Error> {
    match entry {
        ArenaEntry::F32(values) => {
            if dtype.element.size_in_bytes() != 4 {
                return Err(Error::new(
                    ErrorCode::InvalidArgument,
                    format!(
                        "the arena holds f32 but the graph declares `{}`",
                        dtype.element.name()
                    ),
                ));
            }
            let mut out = Vec::with_capacity(values.len() * 4);
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.resize(bytes, 0);
            Ok(Some(out))
        }
        ArenaEntry::Opaque(_) => Ok(None),
    }
}

fn decode_host(bytes: Vec<u8>, dtype: ferrite_strata::dtype::DType) -> Option<ArenaEntry> {
    if dtype.element.size_in_bytes() != 4 {
        return Some(ArenaEntry::Opaque(bytes));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let raw: [u8; 4] = chunk.try_into().expect("chunks_exact(4)");
        values.push(f32::from_le_bytes(raw));
    }
    Some(ArenaEntry::F32(values))
}

/// The three outcomes of an entry point's return code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Status {
    Ok,
    Declined,
    Failed(ErrorCode),
}

fn status_of(rc: i32) -> Status {
    match rc {
        status::OK => Status::Ok,
        status::DECLINED => Status::Declined,
        other => Status::Failed(code_of(other)),
    }
}

/// The error code a failed status implies.
///
/// An unrecognised code becomes [`ErrorCode::Internal`] rather than being passed
/// through as a number nobody can name: a code from a newer plugin is not a
/// licence to invent an error, and `Internal` at least says "this host cannot
/// vouch for what happened".
fn code_of(rc: i32) -> ErrorCode {
    status::to_error_code(rc).unwrap_or(ErrorCode::Internal)
}

/// Turn a plugin error into a host error and free the plugin's copy.
fn take_error(vt: &StrataPluginV1, error: *mut StrataError) -> Option<Error> {
    if error.is_null() {
        return None;
    }

    // SAFETY: both entries were validated at load and `error` is this plugin's.
    let code = code_of(unsafe { (vt.error_code.expect("validated at load"))(error) } as i32);

    let mut text: Vec<u8> = Vec::new();
    // SAFETY: both entries are validated at load, `visit_payload` outlives the
    // call, and the context is the same `Vec` that function appends to. The
    // callback is synchronous, so nothing can outlive the borrow.
    unsafe {
        (vt.error_payload.expect("validated at load"))(
            error,
            Some(visit_payload),
            (&mut text as *mut Vec<u8>).cast(),
        )
    };
    free_error(vt, error);

    let message = String::from_utf8_lossy(&text).into_owned();
    Some(Error::new(
        code,
        if message.is_empty() {
            format!("the plugin reported {}", code.name())
        } else {
            message
        },
    ))
}

fn free_error(vt: &StrataPluginV1, error: *mut StrataError) {
    if !error.is_null() {
        // SAFETY: the pointer came from this plugin, and `error_free` is the entry
        // point that owns its storage.
        unsafe { (vt.error_free.expect("validated at load"))(error) };
    }
}

/// Ask a plugin for its capability blob and decode it.
fn read_capabilities(
    vt: &StrataPluginV1,
    ctx: *mut c_void,
    path: &str,
) -> Result<StrataSupport, PluginError> {
    let mut blob = StrataWireBlob {
        magic: 0,
        format_major: 0,
        format_minor: 0,
        payload_len: 0,
        data: core::ptr::null(),
        nul: 0,
    };
    // SAFETY: `blob` is ours and outlives the call; the plugin writes it only on OK.
    let rc = unsafe { (vt.capabilities.expect("validated at load"))(ctx, &mut blob) };
    if status_of(rc) != Status::Ok {
        return Err(PluginError::Capabilities {
            path: path.to_string(),
            detail: format!("the capabilities entry returned `{}`", status::name(rc)),
        });
    }

    let payload = copy_blob(vt, ctx, &blob).map_err(|detail| PluginError::Capabilities {
        path: path.to_string(),
        detail,
    })?;

    let mut full = Vec::with_capacity(crate::raw::WIRE_HEADER_LEN + payload.len());
    full.extend_from_slice(&crate::MAGIC.to_le_bytes());
    full.extend_from_slice(&wire::FORMAT_MAJOR.to_le_bytes());
    full.extend_from_slice(&wire::FORMAT_MINOR.to_le_bytes());
    full.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    full.extend_from_slice(&payload);

    let fail = |detail: String| PluginError::Capabilities {
        path: path.to_string(),
        detail,
    };

    let parsed = wire::Parsed::parse(&full).map_err(|e: WireError| fail(e.to_string()))?;
    let section = parsed
        .section(SectionKind::Capabilities)
        .ok_or_else(|| fail("the blob carried no capabilities section".into()))?;
    let caps = wire::caps::decode(section).map_err(|e: WireError| fail(e.to_string()))?;

    Ok(StrataSupport::new(caps))
}

/// Copy a plugin-owned blob and free the plugin's copy.
fn copy_blob(
    vt: &StrataPluginV1,
    ctx: *mut c_void,
    blob: &StrataWireBlob,
) -> Result<Vec<u8>, String> {
    if blob.payload_len > isize::MAX as u64 {
        return Err(format!(
            "the plugin declared a {}-byte blob",
            blob.payload_len
        ));
    }
    let len = usize::try_from(blob.payload_len)
        .map_err(|_| "the blob length does not fit in memory".to_string())?;
    if blob.data.is_null() {
        return Err("the plugin returned a null blob pointer".into());
    }
    // SAFETY: the plugin claims `len` readable bytes at `data`. This is the one
    // place the host trusts a pointer from across the boundary, and there is no
    // way to avoid it — a `dlopen`ed plugin is trusted code by definition, which
    // is exactly why every *length inside* the blob is bounds-checked and this
    // one cannot be.
    let out = unsafe { core::slice::from_raw_parts(blob.data, len) }.to_vec();
    // SAFETY: `blob` was produced by this plugin, and `free_blob` is its own
    // deallocator for it.
    unsafe { (vt.free_blob.expect("validated at load"))(ctx, blob as *const _ as *mut _) };
    Ok(out)
}

/// The error-payload visitor: appends the one v1 payload's message to a
/// `Vec<u8>` reached through the context.
///
/// A free function rather than a closure because the ABI wants a plain C function
/// pointer and a non-capturing Rust closure cannot be coerced to one — its ABI is
/// Rust's, not C's. The captured state travels in `ctx` instead, which is what
/// `ctx` is for.
unsafe extern "C" fn visit_payload(ctx: *mut c_void, payload: *mut StrataExtBase) -> i32 {
    if ctx.is_null() || payload.is_null() {
        return 1;
    }
    // SAFETY: the host passes the `&mut Vec<u8>` it allocated for this call as
    // `ctx`, and `error_payload` is documented to call the visitor synchronously,
    // so the borrow is still live here.
    let out = unsafe { &mut *ctx.cast::<Vec<u8>>() };
    // SAFETY: `payload` is one node of the plugin's own extension chain, and the
    // declared size is checked before any field past the header is read.
    let node = unsafe { &*payload };
    if node.ty != STRATA_EXT_MESSAGE || node.struct_size < core::mem::size_of::<StrataMessageExt>()
    {
        return 0;
    }
    // SAFETY: the tag matched and the declared size covers the whole payload.
    let message = unsafe { &*payload.cast::<StrataMessageExt>() }.message;
    if message.is_null() {
        return 0;
    }
    // SAFETY: the ABI says `message` is a NUL-terminated string owned by the plugin
    // and valid for the lifetime of the error, which outlives this call.
    if let Ok(text) = unsafe { CStr::from_ptr(message) }.to_str() {
        out.extend_from_slice(text.as_bytes());
    }
    0
}
