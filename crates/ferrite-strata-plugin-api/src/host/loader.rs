//! Loading a `.so` and getting at its vtable.
//!
//! The load sequence is the ABI's compatibility mechanism, and every step of it
//! exists because a plausible-looking shortcut fails:
//!
//! 1. `dlopen` the path.
//! 2. `dlsym` the one entry symbol. A plugin that exports nothing else cannot be
//!    half-loaded.
//! 3. **Copy the vtable out, by declared size.** The vtable's `struct_size` is
//!    read first, and at most `min(struct_size, size_of::<StrataPluginV1>())` bytes
//!    are copied. A plugin that appended entries is readable; a plugin that
//!    reports a *shorter* struct than the eight mandatory entries is rejected,
//!    because otherwise the copy would read past what the plugin allocated.
//! 4. Negotiate `abi_version`.
//! 5. Reject a vtable missing any mandatory entry. A null function pointer that
//!    survives to a call is a segfault in someone else's process, so it is caught
//!    at load with a message naming the entry.
//! 6. Walk the extension chain, guarded against a cycle and a length shorter
//!    than its own header.
//!
//! # Threading
//!
//! [`Library`] is `Send + Sync`. The `dlopen` handle is safe to share, and the
//! vtable is copied out rather than pointed at, so two threads calling through it
//! read the same immutable function pointers. What that does *not* make safe is
//! the plugin's own state: the ABI has no locking, so a plugin's `ctx` must be
//! usable from any thread. That is a contract on the plugin, stated in
//! [`StrataPluginV1`](crate::raw::StrataPluginV1) and repeated at each call site
//! here.

use core::ffi::{CStr, c_void};
use std::ffi::CString;

use super::PluginError;
use crate::raw::{
    STRATA_EXT_NONE, STRATA_PLUGIN_ENTRY_SYMBOL, StrataAbiVersion, StrataExtBase, StrataPluginV1,
};

/// A `dlopen`ed shared object.
///
/// Owns the handle and closes it on drop. Holding one is what keeps a plugin's
/// code mapped; the [`Plugin`](super::Plugin) holds an `Arc` of it so a compiled
/// executable cannot outlive the code that will run it.
#[derive(Debug)]
pub struct Library {
    path: String,
    handle: *mut c_void,
}

// SAFETY: a `dlopen` handle is an opaque token with no thread affinity. The libc
// functions that take one (`dlsym`, `dlclose`) are documented as thread-safe, and
// the vtable we copy out of it is immutable once read.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    /// `dlopen` a plugin.
    ///
    /// # Errors
    ///
    /// [`PluginError::Open`] if the file is missing, is not a shared object, or
    /// a constructor in it returned non-zero.
    pub fn open(path: impl Into<String>) -> Result<Self, PluginError> {
        let path = path.into();
        let c_path = CString::new(path.as_str()).map_err(|_| PluginError::Open {
            path: path.clone(),
            detail: "the path contains a NUL byte".into(),
        })?;
        // RTLD_NOW rather than RTLD_LAZY, so an undefined symbol in the plugin
        // fails here with a name we can print, instead of at the first call
        // through a lazily-resolved PLT entry.
        // SAFETY: `c_path` is a valid NUL-terminated string for the duration of
        // the call, and the handle is stored unaliased in a struct that owns it.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(PluginError::Open {
                path,
                detail: last_dl_error(),
            });
        }
        Ok(Self { path, handle })
    }

    /// The path this was opened from.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Resolve a symbol.
    ///
    /// # Errors
    ///
    /// [`PluginError::Symbol`] if the symbol is absent.
    pub fn symbol(&self, name: &str) -> Result<*mut c_void, PluginError> {
        let c_name = CString::new(name).map_err(|_| PluginError::Symbol {
            path: self.path.clone(),
            name: name.to_string(),
        })?;
        // SAFETY: the handle came from a successful `dlopen`, and `c_name` is a
        // valid NUL-terminated string. The returned pointer is into the mapped
        // object, which `self` keeps alive.
        let sym = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
        if sym.is_null() {
            return Err(PluginError::Symbol {
                path: self.path.clone(),
                name: name.to_string(),
            });
        }
        Ok(sym)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle came from `dlopen` and is dropped exactly once, and
        // every pointer derived from it is gone by now — `Plugin` holds an `Arc`
        // so no executable outlives the library.
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

/// The result of reading a plugin's vtable and version extensions.
#[derive(Debug)]
pub struct Loaded {
    /// The vtable, copied out of the mapped object.
    pub vtable: StrataPluginV1,
    /// The vendor's version, if the extension chain carried it.
    pub plugin_version: Option<PluginVersionTriple>,
}

/// A vendor's own version, as read from the extension chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PluginVersionTriple {
    /// The vendor's major version.
    pub major: u16,
    /// The vendor's minor version.
    pub minor: u16,
    /// The vendor's patch version.
    pub patch: u16,
    /// The Strata version it targets.
    pub targets: StrataAbiVersion,
}

/// The eight entry points that must be present.
///
/// Listed as names so the error message can name the one that is missing. A null
/// in any of these is a crash waiting for a caller, so it is a load failure.
const MANDATORY: [&str; 9] = [
    "error_free",
    "error_code",
    "error_payload",
    "capabilities",
    "supports",
    "compile",
    "run",
    "free_executable",
    "free_blob",
];

/// Read the vtable out of a loaded library.
///
/// # Errors
///
/// [`PluginError::Symbol`] if the entry symbol is absent,
/// [`PluginError::Vtable`] if the struct size is implausible or a mandatory entry
/// is null, or [`PluginError::Abi`] if the version cannot be negotiated.
pub(super) fn read_vtable(library: &Library) -> Result<Loaded, PluginError> {
    let symbol = library.symbol(STRATA_PLUGIN_ENTRY_SYMBOL)?;
    // SAFETY: the symbol is a `StrataPluginV1` because that is what the ABI says
    // the entry symbol names. We read only `struct_size` bytes of it first, and
    // only copy what that declares.
    let raw = unsafe { &*symbol.cast::<StrataPluginV1>() };

    let declared = raw.struct_size;
    let ours = core::mem::size_of::<StrataPluginV1>();
    if declared < core::mem::size_of::<StrataAbiVersion>() + 3 * core::mem::size_of::<usize>() {
        return Err(PluginError::Vtable {
            path: library.path().to_string(),
            detail: format!(
                "struct_size is {declared}, too small to hold the header, the \
                 version and the extension pointer"
            ),
        });
    }

    let vtable = copy_prefix(raw, declared.min(ours));

    negotiate(&vtable).map_err(|detail| PluginError::Abi {
        path: library.path().to_string(),
        detail,
    })?;

    for name in MANDATORY {
        if entry(&vtable, name).is_none() {
            return Err(PluginError::Vtable {
                path: library.path().to_string(),
                detail: format!("the mandatory entry `{name}` is null"),
            });
        }
    }

    let plugin_version = walk_extensions(vtable.extension_start);

    Ok(Loaded {
        vtable,
        plugin_version,
    })
}

/// Copy `len` bytes out of `src`, zeroing the tail.
///
/// The zeroing matters: a plugin built against an older header has a shorter
/// vtable, so the fields this host knows about but the plugin does not are read
/// from uninitialised memory unless we clear them. Nulling them turns "an older
/// plugin" into a clean "that optional feature is absent" rather than a jump
/// through whatever was in the padding.
fn copy_prefix(src: &StrataPluginV1, len: usize) -> StrataPluginV1 {
    let mut bytes = vec![0u8; core::mem::size_of::<StrataPluginV1>()];
    // SAFETY: `len` is `min(declared, size_of::<StrataPluginV1>())` and the
    // destination is exactly `size_of::<StrataPluginV1>()` bytes, so both sides
    // of the copy stay inside their own allocation.
    unsafe {
        core::ptr::copy_nonoverlapping(
            (src as *const StrataPluginV1).cast::<u8>(),
            bytes.as_mut_ptr(),
            len,
        );
    }
    // SAFETY: every bit of `bytes` came from a `StrataPluginV1` of that size, so
    // it is a valid one whatever padding the plugin left.
    unsafe { core::ptr::read(bytes.as_ptr().cast::<StrataPluginV1>()) }
}

fn negotiate(vtable: &StrataPluginV1) -> Result<(), String> {
    let theirs = vtable.abi_version;
    let ours = StrataAbiVersion::CURRENT;
    if theirs.major != ours.major {
        return Err(format!(
            "plugin speaks ABI major {}.{}, this host speaks {}.{}; a major \
             mismatch needs a different host, not a re-dlopen",
            theirs.major, theirs.minor, ours.major, ours.minor
        ));
    }
    if theirs.minor < ours.minor {
        return Err(format!(
            "plugin speaks ABI {}.{}, this host needs at least {}.{}; a plugin \
             older than the host cannot be taught the fields it is missing",
            theirs.major, theirs.minor, ours.major, ours.minor
        ));
    }
    Ok(())
}

fn entry(vtable: &StrataPluginV1, name: &str) -> Option<*const c_void> {
    match name {
        "error_free" => vtable.error_free.map(|f| f as *const c_void),
        "error_code" => vtable.error_code.map(|f| f as *const c_void),
        "error_payload" => vtable.error_payload.map(|f| f as *const c_void),
        "capabilities" => vtable.capabilities.map(|f| f as *const c_void),
        "supports" => vtable.supports.map(|f| f as *const c_void),
        "compile" => vtable.compile.map(|f| f as *const c_void),
        "run" => vtable.run.map(|f| f as *const c_void),
        "free_executable" => vtable.free_executable.map(|f| f as *const c_void),
        "free_blob" => vtable.free_blob.map(|f| f as *const c_void),
        _ => None,
    }
}

/// Walk the extension chain, collecting what this build understands.
///
/// Bounded in two ways, both because the chain is a linked list in someone else's
/// address space: a depth cap stops a cycle, and the `struct_size` check stops a
/// node whose size is smaller than its own header from advancing the pointer by
/// less than the header. Without the second, a malformed chain walks backwards
/// and never terminates.
fn walk_extensions(start: *mut StrataExtBase) -> Option<PluginVersionTriple> {
    let mut version = None;
    let mut cursor = start;
    for _ in 0..64 {
        if cursor.is_null() {
            break;
        }
        // SAFETY: non-null and, by the loop invariant, one of the nodes the plugin
        // linked. `struct_size` is read before anything else so a short node is
        // rejected rather than dereferenced past its header.
        let node = unsafe { &*cursor };
        let header = core::mem::size_of::<StrataExtBase>();
        if node.struct_size < header {
            break;
        }
        if node.ty == crate::raw::STRATA_EXT_PLUGIN_VERSION
            && node.struct_size >= core::mem::size_of::<crate::raw::StrataPluginVersionExt>()
        {
            // SAFETY: the tag matched and the declared size covers the whole
            // extension, so the wider struct's fields are all present.
            version = Some(unsafe {
                let ext = &*cursor.cast::<crate::raw::StrataPluginVersionExt>();
                PluginVersionTriple {
                    major: ext.major,
                    minor: ext.minor,
                    patch: ext.patch,
                    targets: ext.targets,
                }
            });
            break;
        }
        if node.ty == STRATA_EXT_NONE {
            break;
        }
        cursor = node.next;
    }
    version
}

fn last_dl_error() -> String {
    // SAFETY: `dlerror` returns either null or a pointer to a NUL-terminated
    // string owned by the loader, valid until the next `dlerror` call. Nothing
    // here can call it in between.
    unsafe {
        let err = libc::dlerror();
        if err.is_null() {
            return "no loader diagnostic was available".into();
        }
        CStr::from_ptr(err).to_string_lossy().into_owned()
    }
}
