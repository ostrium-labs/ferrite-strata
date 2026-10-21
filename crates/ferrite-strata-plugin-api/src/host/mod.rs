//! The host side of the plugin ABI.
//!
//! Not needed by a vendor: this is the code that `dlopen`s a plugin, negotiates
//! its version, and presents it as a [`Backend`](ferrite_strata::Backend). A
//! vendor implements [`crate::raw`] and nothing here.
//!
//! [`Plugin`] is the whole API. It owns the library, the vtable copy, the
//! plugin's context and the decoded capability blob, and it hands out
//! [`CompileOutcome`](ferrite_strata::backend::CompileOutcome) values that are
//! indistinguishable from an in-process backend's — which is the property that
//! matters, because a partitioner that treats a plugin differently from a
//! built-in backend has stopped being vendor-neutral.

mod loader;
mod plugin;

pub use loader::{Library, Loaded, PluginVersionTriple};
pub use plugin::Plugin;

use core::fmt;

/// Why a plugin could not be used.
///
/// Every variant is a *load* failure. Once [`Plugin::open`] has returned, a
/// plugin is either fully usable or reports errors from its own entry points, and
/// those come back as
/// [`Error`](ferrite_strata::error::Error) rather than as this type. A
/// partially-usable plugin is not representable.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PluginError {
    /// `dlopen` failed.
    Open {
        /// The path that was tried.
        path: String,
        /// What the loader said.
        detail: String,
    },
    /// The entry symbol was absent.
    Symbol {
        /// The path that was tried.
        path: String,
        /// The symbol name.
        name: String,
    },
    /// The vtable was structurally wrong.
    Vtable {
        /// The path that was tried.
        path: String,
        /// What was wrong with it.
        detail: String,
    },
    /// The ABI version could not be negotiated.
    ///
    /// A separate variant from [`PluginError::Vtable`] because the remedy is
    /// different: a version mismatch means "install a different plugin", and a
    /// structural one means "fix the plugin".
    Abi {
        /// The path that was tried.
        path: String,
        /// The versions involved.
        detail: String,
    },
    /// The capability blob would not decode.
    Capabilities {
        /// The path that was tried.
        path: String,
        /// What was wrong with it.
        detail: String,
    },
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { path, detail } => {
                write!(f, "could not dlopen `{path}`: {detail}")
            }
            Self::Symbol { path, name } => {
                write!(f, "`{path}` does not export `{name}`")
            }
            Self::Vtable { path, detail } => {
                write!(f, "`{path}` has a malformed vtable: {detail}")
            }
            Self::Abi { path, detail } => {
                write!(f, "`{path}` speaks an unusable ABI: {detail}")
            }
            Self::Capabilities { path, detail } => {
                write!(
                    f,
                    "`{path}` declared capabilities that did not decode: {detail}"
                )
            }
        }
    }
}

impl core::error::Error for PluginError {}
