//! The C ABI a Ferrite Strata vendor plugin implements, and the host side that
//! loads one.
//!
//! # What this crate is for
//!
//! The rest of `ferrite-strata` is Rust types: traits, a graph IR, a
//! partitioner. A backend implemented as a `Box<dyn Backend>` is not something a
//! vendor who did not write our compiler can provide, because it requires
//! agreeing on Rust's trait object layout, monomorphisation and panic behaviour.
//! This crate is the boundary that removes all three: a C vtable, a versioned
//! binary format, and a `dlopen`.
//!
//! # The three pieces
//!
//! | Module | Who needs it |
//! |---|---|
//! | [`raw`] | anyone writing a plugin, in any language — it mirrors `include/strata_plugin.h` |
//! | [`wire`] | anyone who has to read or write a subgraph or a capability set |
//! | [`host`] | a host that wants to load a plugin; not needed by a vendor |
//!
//! A vendor in Rust depends on this crate and implements the C function pointers.
//! A vendor in anything else compiles `include/strata_plugin.h` and implements
//! the same declarations. Neither writes a Rust trait, and neither links our
//! compiler.
//!
//! # The idea that survives the crossing
//!
//! **A plugin may decline.** [`status::DECLINED`] is a first-class return code, not
//! an error, and it is the reason partial adoption works: a vendor that can fuse
//! attention and run nothing else says so, the partitioner keeps the rest, and
//! nobody has to special-case the vendor's limitations. See
//! [ADR-0006](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0006-compile-returns-a-tri-state-with-decline-as-a-first-class-outcome.md).
//!
//! # Trust
//!
//! A loaded plugin is arbitrary code. Two consequences are designed in rather
//! than assumed:
//!
//! - Nothing a plugin sends is trusted before it is bounds-checked. See
//!   [`wire`] for the two rules and why each is load-bearing.
//! - Nothing is freed across the boundary by anything other than its allocator.
//!   A plugin linked against a different allocator than the host will otherwise
//!   corrupt the heap in a way that presents as a memory bug in somebody's
//!   kernel.
//!
//! # Example, from the host side
//!
//! ```no_run
//! use ferrite_strata::DType;
//! use ferrite_strata::ElementType;
//! use ferrite_strata::shape::ShapeClass;
//! use ferrite_strata_plugin_api::host::{Plugin, PluginError};
//!
//! # fn main() -> Result<(), PluginError> {
//! let plugin = Plugin::open("/opt/acme/lib/libstrata-acme.so")?;
//! println!("{} speaks Strata {}", plugin.backend_name(), plugin.strata_version());
//! for pattern in plugin.pattern_ids() {
//!     let dtype = DType::plain(ElementType::F32);
//!     println!("{pattern}: {:?}", plugin.ask(pattern, dtype, ShapeClass::Matrix));
//! }
//! # Ok(())
//! # }
//! ```

#![doc(html_root_url = "https://docs.rs/ferrite-strata-plugin-api/0.0.1")]

/// The host side. The only module in this crate that touches a raw pointer, and
/// therefore the only one that opts out of the workspace's `unsafe_code = "deny"`
/// — scoped to this one `mod` rather than the crate, so every other line here is
/// still checked by default and the opt-out stays greppable.
#[allow(unsafe_code)]
pub mod host;
pub mod raw;
pub mod status;
pub mod wire;

pub use host::{Plugin, PluginError};
pub use raw::{MAGIC, STRATA_PLUGIN_ENTRY_SYMBOL, StrataAbiVersion, StrataExtBase, StrataPluginV1};
pub use wire::{FORMAT_MAJOR, FORMAT_MINOR, WireError};
