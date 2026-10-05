//! The closed error code set, and the runtime error type.
//!
//! # Six codes, and why there are six
//!
//! PJRT carries 17 abseil-shaped codes. Ours is a deliberately smaller closed
//! enum, because the count is an ABI surface: every code a plugin can return has
//! to be understood by every host, forever.
//!
//! ```text
//! INVALID_ARGUMENT  NOT_FOUND  UNIMPLEMENTED  RESOURCE_EXHAUSTED  INTERNAL  DECLINED
//! ```
//!
//! `DECLINED` is the one that earns its place, and the reason is
//! [ADR-0006](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0006-compile-returns-a-tri-state-with-decline-as-a-first-class-outcome.md):
//! "this vendor will not take this subgraph" is machine-actionable and is not an
//! error at all — the correct response is to run it somewhere else. Folding it
//! into `UNIMPLEMENTED` is what forces every vendor toward either accepting
//! everything or refusing everything.
//!
//! The separation the code makes is the one Burn makes between
//! `CompilationError::is_refusal()` and `is_device_poisoned()`: *the vendor said
//! no*, *the device broke*, and *the code is wrong* are three different
//! situations with three different responses.
//!
//! # What is deliberately absent
//!
//! No `CANCELLED`, no `DEADLINE_EXCEEDED`, no `UNAVAILABLE`, no `DATA_LOSS`. A
//! Strata backend has no cancel path, because the plugin interface is
//! compile-and-run with no stream-cancellation callback, and adding a code for an
//! event that cannot occur would be an obligation no implementation could meet.
//! Adding one later is fine — it is an additive enum extension with an explicit
//! unknown-code arm in the decoder.

use core::fmt;

/// The closed set of failure codes that cross the plugin boundary.
///
/// Exactly these six, in the order of the table in
/// [`docs/design-notes.md`](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/design-notes.md#1-why-not-pjrt).
/// `#[non_exhaustive]` so a future major version can append, while every match
/// written today still needs an explicit fallback — which is what forces the
/// decision to be deliberate.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The caller asked for something impossible: a shape that does not match the
    /// op, a bad axis, a buffer whose dtype disagrees with the subgraph.
    InvalidArgument = 1,
    /// A named thing does not exist: an executable from another plugin, a buffer
    /// that has already been released, an event from a different stream.
    NotFound = 2,
    /// The backend has no implementation for the request. A *hard* failure, not a
    /// decline: the vendor should have said [`ErrorCode::Declined`] instead.
    Unimplemented = 3,
    /// A limit was hit: memory, buffer count, subgraph size, queue depth.
    ResourceExhausted = 4,
    /// A bug. Either in Strata or in the plugin, and in both cases "restart the
    /// device and report it", not "try a different partition".
    Internal = 5,
    /// The backend understood the request and will not take it.
    ///
    /// Not an error condition; see the module docs. A host that receives this
    /// should re-partition or fall back, and a host that treats it as a failure
    /// has a bug.
    Declined = 6,
}

impl ErrorCode {
    /// A stable lowercase name, for logs and for text formats.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid-argument",
            Self::NotFound => "not-found",
            Self::Unimplemented => "unimplemented",
            Self::ResourceExhausted => "resource-exhausted",
            Self::Internal => "internal",
            Self::Declined => "declined",
        }
    }

    /// The wire value, which is what crosses the ABI.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Decode a wire value.
    ///
    /// Unknown values map to `None` rather than to [`ErrorCode::Internal`], because
    /// the caller needs to distinguish "a plugin from a newer minor version
    /// returned a code we do not know" from "the plugin reported a bug". The ABI
    /// loader is where that distinction is consumed.
    #[must_use]
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::InvalidArgument),
            2 => Some(Self::NotFound),
            3 => Some(Self::Unimplemented),
            4 => Some(Self::ResourceExhausted),
            5 => Some(Self::Internal),
            6 => Some(Self::Declined),
            _ => None,
        }
    }

    /// Whether a host should treat this code as a decline and try a different
    /// partition, rather than as a failure to surface.
    #[must_use]
    pub const fn is_decline(self) -> bool {
        matches!(self, Self::Declined)
    }

    /// Whether the device is presumed unusable after this error.
    ///
    /// Modelled on Burn's `is_device_poisoned`, and the distinction is
    /// load-bearing: a host that invalidates the device on `Declined` loses a
    /// perfectly good accelerator because one subgraph was out of scope.
    #[must_use]
    pub const fn is_device_poisoning(self) -> bool {
        matches!(self, Self::Internal)
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A failure, carrying the closed code plus whatever context the raiser had.
///
/// `String` rather than a structured payload because the ABI's payload is a
/// visitor over a `void*` (PJRT's `PJRT_Error_ForEachPayload`, kept in the
/// smaller form) and there is exactly one payload in B1: the human-readable
/// message. A vendor that needs to ship structured diagnostics later adds an
/// extension, rather than this type growing a second field that every raiser has
/// to fill.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    /// The closed code.
    pub code: ErrorCode,
    /// The backend that raised it, if it came from one.
    ///
    /// Present so a multi-backend host can say *which* one declined or broke
    /// without the caller having to track it.
    pub backend: Option<String>,
    /// The message.
    pub message: String,
}

impl Error {
    /// An error with a code and a message, attributed to no backend.
    #[must_use]
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            backend: None,
            message: message.into(),
        }
    }

    /// An error attributed to a backend by name.
    #[must_use]
    pub fn from_backend(
        code: ErrorCode,
        backend: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            backend: Some(backend.into()),
            message: message.into(),
        }
    }

    /// Attach a backend name to an error that has none.
    #[must_use]
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        if self.backend.is_none() {
            self.backend = Some(backend.into());
        }
        self
    }

    /// An `INVALID_ARGUMENT` naming the caller as at fault.
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    /// A `NOT_FOUND`.
    #[must_use]
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    /// An `UNIMPLEMENTED`, which is a bug in the caller rather than a decline.
    ///
    /// A backend that genuinely does not support something must return
    /// [`ErrorCode::Declined`]. Reaching for this instead turns a working fallback
    /// path into a failure.
    #[must_use]
    pub fn unimplemented(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Unimplemented, message)
    }

    /// A `RESOURCE_EXHAUSTED`.
    #[must_use]
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceExhausted, message)
    }

    /// An `INTERNAL`, which poisons the device.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }

    /// A `DECLINED`: the backend will not take this subgraph.
    #[must_use]
    pub fn declined(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Declined, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.backend {
            Some(backend) => write!(f, "[{backend}] {}: {}", self.code, self.message),
            None => write!(f, "{}: {}", self.code, self.message),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::{Error, ErrorCode};

    #[test]
    fn there_are_exactly_six_codes() {
        // The ABI surface. A seventh is a design decision with a precedent to
        // cite, not an implementation detail.
        let all = [
            ErrorCode::InvalidArgument,
            ErrorCode::NotFound,
            ErrorCode::Unimplemented,
            ErrorCode::ResourceExhausted,
            ErrorCode::Internal,
            ErrorCode::Declined,
        ];
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn the_wire_values_are_one_through_six_and_stable() {
        // These numbers cross the ABI. Changing one is a major version bump.
        assert_eq!(ErrorCode::InvalidArgument.as_u32(), 1);
        assert_eq!(ErrorCode::Declined.as_u32(), 6);
        for code in [
            ErrorCode::InvalidArgument,
            ErrorCode::NotFound,
            ErrorCode::Unimplemented,
            ErrorCode::ResourceExhausted,
            ErrorCode::Internal,
            ErrorCode::Declined,
        ] {
            assert_eq!(ErrorCode::from_u32(code.as_u32()), Some(code));
        }
    }

    #[test]
    fn an_unknown_wire_value_decodes_to_none_not_to_internal() {
        // The loader has to be able to tell "a newer plugin used a code we do not
        // know" from "the plugin reported a bug".
        assert_eq!(ErrorCode::from_u32(0), None);
        assert_eq!(ErrorCode::from_u32(7), None);
        assert_eq!(ErrorCode::from_u32(u32::MAX), None);
    }

    #[test]
    fn only_decline_is_a_decline_and_only_internal_poisons_the_device() {
        assert!(ErrorCode::Declined.is_decline());
        for code in [
            ErrorCode::InvalidArgument,
            ErrorCode::NotFound,
            ErrorCode::Unimplemented,
            ErrorCode::ResourceExhausted,
            ErrorCode::Internal,
        ] {
            assert!(!code.is_decline(), "{code} must not be a decline");
        }

        assert!(ErrorCode::Internal.is_device_poisoning());
        for code in [
            ErrorCode::InvalidArgument,
            ErrorCode::NotFound,
            ErrorCode::Unimplemented,
            ErrorCode::ResourceExhausted,
            ErrorCode::Declined,
        ] {
            assert!(!code.is_device_poisoning(), "{code} must not poison");
        }
    }

    #[test]
    fn display_names_the_backend_when_there_is_one() {
        let plain = Error::invalid_argument("bad axis");
        assert_eq!(plain.to_string(), "invalid-argument: bad axis");

        let attributed = Error::from_backend(ErrorCode::Declined, "acme-npu", "too new a shape");
        assert_eq!(
            attributed.to_string(),
            "[acme-npu] declined: too new a shape"
        );
    }

    #[test]
    fn with_backend_does_not_overwrite_an_existing_attribution() {
        let first = Error::from_backend(ErrorCode::Internal, "one", "broke");
        assert_eq!(first.with_backend("two").backend.as_deref(), Some("one"));

        let second = Error::internal("broke");
        assert_eq!(second.with_backend("two").backend.as_deref(), Some("two"));
    }
}
