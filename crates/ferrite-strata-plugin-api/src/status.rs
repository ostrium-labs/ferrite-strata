//! The `int` status code every fallible entry point returns.
//!
//! # The sign is the shape of the answer
//!
//! ```text
//!   0  OK              the call did what was asked
//!   1  DECLINED        not an error: "not mine", try another backend
//!  -2  UNIMPLEMENTED   this ABI version has no such entry point / feature
//!  -3  INVALID_ARGUMENT
//!  -4  NOT_FOUND
//!  -5  RESOURCE_EXHAUSTED
//!  -6  INTERNAL
//! ```
//!
//! C has no tri-state, so the ABI invents one by using the *sign* to separate
//! "not mine" from "that failed". A host that checks only `if (rc < 0)` — which
//! is what someone integrating this into something else under time pressure will
//! write — gets the safe answer: it treats a decline as a non-failure, and only a
//! genuine failure as an error. The one thing it gets wrong is that it will also
//! accept `DECLINED` as success, which is the failure mode the tri-state exists to
//! prevent, so `Status` in the host module still matches all three explicitly.
//!
//! # Decline is not an error
//!
//! [`DECLINED`] is positive and carries no `StrataError`. That is
//! [ADR-0006](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0006-compile-returns-a-tri-state-with-decline-as-a-first-class-outcome.md)
//! expressed in C: a plugin that cannot take a subgraph says so in the return
//! code, and a host that treats it as failure has a bug — but there is no
//! `is_refusal()` to reimplement, because there is no error object to inspect.

use ferrite_strata::error::ErrorCode;

/// The call succeeded.
pub const OK: i32 = 0;

/// The plugin understood the request and will not take it.
///
/// Not a failure. The host should re-partition or fall back. Carries no error
/// object, and the plugin must not have left anything allocated when it returns
/// this.
pub const DECLINED: i32 = 1;

/// The plugin has no implementation for this request.
///
/// A hard failure, unlike [`DECLINED`]: it means the host asked for something
/// outside the ABI rather than something the plugin chose not to do.
pub const UNIMPLEMENTED: i32 = -2;

/// The arguments were malformed: a bad id, a truncated blob, a null out-param.
pub const INVALID_ARGUMENT: i32 = -3;

/// A named thing does not exist: a foreign executable, a released buffer.
pub const NOT_FOUND: i32 = -4;

/// A limit was hit: memory, buffer count, subgraph size.
pub const RESOURCE_EXHAUSTED: i32 = -5;

/// A bug, in the plugin or in the host.
pub const INTERNAL: i32 = -6;

/// Every code this ABI version defines.
///
/// A list rather than a count, because the set is not contiguous once success and
/// decline are on the positive side, and a `STATUS_COUNT` that a caller used as a
/// range bound would be quietly wrong.
pub const ALL: [i32; 7] = [
    OK,
    DECLINED,
    UNIMPLEMENTED,
    INVALID_ARGUMENT,
    NOT_FOUND,
    RESOURCE_EXHAUSTED,
    INTERNAL,
];

/// A stable lowercase name, matching [`ErrorCode::name`].
///
/// The two sets are kept identical on purpose: a code that crosses the boundary
/// as an `int` and is turned back into an `ErrorCode` on the host side must land
/// on the same name it left with, or a log line says one thing while the error
/// says another.
#[must_use]
pub const fn name(code: i32) -> &'static str {
    match code {
        OK => "ok",
        DECLINED => "declined",
        UNIMPLEMENTED => "unimplemented",
        INVALID_ARGUMENT => "invalid-argument",
        NOT_FOUND => "not-found",
        RESOURCE_EXHAUSTED => "resource-exhausted",
        INTERNAL => "internal",
        _ => "unknown",
    }
}

/// Whether `code` means "try a different backend" rather than "surface this".
#[must_use]
pub const fn is_decline(code: i32) -> bool {
    code == DECLINED
}

/// Whether the device should be treated as unusable after `code`.
///
/// Modelled on Burn's `is_device_poisoned`. Only [`INTERNAL`] poisons: a host that
/// invalidated the device on a decline would lose a perfectly good accelerator
/// because one subgraph was out of scope.
#[must_use]
pub const fn is_device_poisoning(code: i32) -> bool {
    code == INTERNAL
}

/// Whether `code` was defined by this ABI version.
///
/// `false` for a code from a newer plugin, which is the distinction the whole
/// range is for: "a plugin newer than this host returned something we have never
/// heard of" is not "the plugin reported a bug", and collapsing them turns a
/// version skew into a support ticket.
#[must_use]
pub const fn is_known(code: i32) -> bool {
    matches!(
        code,
        OK | DECLINED
            | UNIMPLEMENTED
            | INVALID_ARGUMENT
            | NOT_FOUND
            | RESOURCE_EXHAUSTED
            | INTERNAL
    )
}

/// Map an ABI status code onto the core error taxonomy.
///
/// One function rather than reusing [`ErrorCode::from_u32`], because the two
/// numbering schemes are deliberately different: the core codes are `1..=6`
/// because they never needed to express "not an error", and the ABI codes are
/// signed because its callers cannot. Both sets of *names* are identical, which
/// is the property worth preserving.
#[must_use]
pub const fn to_error_code(code: i32) -> Option<ErrorCode> {
    match code {
        OK => None,
        DECLINED => Some(ErrorCode::Declined),
        UNIMPLEMENTED => Some(ErrorCode::Unimplemented),
        INVALID_ARGUMENT => Some(ErrorCode::InvalidArgument),
        NOT_FOUND => Some(ErrorCode::NotFound),
        RESOURCE_EXHAUSTED => Some(ErrorCode::ResourceExhausted),
        INTERNAL => Some(ErrorCode::Internal),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ALL, DECLINED, INTERNAL, INVALID_ARGUMENT, NOT_FOUND, OK, RESOURCE_EXHAUSTED,
        UNIMPLEMENTED, is_decline, is_device_poisoning, is_known, name, to_error_code,
    };
    use ferrite_strata::error::ErrorCode;

    #[test]
    fn success_and_decline_are_the_only_non_negative_codes() {
        // The `if (rc < 0)` almost-works property, stated as a test so renumbering
        // cannot quietly break it.
        const { assert!(OK >= 0) };
        const { assert!(DECLINED >= 0) };
        for code in [
            UNIMPLEMENTED,
            INVALID_ARGUMENT,
            NOT_FOUND,
            RESOURCE_EXHAUSTED,
            INTERNAL,
        ] {
            assert!(code < 0, "{code} must be negative");
        }
    }

    #[test]
    fn every_code_is_distinct() {
        let mut sorted = ALL.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "two status codes share a value");
    }

    #[test]
    fn a_decline_is_not_a_failure() {
        assert!(is_decline(DECLINED));
        assert!(!is_device_poisoning(DECLINED));
    }

    #[test]
    fn only_internal_poisons_the_device() {
        assert!(is_device_poisoning(INTERNAL));
        for code in [
            OK,
            DECLINED,
            UNIMPLEMENTED,
            INVALID_ARGUMENT,
            RESOURCE_EXHAUSTED,
        ] {
            assert!(!is_device_poisoning(code), "{code} poisoned the device");
        }
    }

    #[test]
    fn a_code_from_a_newer_plugin_is_unknown_rather_than_mapped() {
        let future = 4242;
        assert!(!is_known(future));
        assert_eq!(name(future), "unknown");
        assert_eq!(to_error_code(future), None);
        assert_eq!(to_error_code(OK), None, "success is not an error code");
    }

    #[test]
    fn the_two_code_sets_share_their_names() {
        // A code crossing the boundary and becoming an `ErrorCode` again must land
        // on the same name, or a log and an error message disagree.
        for code in ALL {
            let Some(error_code) = to_error_code(code) else {
                assert_eq!(code, OK);
                continue;
            };
            assert_eq!(name(code), error_code.name(), "code {code}");
        }
    }

    #[test]
    fn every_error_code_reachable_over_the_abi() {
        // Every core error code except `Declined` has an ABI status, and `Declined`
        // has one too, so the mapping is total over the taxonomy.
        let reachable = [
            ErrorCode::InvalidArgument,
            ErrorCode::NotFound,
            ErrorCode::Unimplemented,
            ErrorCode::ResourceExhausted,
            ErrorCode::Internal,
            ErrorCode::Declined,
        ];
        for expected in reachable {
            assert!(
                ALL.iter().any(|&c| to_error_code(c) == Some(expected)),
                "{expected} is not reachable"
            );
        }
    }
}
