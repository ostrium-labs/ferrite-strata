//! Node attributes: the typed key/value pairs an op carries.
//!
//! # Why attributes are data rather than per-op fields
//!
//! Two reasons, both about the boundary rather than about convenience.
//!
//! **The vendor boundary is a blob.** `compile` receives the subgraph as opaque
//! bytes. If every op had bespoke Rust fields, serialisation would need a
//! per-op encoder that every future op version has to extend, and a vendor's
//! parser has to know every op it has never heard of. A key/value bag with a
//! self-describing encoding means an unknown key is *skipped*, not a parse
//! failure — which is what lets a subgraph built by a newer Strata still reach an
//! older vendor as something it can decline rather than reject.
//!
//! **Fusion needs names.** A fused pattern is matched against a pattern id
//! (an extensible vendor-owned string, per
//! [ADR-0005](https://github.com/ostrium-labs/ferrite-strata/blob/dev/docs/adr/0005-capabilities-are-versioned-data-plus-a-pull-supports-query.md)),
//! and a subgraph handed to `compile` has to survive the trip without the vendor
//! having to link our IR crate. Attributes are where the *parameters* of a fused
//! pattern live, so the vendor can read `head_dim` without knowing what a
//! `FlashAttentionV3` node is.
//!
//! # Keys are checked, not trusted
//!
//! Attribute keys are snake_case and validated at insertion. An unvalidated key
//! in a wire format is a silent collision waiting to happen: `axis` and `Axis`
//! are the same key to a human and different keys to a map, and a vendor reading
//! the wrong one gets a plausible wrong answer.

use core::fmt;

/// A typed attribute value.
///
/// Deliberately narrow. There is no variant for a nested list or an arbitrary
/// map, because both would let an op carry structure that no vendor can
/// interpret without linking our IR crate — which is exactly the coupling the
/// plugin boundary exists to avoid.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    /// A signed integer, for axes, dimensions and counts.
    Int(i64),
    /// A float, for scales and epsilons. `f32` specifically: attribute floats
    /// describe numeric parameters of a graph, and a `f64` here would imply a
    /// precision the vendor is not obliged to honour.
    Float(f32),
    /// A boolean.
    Bool(bool),
    /// A string, for names and for the pattern id of a vendor-owned op.
    Str(String),
    /// A homogeneous list of integers, for an axis set or a permutation.
    Ints(Vec<i64>),
}

impl AttrValue {
    /// This value as an `i64`, if it is one.
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// This value as an `f32`, if it is one.
    #[must_use]
    pub const fn as_float(&self) -> Option<f32> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// This value as a `bool`, if it is one.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// This value as a `&str`, if it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v),
            _ => None,
        }
    }

    /// This value as a slice of integers, if it is one.
    #[must_use]
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Self::Ints(v) => Some(v),
            _ => None,
        }
    }

    /// The type name, for a type-mismatch message.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Bool(_) => "bool",
            Self::Str(_) => "str",
            Self::Ints(_) => "ints",
        }
    }
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v:?}"),
            Self::Ints(v) => {
                f.write_str("[")?;
                for (i, item) in v.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// A validated attribute key.
///
/// A newtype rather than a `String` so an unvalidated key cannot reach a map: the
/// only way to hold one is through [`AttrKey::new`], which checks the shape.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AttrKey(String);

impl AttrKey {
    /// Validate and wrap a key.
    ///
    /// # Errors
    ///
    /// If the key is empty, is longer than 64 bytes, or contains anything other
    /// than `a`–`z`, `0`–`9` and `_`, or does not start with a letter.
    ///
    /// The 64-byte ceiling is not cosmetic: keys go into a wire format, and an
    /// unbounded key length is an unbounded allocation in a plugin's parser.
    pub fn new(key: &str) -> Result<Self, AttrError> {
        if key.is_empty() {
            return Err(AttrError::EmptyKey);
        }
        if key.len() > 64 {
            return Err(AttrError::KeyTooLong {
                len: key.len(),
                limit: 64,
            });
        }
        let mut chars = key.chars();
        let first = chars.next().unwrap_or('\0');
        if !first.is_ascii_lowercase() {
            return Err(AttrError::MalformedKey {
                key: key.to_string(),
                reason: "a key must start with an ASCII lowercase letter",
            });
        }
        for ch in key.chars() {
            let ok = ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_';
            if !ok {
                return Err(AttrError::MalformedKey {
                    key: key.to_string(),
                    reason: "a key may contain only ASCII lowercase letters, digits and underscores",
                });
            }
        }
        Ok(Self(key.to_string()))
    }

    /// The key as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AttrKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Something wrong with an attribute or its key.
///
/// Only `PartialEq`, not `Eq`: [`AttrError::DuplicateKey`] carries two
/// [`AttrValue`]s, and `AttrValue` holds an `f32`.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrError {
    /// An empty key.
    EmptyKey,
    /// A key longer than the ceiling.
    KeyTooLong {
        /// The key's length in bytes.
        len: usize,
        /// The ceiling that was exceeded.
        limit: usize,
    },
    /// A key with characters outside the permitted set.
    MalformedKey {
        /// The offending key.
        key: String,
        /// What is wrong with it, phrased as the rule.
        reason: &'static str,
    },
    /// The same key was set twice on one node.
    DuplicateKey {
        /// The key that collided.
        key: String,
        /// The value that was already there.
        first: AttrValue,
        /// The value that was refused.
        second: AttrValue,
    },
    /// An attribute was read as the wrong type.
    TypeMismatch {
        /// The key that was read.
        key: String,
        /// The type it actually holds.
        found: &'static str,
        /// The type the caller asked for.
        wanted: &'static str,
    },
}

impl fmt::Display for AttrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttrError::EmptyKey => f.write_str(
                "an attribute key cannot be empty; an unnamed attribute has no \
                 address, so the second one set would silently overwrite the first",
            ),
            AttrError::KeyTooLong { len, limit } => write!(
                f,
                "an attribute key of {len} bytes exceeds the {limit}-byte limit. \
                 Keys go into a wire format, and an unbounded key length is an \
                 unbounded allocation in a plugin's parser."
            ),
            AttrError::MalformedKey { key, reason } => {
                write!(f, "the attribute key `{key}` is malformed: {reason}.")
            }
            AttrError::DuplicateKey { key, first, second } => write!(
                f,
                "the attribute `{key}` is already {first}, so the later {second} was \
                 refused. Two values for one key is an ambiguity, and the subgraph \
                 that reaches a vendor has to have exactly one answer for it."
            ),
            AttrError::TypeMismatch { key, found, wanted } => write!(
                f,
                "the attribute `{key}` is a {found}, not a {wanted}. The op \
                 declares which type each key holds, so this is either an op that \
                 is missing the attribute or one carrying it with the wrong type."
            ),
        }
    }
}

impl std::error::Error for AttrError {}

/// One node's attributes, insertion-ordered.
///
/// A `Vec` of pairs rather than a `BTreeMap`: nodes carry two or three
/// attributes, so a map's log-factor lookup and per-node allocation cost more than
/// the linear scan, and the duplicate check needs a scan anyway. Order is
/// preserved because a serialised subgraph has to be byte-identical across two
/// builds of the same graph — a property the partitioner's caching relies on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attributes {
    entries: Vec<(AttrKey, AttrValue)>,
}

impl Attributes {
    /// An empty set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Set an attribute.
    ///
    /// # Errors
    ///
    /// If the key is already set. Not an overwrite, deliberately: see
    /// [`AttrError::DuplicateKey`].
    pub fn set(&mut self, key: AttrKey, value: AttrValue) -> Result<(), AttrError> {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            return Err(AttrError::DuplicateKey {
                key: key.as_str().to_string(),
                first: slot.1.clone(),
                second: value,
            });
        }
        self.entries.push((key, value));
        Ok(())
    }

    /// Look an attribute up by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&AttrValue> {
        self.entries
            .iter()
            .find(|(k, _)| k.as_str() == key)
            .map(|(_, v)| v)
    }

    /// Look an attribute up, requiring a particular type.
    ///
    /// `wanted` is `&'static str` rather than `&str` because it is stored in the
    /// error, and a message that borrowed a caller's buffer would not outlive the
    /// stack frame the caller is in.
    ///
    /// # Errors
    ///
    /// If the key is absent, with a message naming the op's requirement, or if it
    /// holds a different type.
    pub fn require(&self, key: &str, wanted: &'static str) -> Result<&AttrValue, AttrError> {
        match self.get(key) {
            None => Err(AttrError::TypeMismatch {
                key: key.to_string(),
                found: "absent",
                wanted,
            }),
            Some(value) => {
                let matches = matches!(
                    (value, wanted),
                    (AttrValue::Int(_), "int")
                        | (AttrValue::Float(_), "float")
                        | (AttrValue::Bool(_), "bool")
                        | (AttrValue::Str(_), "str")
                        | (AttrValue::Ints(_), "ints")
                );
                if matches {
                    Ok(value)
                } else {
                    Err(AttrError::TypeMismatch {
                        key: key.to_string(),
                        found: value.type_name(),
                        wanted,
                    })
                }
            }
        }
    }

    /// Every pair, in insertion order.
    #[must_use]
    pub fn entries(&self) -> &[(AttrKey, AttrValue)] {
        &self.entries
    }

    /// How many attributes are set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no attributes are set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{AttrError, AttrKey, AttrValue, Attributes};

    #[test]
    fn a_well_formed_key_is_accepted() {
        for key in ["axis", "head_dim", "epsilon2", "a1"] {
            assert_eq!(
                AttrKey::new(key).expect("well formed").as_str(),
                key,
                "{key} should be accepted"
            );
        }
    }

    #[test]
    fn an_empty_key_is_rejected() {
        assert_eq!(AttrKey::new("").expect_err("empty"), AttrError::EmptyKey);
    }

    #[test]
    fn a_key_that_does_not_start_lowercase_is_rejected() {
        // `Axis` and `axis` are the same key to a human and different keys to a
        // map, which is the whole reason keys are validated.
        assert!(matches!(
            AttrKey::new("Axis").expect_err("leading capital"),
            AttrError::MalformedKey { .. }
        ));
        assert!(matches!(
            AttrKey::new("_axis").expect_err("leading underscore"),
            AttrError::MalformedKey { .. }
        ));
        assert!(matches!(
            AttrKey::new("1axis").expect_err("leading digit"),
            AttrError::MalformedKey { .. }
        ));
    }

    #[test]
    fn a_key_with_punctuation_is_rejected() {
        assert!(matches!(
            AttrKey::new("head dim").expect_err("a space"),
            AttrError::MalformedKey { .. }
        ));
        assert!(matches!(
            AttrKey::new("head-dim").expect_err("a hyphen"),
            AttrError::MalformedKey { .. }
        ));
        assert!(matches!(
            AttrKey::new("head.dim").expect_err("a dot"),
            AttrError::MalformedKey { .. }
        ));
    }

    #[test]
    fn a_long_key_is_rejected_with_the_limit_named() {
        let long = "a".repeat(65);
        assert_eq!(
            AttrKey::new(&long).expect_err("65 bytes"),
            AttrError::KeyTooLong { len: 65, limit: 64 }
        );
    }

    #[test]
    fn setting_the_same_key_twice_is_an_error_not_an_overwrite() {
        let key = AttrKey::new("axis").expect("valid");
        let mut attrs = Attributes::new();
        attrs
            .set(key.clone(), AttrValue::Int(-1))
            .expect("first set");

        let err = attrs
            .set(key, AttrValue::Int(1))
            .expect_err("second set of the same key");
        match err {
            AttrError::DuplicateKey {
                ref first,
                ref second,
                ..
            } => {
                assert_eq!(first, &AttrValue::Int(-1));
                assert_eq!(second, &AttrValue::Int(1));
            }
            other => panic!("expected a duplicate-key error, got {other:?}"),
        }
        assert_eq!(attrs.get("axis"), Some(&AttrValue::Int(-1)));
    }

    #[test]
    fn insertion_order_is_preserved() {
        let mut attrs = Attributes::new();
        for (k, v) in [("z", 1), ("a", 2), ("m", 3)] {
            attrs
                .set(AttrKey::new(k).expect("valid"), AttrValue::Int(v))
                .expect("first set");
        }
        let keys: Vec<&str> = attrs.entries().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["z", "a", "m"]);
        assert_eq!(attrs.len(), 3);
        assert!(!attrs.is_empty());
    }

    #[test]
    fn require_distinguishes_absent_from_wrong_type() {
        let mut attrs = Attributes::new();
        attrs
            .set(
                AttrKey::new("axis").expect("valid"),
                AttrValue::Str("not an int".into()),
            )
            .expect("first set");

        let wrong_type = attrs
            .require("axis", "int")
            .expect_err("a str is not an int");
        assert_eq!(
            wrong_type,
            AttrError::TypeMismatch {
                key: "axis".into(),
                found: "str",
                wanted: "int",
            }
        );

        let absent = attrs.require("head_dim", "int").expect_err("not set");
        assert!(absent.to_string().contains("absent"), "{absent}");
    }

    #[test]
    fn every_value_type_round_trips_through_its_accessor() {
        assert_eq!(AttrValue::Int(-3).as_int(), Some(-3));
        assert_eq!(AttrValue::Float(0.5).as_float(), Some(0.5));
        assert_eq!(AttrValue::Bool(true).as_bool(), Some(true));
        assert_eq!(AttrValue::Str("x".into()).as_str(), Some("x"));
        assert_eq!(
            AttrValue::Ints(vec![1, 2]).as_ints(),
            Some([1, 2].as_slice())
        );
    }

    #[test]
    fn an_accessor_on_the_wrong_type_is_none_not_a_panic() {
        assert_eq!(AttrValue::Int(1).as_float(), None);
        assert_eq!(AttrValue::Str("x".into()).as_int(), None);
        assert_eq!(AttrValue::Bool(false).as_ints(), None);
    }
}
