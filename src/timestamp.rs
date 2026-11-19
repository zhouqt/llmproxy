//! Minimal UTC timestamp with RFC 3339 parsing and formatting.
//!
//! This replaces `chrono`, which is unmaintained (RUSTSEC-2026-0543,
//! `patched = []`) and pulled in `iana-time-zone` and its platform
//! shims. The proxy needs exactly three things: "what time is it now"
//! ([`Timestamp::now`]), parsing an operator-supplied RFC 3339 bound
//! ([`Timestamp::parse_rfc3339`]), and formatting a value back as a
//! JSON string ([`Timestamp::to_rfc3339`], also the serde impl).
//!
//! The wire format matches what `chrono` produced before the swap:
//! `SecondsFormat::AutoSi` (0, 3, 6 or 9 fractional digits) with a
//! `Z` suffix, so `/admin/usage` responses keep their exact shape.
//!
//! Only UTC is representable. A parsed offset is applied immediately
//! and discarded, as before (`parse_from_rfc3339(..).with_timezone(&Utc)`).

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::fmt::Write as _;
use std::ops::Add;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const NANOS_PER_SEC: u32 = 1_000_000_000;
const SECS_PER_DAY: i64 = 86_400;

/// A point in time, stored as seconds/nanoseconds since the Unix
/// epoch in UTC. `nanos` is always in `0..1_000_000_000`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    secs: i64,
    nanos: u32,
}

impl Timestamp {
    /// The current wall-clock time.
    pub fn now() -> Self {
        let (secs, nanos) = system_time_to_parts(SystemTime::now());
        Self { secs, nanos }
    }

    /// Build a timestamp from a Unix second count and a sub-second
    /// nanosecond count. Returns `None` if `nanos >= 1_000_000_000`.
    pub fn from_unix(secs: i64, nanos: u32) -> Option<Self> {
        if nanos >= NANOS_PER_SEC {
            return None;
        }
        Some(Self { secs, nanos })
    }

    /// Whole seconds since the Unix epoch, rounded toward negative
    /// infinity (the floor). This is the same value chrono's
    /// `DateTime::timestamp()` returns for the equivalent instant.
    pub fn unix_timestamp(&self) -> i64 {
        self.secs
    }

    /// Nanoseconds since the Unix epoch. Exact for the whole `i64`
    /// range of seconds.
    pub fn unix_timestamp_nanos(&self) -> i128 {
        self.secs as i128 * NANOS_PER_SEC as i128 + self.nanos as i128
    }

    /// Format as RFC 3339 in UTC.
    ///
    /// Fractional seconds use `chrono`'s `AutoSi` rule so the output
    /// is byte-identical to the old code: none when zero, otherwise
    /// 3, 6 or 9 digits depending on the smallest exact precision.
    ///
    /// Years outside `0..=9999` use the ISO 8601 explicit-sign form
    /// chrono uses (`+10000`, `-0001`); only [`Timestamp::from_unix`]
    /// with extreme values can construct those.
    pub fn to_rfc3339(&self) -> String {
        let days = self.secs.div_euclid(SECS_PER_DAY);
        let sod = self.secs.rem_euclid(SECS_PER_DAY) as u32;
        let (year, month, day) = civil_from_days(days);
        let hour = sod / 3600;
        let minute = (sod % 3600) / 60;
        let second = sod % 60;

        let mut out = String::with_capacity(32);
        if (0..=9999).contains(&year) {
            let _ = write!(out, "{year:04}");
        } else {
            let _ = write!(out, "{year:+05}");
        }
        // `write!` to a String cannot fail.
        let _ = write!(
            out,
            "-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
        );
        if self.nanos != 0 {
            if self.nanos % 1_000_000 == 0 {
                let _ = write!(out, ".{:03}", self.nanos / 1_000_000);
            } else if self.nanos % 1_000 == 0 {
                let _ = write!(out, ".{:06}", self.nanos / 1_000);
            } else {
                let _ = write!(out, ".{:09}", self.nanos);
            }
        }
        out.push('Z');
        out
    }

    /// Parse an RFC 3339 timestamp.
    ///
    /// The accepted grammar follows `chrono::DateTime::parse_from_rfc3339`:
    ///
    /// - `YYYY-MM-DD`, separator `T`, `t` or a space, `HH:MM:SS`;
    /// - optional `.` followed by one or more digits (only the first
    ///   nine are significant, the rest are ignored);
    /// - mandatory zone: `Z`, `z`, or `±HH:MM` (the sign may also be
    ///   U+2212 MINUS SIGN, which chrono's strict parser accepts) with
    ///   a colon and an offset below 24 hours.
    ///
    /// Calendar fields are validated (month 1–12, day against month
    /// length and leap years, hour 0–23, minute 0–59). A second of
    /// `60` is accepted as a leap second and normalised to the
    /// following second, preserving any fractional part — the proxy
    /// only compares parsed bounds, it never formats one back.
    ///
    /// Trailing characters are rejected. The scanner works on bytes
    /// and only ever slices at positions already validated as ASCII,
    /// so multi-byte input cannot panic on a non-UTF-8 boundary.
    pub fn parse_rfc3339(s: &str) -> Result<Self, ParseError> {
        let b = s.as_bytes();
        if b.len() < 19 {
            return Err(ParseError::new("timestamp is too short"));
        }

        let year = digits(b, 0, 4)? as i64;
        expect(b, 4, b'-')?;
        let month = digits(b, 5, 2)?;
        expect(b, 7, b'-')?;
        let day = digits(b, 8, 2)?;
        if !matches!(b[10], b'T' | b't' | b' ') {
            return Err(ParseError::new("expected 'T' between date and time"));
        }
        let hour = digits(b, 11, 2)?;
        expect(b, 13, b':')?;
        let minute = digits(b, 14, 2)?;
        expect(b, 16, b':')?;
        let second = digits(b, 17, 2)?;
        let mut i = 19;

        // Fractional seconds: at least one digit, first nine significant.
        let mut nanos = 0u32;
        if b.get(i) == Some(&b'.') {
            i += 1;
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                if i - start < 9 {
                    nanos = nanos * 10 + (b[i] - b'0') as u32;
                }
                i += 1;
            }
            let count = i - start;
            if count == 0 {
                return Err(ParseError::new("expected digits after '.'"));
            }
            // Scale a short fraction up to nanoseconds (".5" = 500ms).
            for _ in count..9 {
                nanos *= 10;
            }
        }

        // Zone. `+`/`-`/U+2212 then `HH:MM`; a bare `Z` is UTC.
        let offset_secs: i64 = match b.get(i) {
            Some(b'Z') | Some(b'z') => {
                i += 1;
                0
            }
            Some(&c) if c == b'+' || c == b'-' || b[i..].starts_with("\u{2212}".as_bytes()) => {
                let (negative, sign_len) = if c == b'+' {
                    (false, 1)
                } else if c == b'-' {
                    (true, 1)
                } else {
                    (true, '\u{2212}'.len_utf8())
                };
                let digits_at = i + sign_len;
                let hh = digits(b, digits_at, 2)?;
                expect(b, digits_at + 2, b':')?;
                let mm = digits(b, digits_at + 3, 2)?;
                if hh > 23 || mm > 59 {
                    return Err(ParseError::new("timezone offset out of range"));
                }
                i = digits_at + 5;
                let magnitude = (hh * 3600 + mm * 60) as i64;
                if negative {
                    -magnitude
                } else {
                    magnitude
                }
            }
            Some(_) => return Err(ParseError::new("expected a timezone offset")),
            None => return Err(ParseError::new("missing timezone offset")),
        };

        if i != b.len() {
            return Err(ParseError::new("trailing characters after timestamp"));
        }

        if !(1..=12).contains(&month) {
            return Err(ParseError::new("month out of range"));
        }
        if day == 0 || day > days_in_month(year, month) {
            return Err(ParseError::new("day out of range"));
        }
        if hour > 23 {
            return Err(ParseError::new("hour out of range"));
        }
        if minute > 59 {
            return Err(ParseError::new("minute out of range"));
        }
        if second > 60 {
            return Err(ParseError::new("second out of range"));
        }

        let mut secs = days_from_civil(year, month, day) * SECS_PER_DAY
            + (hour * 3600 + minute * 60 + second.min(59)) as i64;
        if second == 60 {
            // Leap second: normalise to the following second, keeping
            // the fractional part (`23:59:60.5` -> `00:00:00.5`).
            // chrono represents this as `:59` + 1e9 ns internally; the
            // two forms compare to the same ordering for all bounds
            // this proxy applies.
            secs += 1;
        }
        secs -= offset_secs;

        Ok(Self { secs, nanos })
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self {
        let mut nanos = self.nanos as u64 + rhs.subsec_nanos() as u64;
        let carry = (nanos >= NANOS_PER_SEC as u64) as i128;
        if carry == 1 {
            nanos -= NANOS_PER_SEC as u64;
        }
        // Saturate rather than wrap at the top of the range.
        let secs = (self.secs as i128 + rhs.as_secs() as i128 + carry)
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        Self {
            secs,
            nanos: nanos as u32,
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_rfc3339())
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({})", self.to_rfc3339())
    }
}

/// Serializes as an RFC 3339 string, byte-identical to chrono's
/// `DateTime<Utc>` (`AutoSi`, trailing `Z`).
impl Serialize for Timestamp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// Deserializes a strict RFC 3339 string.
///
/// This is intentionally stricter than chrono's serde impl, which goes
/// through `FromStr`/`parse_rfc3339_relaxed` (unpadded fields, spaces
/// between components, `UTC` as a zone, signed or 5-digit years, a
/// missing colon in the offset, trailing whitespace). Nothing in the
/// proxy deserializes a `Timestamp` in production — `UsageRecord` is
/// only constructed in code, and its `Deserialize` exists for tests —
/// so the strict parser is the single, predictable grammar here.
impl<'de> Deserialize<'de> for Timestamp {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TimestampVisitor;

        impl Visitor<'_> for TimestampVisitor {
            type Value = Timestamp;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an RFC 3339 timestamp string")
            }

            fn visit_str<E: de::Error>(self, s: &str) -> Result<Timestamp, E> {
                Timestamp::parse_rfc3339(s).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(TimestampVisitor)
    }
}

/// Error returned by [`Timestamp::parse_rfc3339`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseError {
    msg: &'static str,
}

impl ParseError {
    fn new(msg: &'static str) -> Self {
        Self { msg }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Split a `SystemTime` into floor-normalised `(seconds, nanoseconds)`.
///
/// Kept as a free function so the pre-epoch branch is testable without
/// a clock injection (a running test is, in practice, always after
/// 1970).
fn system_time_to_parts(time: SystemTime) -> (i64, u32) {
    match time.duration_since(UNIX_EPOCH) {
        Ok(d) => (d.as_secs().min(i64::MAX as u64) as i64, d.subsec_nanos()),
        Err(e) => {
            // Before the epoch: the duration is the magnitude of a
            // negative offset, so round toward negative infinity.
            let d = e.duration();
            let secs = d.as_secs().min(i64::MAX as u64) as i64;
            if d.subsec_nanos() == 0 {
                (-secs, 0)
            } else {
                (-secs - 1, NANOS_PER_SEC - d.subsec_nanos())
            }
        }
    }
}

/// Read exactly `n` ASCII digits starting at `start`.
fn digits(b: &[u8], start: usize, n: usize) -> Result<u32, ParseError> {
    let end = start
        .checked_add(n)
        .filter(|&end| end <= b.len())
        .ok_or_else(|| ParseError::new("truncated timestamp"))?;
    let mut value = 0u32;
    for &c in &b[start..end] {
        if !c.is_ascii_digit() {
            return Err(ParseError::new("expected a decimal digit"));
        }
        value = value * 10 + (c - b'0') as u32;
    }
    Ok(value)
}

fn expect(b: &[u8], i: usize, want: u8) -> Result<(), ParseError> {
    if b.get(i) == Some(&want) {
        Ok(())
    } else {
        Err(ParseError::new("unexpected character in timestamp"))
    }
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days since 1970-01-01 from a proleptic Gregorian date
/// (Howard Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (month as i64 + 9) % 12; // March = 0
    let doy = (153 * mp + 2) / 5 + day as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil`] (Howard Hinnant's `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // March = 0
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { y + 1 } else { y };
    (year, month, day)
}
