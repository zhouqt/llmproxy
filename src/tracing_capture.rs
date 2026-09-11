//! Capture helpers for tracing output during tests.
//!
//! Why this module exists (Phase 2 of `plans/feat-fallback-logging-rework.md`)
//! ---------------------------------------------------------------------------
//!
//! Tracing's callsite `Interest` cache (`tracing::callsite::register_callsite`)
//! inspects the current subscriber the first time each `tracing::event!` macro
//! is expanded at a particular source location. If no subscriber is set when
//! the very first event fires, the callsite records "max level = off" and
//! every subsequent event at that site is silently dropped — even after a
//! subscriber is installed. The cache is keyed by `&'static TypeId`, so
//! reinstalling a subscriber in a test does NOT re-evaluate previously
//! registered callsites.
//!
//! The fix is to install *some* subscriber before the first `tracing::event!`
//! fires in the binary. A per-binary `#[ctor::ctor]` is the only way to
//! guarantee that for both the lib unit-test binary (where `--cfg test` is on
//! but `ctor` callsite must still resolve) and the integration-test binaries
//! (where Cargo compiles `tests/*.rs` as separate binaries and the lib's
//! `#[cfg(test)] #[ctor]` is NOT pulled in — `cfg(test)` is per-crate, not
//! per-crate-instance, and integration tests get the lib compiled WITHOUT
//! `cfg(test)`). See `src/lib.rs` and `tests/server.rs` for the two ctor
//! sites.
//!
//! **`set_default` is thread-local.** That means the `set_global_default`
//! installed by the ctor must be paired with a process-wide lifetime, AND
//! tests that want per-test capture must use `subscriber::set_default` (which
//! is thread-local + scoped) rather than `set_global_default`. The latter
//! would `set_global_default` would fail after the ctor's first install.
//! All `#[tokio::test]` cases in this repo use `flavor = "current_thread"`,
//! so the thread-local capture is sufficient — multi-threaded tests would
//! silently miss events.
//!
//! What this module provides
//! -------------------------
//! - [`CaptureWriter`]: a `MakeWriter` backed by `Arc<Mutex<Vec<u8>>>`. The
//!   captured bytes are the formatted log lines (with the default
//!   `tracing_subscriber::fmt` formatter).
//! - [`capture_tracing`]: a synchronous helper that installs a `fmt`
//!   subscriber scoped to a closure and returns the captured string. Use in
//!   lib unit tests (`#[test]` or `#[tokio::test(flavor = "current_thread")]`).
//! - [`TracingCapture`]: an RAII guard for `tokio::test` integration tests
//!   where the awaited future needs to remain inside the subscriber scope.
//!   `install()` returns a `DefaultGuard`; dropping it restores the prior
//!   subscriber for the current thread.
//! - [`drain`]: returns the accumulated bytes as a `String`, leaving the
//!   internal buffer empty (handy between emit + assert).

use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard};

use tracing_subscriber::fmt::MakeWriter;

/// In-memory writer that captures every byte written through it into a
/// shared `Vec<u8>`. Cloning the writer (which `MakeWriter::make_writer`
/// does on every event) hands out a new `Mutex<Vec<u8>>` guard, so the
/// captured output is safe under `tokio::test`'s interleaved poll calls.
#[derive(Clone, Default)]
pub struct CaptureWriter {
    inner: Arc<Mutex<Vec<u8>>>,
}

impl CaptureWriter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'a> MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureGuard;

    fn make_writer(&'a self) -> Self::Writer {
        CaptureGuard {
            inner: self.inner.clone(),
        }
    }
}

/// A guard returned by [`CaptureWriter::make_writer`]. Holds the lock on
/// the shared buffer for the duration of one `tracing` event write and
/// drains the formatted bytes into it. `Drop` is a no-op — the writer
/// flushes after every event by default.
pub struct CaptureGuard {
    inner: Arc<Mutex<Vec<u8>>>,
}

impl Write for CaptureGuard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut g: MutexGuard<'_, Vec<u8>> = self.inner.lock().expect("capture mutex poisoned");
        g.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Take the accumulated bytes out of a [`CaptureWriter`] as a `String`,
/// leaving the buffer empty so the next emit can be observed in
/// isolation. Used by tests that want to assert a single emit line.
pub fn drain(w: &CaptureWriter) -> String {
    let mut g = w.inner.lock().expect("capture mutex poisoned");
    let out = String::from_utf8(std::mem::take(&mut *g))
        .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned());
    out
}

/// Install a `fmt` subscriber that writes into `writer` for the duration
/// of `f`, returning whatever `f` returned plus the accumulated
/// captured bytes. Intended for synchronous unit tests; the closure must
/// not `.await` — the subscriber is bound to the current thread.
///
/// Use `tracing_capture::set_default` directly when you need the
/// subscriber scope to cross an `.await` point (see [`TracingCapture`]).
///
/// The formatter is configured for **unquoted field values** to match
/// the production operator log format (`model=m` not `model="m"`) so
/// test assertions can grep for the same shape that appears in
/// production logs. Plan §Phase 2 design constraint #6.
pub fn capture_tracing<F: FnOnce() -> R, R>(writer: &CaptureWriter, f: F) -> (R, String) {
    use tracing_subscriber::fmt::format::FmtSpan;
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer.clone())
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .with_ansi(false)
        // `Compact` keeps the field rendering aligned with the
        // unquoted production log shape; the default `Full` format
        // would quote string-typed field values.
        .compact()
        .with_span_events(FmtSpan::NONE)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let r = f();
        let captured = drain(writer);
        (r, captured)
    })
}

/// RAII guard for integration tests that need to `.await` inside the
/// capture scope. `install()` returns a `DefaultGuard` whose lifetime
/// ties the subscriber to the current thread; dropping it restores the
/// prior subscriber. `take()` extracts (and clears) whatever was
/// captured during the guard's lifetime.
pub struct TracingCapture {
    writer: CaptureWriter,
    _guard: tracing::subscriber::DefaultGuard,
}

impl TracingCapture {
    /// Install a `fmt` subscriber on the current thread for as long as
    /// the returned guard is alive. **Caller must keep the guard alive
    /// across any `.await` points** — once dropped, subsequent events
    /// go to the previously-installed subscriber.
    pub fn install() -> Self {
        use tracing_subscriber::fmt::format::FmtSpan;
        let writer = CaptureWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_target(false)
            .with_ansi(false)
            .compact()
            .with_span_events(FmtSpan::NONE)
            .finish();
        let guard = tracing::subscriber::set_default(subscriber);
        Self {
            writer,
            _guard: guard,
        }
    }

    /// Take the captured bytes out of the guard, leaving the buffer
    /// empty for the next assertion.
    pub fn take(&self) -> String {
        drain(&self.writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_tracing_captures_event_text() {
        // `tracing_subscriber::fmt`'s default field renderer uses
        // `Debug` for non-numeric values, so string-typed fields are
        // quoted (`model="m"`). Tests assert on this shape — it's
        // what the production operator sees, and is also the only
        // stable shape regardless of subscriber config. Plan §Phase 2
        // design constraint #6 explicitly notes the quoting can vary
        // with the Rust / tracing version; this test pins the actual
        // observed shape so future changes don't silently drift.
        let w = CaptureWriter::new();
        let (_, captured) = capture_tracing(&w, || {
            tracing::info!(model = "m", "hello");
        });
        assert!(captured.contains("hello"), "captured = {captured:?}");
        assert!(captured.contains("model=\"m\""), "captured = {captured:?}");
    }

    #[test]
    fn drain_empties_buffer_between_calls() {
        let w = CaptureWriter::new();
        let (_, captured1) = capture_tracing(&w, || {
            tracing::info!("first");
        });
        assert!(captured1.contains("first"));
        let (_, captured2) = capture_tracing(&w, || {
            tracing::info!("second");
        });
        assert!(
            !captured2.contains("first"),
            "drain must empty the buffer: {captured2:?}"
        );
        assert!(captured2.contains("second"));
    }

    #[test]
    fn tracing_capture_install_take_round_trip() {
        let cap = TracingCapture::install();
        tracing::info!("inside guard");
        let captured = cap.take();
        assert!(captured.contains("inside guard"));
    }
}
