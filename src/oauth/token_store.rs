//! Token persistence for GitHub OAuth + Copilot API tokens.

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::{ProxyError, Result};
use crate::oauth::token_crypto::{open, parse_key, seal, TokenEnvelope};

/// Persisted GitHub OAuth + Copilot API tokens.
///
/// The two token fields are **plaintext `String`s on purpose**: they are
/// the on-disk representation, and `write_atomic` has to persist the real
/// bytes. `secrecy::SecretString` is therefore *not* usable here — its
/// `Serialize` impl emits `"[REDACTED]"`, which would corrupt the
/// persisted file (see the audit note in
/// plans/fix-security-audit-2026-10-25.md, phase 2).
///
/// What we seal instead is the *log face*: `Debug` is hand-written below
/// so that a `{:?}` of anything holding tokens can never print them.
/// Every other credential in the codebase lives in a `SecretString`; this
/// is the one place where the secret has to exist in the clear, so the
/// Debug impl is the invariant's enforcement point.
#[derive(Clone, Serialize, Deserialize)]
pub struct StoredTokens {
    pub github_access_token: String,
    pub copilot_token: String,
    pub copilot_expires_at: i64,
    pub refresh_in: i64,
}

impl fmt::Debug for StoredTokens {
    /// Prints every field redacted. The token fields report their byte
    /// length only — length is not secret and is useful when debugging a
    /// truncated/rotated token.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoredTokens")
            .field(
                "github_access_token",
                &format_args!("[REDACTED {} bytes]", self.github_access_token.len()),
            )
            .field(
                "copilot_token",
                &format_args!("[REDACTED {} bytes]", self.copilot_token.len()),
            )
            .field("copilot_expires_at", &"[REDACTED]")
            .field("refresh_in", &"[REDACTED]")
            .finish()
    }
}

impl StoredTokens {
    pub fn data_dir() -> Result<PathBuf> {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
            })
            .ok_or_else(|| ProxyError::Other(anyhow::anyhow!("no XDG_DATA_HOME or HOME set")))?;
        Ok(base.join("llmproxy"))
    }

    pub fn path() -> Result<PathBuf> {
        Ok(Self::data_dir()?.join("github_token.json"))
    }
}

#[derive(Clone)]
pub struct TokenStore {
    path: PathBuf,
}

impl TokenStore {
    pub fn new() -> Result<Self> {
        let path = StoredTokens::path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self { path })
    }

    #[cfg(test)]
    pub(crate) fn from_path(path: PathBuf) -> Self {
        Self { path }
    }

    /// Load the stored tokens.
    ///
    /// `key` is the provider's `token_store_key` — `None` means the
    /// operator did not configure one and the store is (or should be)
    /// plaintext.
    ///
    /// Three shapes are recognized, in this order:
    ///
    /// 1. **Plaintext** — the body parses as [`StoredTokens`]. It is
    ///    returned as-is, *regardless of `key`*. This is the migration
    ///    path: an existing plaintext store keeps working, and the next
    ///    `save` rewrites it encrypted.
    /// 2. **Envelope** — the body parses as a [`TokenEnvelope`] carrying
    ///    our magic string. Decrypting it requires `key`:
    ///    * `None` → warn and return `Ok(None)`. A device-flow re-auth
    ///      beats bricking the process; the operator can fix the key
    ///      and the next load succeeds.
    ///    * `Some` → decrypt. A failure (wrong key, tampered file) also
    ///      warns and returns `Ok(None)` — we deliberately do **not**
    ///      delete the file, so the next successful authentication can
    ///      overwrite it. This mirrors the existing 401/403
    ///      clears-the-store behavior.
    /// 3. **Neither** → the historical corrupt-file handling: the
    ///    original parse error propagates and callers treat it as "no
    ///    tokens".
    pub fn load(&self, key: Option<&SecretString>) -> Result<Option<StoredTokens>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&self.path)?;

        // (1) Legacy plaintext store.
        if let Ok(tokens) = serde_json::from_str::<StoredTokens>(&raw) {
            return Ok(Some(tokens));
        }

        // (2) Encrypted store.
        if let Ok(envelope) = serde_json::from_str::<TokenEnvelope>(&raw) {
            if envelope.is_ours() {
                return self.load_encrypted(&envelope, key);
            }
        }

        // (3) Neither — keep the historical corrupt-file behavior and
        // surface the original parse error.
        Err(ProxyError::Json(
            serde_json::from_str::<StoredTokens>(&raw).unwrap_err(),
        ))
    }

    /// Decrypt an envelope. Never fails hard: a missing or wrong key
    /// degrades to "no tokens" with a warning, because bricking the
    /// proxy is worse than asking the operator to re-authenticate.
    fn load_encrypted(
        &self,
        envelope: &TokenEnvelope,
        key: Option<&SecretString>,
    ) -> Result<Option<StoredTokens>> {
        let Some(key) = key else {
            tracing::warn!(
                path = %self.path.display(),
                "token store is encrypted but provider.token_store_key is not configured; \
                 treating it as empty (re-authenticate via /admin/copilot/auth)"
            );
            return Ok(None);
        };
        let key_bytes = parse_key(key.expose_secret()).map_err(ProxyError::Config)?;
        match open(&key_bytes, envelope) {
            Ok(plain) => Ok(Some(serde_json::from_slice(&plain)?)),
            Err(e) => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = %e,
                    "failed to decrypt the token store; treating it as empty \
                     (the next successful authentication overwrites it)"
                );
                Ok(None)
            }
        }
    }

    /// Persist the tokens.
    ///
    /// With a `key` the tokens are sealed into an AES-256-GCM envelope
    /// (see [`crate::oauth::token_crypto`]); without one they are
    /// written as plaintext JSON exactly as before. The file permissions
    /// and atomicity are unchanged either way — `write_atomic` still
    /// creates the file 0600 and renames it into place.
    pub fn save(&self, tokens: &StoredTokens, key: Option<&SecretString>) -> Result<()> {
        let raw = match key {
            None => serde_json::to_string_pretty(tokens)?,
            Some(key) => {
                let key_bytes = parse_key(key.expose_secret()).map_err(ProxyError::Config)?;
                let plaintext = serde_json::to_vec(tokens)?;
                let envelope = seal(&key_bytes, &plaintext)?;
                serde_json::to_string_pretty(&envelope)?
            }
        };
        write_atomic(&self.path, raw.as_bytes())?;
        Ok(())
    }

    pub fn clear(&self) -> Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests_extra {
    use super::*;
    use crate::test_support::env_lock;

    #[test]
    fn data_dir_falls_back_to_home_when_xdg_unset() {
        // data_dir() reads XDG_DATA_HOME / HOME, and other tests unset both.
        // The guard serializes us against them and restores whatever we
        // change, including when an assertion below fails.
        let mut env = env_lock();
        env.remove_var("XDG_DATA_HOME");

        let dir = StoredTokens::data_dir().unwrap();

        assert!(dir.ends_with("llmproxy"));
        assert!(dir.to_string_lossy().contains(".local/share"));
    }

    #[test]
    fn data_dir_errors_when_both_unset() {
        let mut env = env_lock();
        env.remove_var("XDG_DATA_HOME");
        env.remove_var("HOME");

        let err = StoredTokens::data_dir().unwrap_err();

        assert!(err.to_string().contains("no XDG_DATA_HOME or HOME"));
    }

    #[test]
    fn new_creates_data_dir_when_missing() {
        // Point XDG_DATA_HOME at a fresh nested path so TokenStore::new must
        // create the missing parent directory.
        let mut env = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let xdg_root = tmp.path().join("xdg").join("nested");
        env.set_var("XDG_DATA_HOME", xdg_root.as_os_str());

        let store = TokenStore::new().unwrap();

        assert!(xdg_root.join("llmproxy").exists());
        assert_eq!(
            store.path(),
            xdg_root.join("llmproxy").join("github_token.json").as_path()
        );
    }

    #[test]
    fn clear_removes_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore {
            path: dir.path().join("github_token.json"),
        };
        let tokens = StoredTokens {
            github_access_token: "g".into(),
            copilot_token: "c".into(),
            copilot_expires_at: 0,
            refresh_in: 1,
        };
        store.save(&tokens, None).unwrap();
        assert!(store.path().exists());

        store.clear().unwrap();
        assert!(!store.path().exists());

        // Clearing again is idempotent and does not error.
        store.clear().unwrap();
    }
}

pub(crate) fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;

    let dir = path
        .parent()
        .ok_or_else(|| ProxyError::Other(anyhow::anyhow!("token path has no parent dir")))?;
    let unique = format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("github_token"),
        uuid::Uuid::new_v4().simple()
    );
    let tmp = dir.join(unique);

    let mut file = {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&tmp)?
        }
        #[cfg(not(unix))]
        {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?
        }
    };
    file.write_all(data)?;
    file.sync_all()?;
    drop(file);

    match std::fs::rename(&tmp, path) {
        Ok(()) => {}
        Err(e) => {
            // Best-effort cleanup of the orphaned temp file before propagating.
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
    }

    // Best-effort fsync on the parent directory so the rename is durable.
    #[cfg(unix)]
    {
        let _ = std::fs::File::open(dir).and_then(|f| f.sync_all());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_store() -> (tempfile::TempDir, TokenStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore {
            path: dir.path().join("github_token.json"),
        };
        (dir, store)
    }

    /// Asserts that a directory entry name is not a leftover temp file.
    /// Single canonical message; the failing name is in the panic payload.
    fn assert_no_tmp(name: &str) {
        assert!(
            !name.ends_with(".tmp") && !name.starts_with(".github_token.json."),
            "leftover tmp file {name} after save"
        );
    }

    #[test]
    fn round_trip_save_load() {
        let (_dir, store) = fresh_store();
        assert!(store.load(None).unwrap().is_none());

        let tokens = StoredTokens {
            github_access_token: "gh-xxx".into(),
            copilot_token: "co-yyy".into(),
            copilot_expires_at: 1234567890,
            refresh_in: 1500,
        };
        store.save(&tokens, None).unwrap();
        let loaded = store.load(None).unwrap().expect("file should exist");
        assert_eq!(loaded.github_access_token, "gh-xxx");
        assert_eq!(loaded.copilot_token, "co-yyy");
        assert_eq!(loaded.copilot_expires_at, 1234567890);

        store.clear().unwrap();
        assert!(store.load(None).unwrap().is_none());
    }

    #[test]
    fn load_missing_file_returns_none() {
        let (_dir, store) = fresh_store();
        assert!(store.load(None).unwrap().is_none());
    }

    #[test]
    fn debug_never_contains_token() {
        // The Debug impl is the only thing standing between a `{:?}` in a
        // log line and a live GitHub/Copilot credential.
        let tokens = StoredTokens {
            github_access_token: "gh-secret-sentinel".into(),
            copilot_token: "copilot-secret-sentinel".into(),
            copilot_expires_at: 1234567890,
            refresh_in: 1500,
        };

        let dumped = format!("{:?}", tokens);

        assert!(!dumped.contains("gh-secret-sentinel"), "{dumped}");
        assert!(!dumped.contains("copilot-secret-sentinel"), "{dumped}");
        assert!(dumped.contains("[REDACTED"), "{dumped}");
        // The byte length is not a secret and helps debug a truncated token.
        assert!(
            dumped.contains(&format!(
                "[REDACTED {} bytes]",
                tokens.github_access_token.len()
            )),
            "{dumped}"
        );
        assert!(
            dumped.contains(&format!("[REDACTED {} bytes]", tokens.copilot_token.len())),
            "{dumped}"
        );
    }

    #[test]
    fn save_sets_0600_on_unix() {
        let (_dir, store) = fresh_store();
        let tokens = StoredTokens {
            github_access_token: "g".into(),
            copilot_token: "c".into(),
            copilot_expires_at: 0,
            refresh_in: 1,
        };
        store.save(&tokens, None).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "expected 0600, got {:o}", mode);
        }
    }

    /// Envelope-encryption tests for the token store.
    ///
    /// No env vars are touched, so no `EnvGuard` is needed — the store
    /// is addressed through an explicit path.
    mod encryption {
        use super::*;
        use crate::secrets::secret_string;
        use crate::tracing_capture::CaptureWriter;

        const KEY_A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
        const KEY_B: &str = "2222222222222222222222222222222222222222222222222222222222222222";

        fn sentinel_tokens() -> StoredTokens {
            StoredTokens {
                github_access_token: "gh-plaintext-sentinel".into(),
                copilot_token: "copilot-plaintext-sentinel".into(),
                copilot_expires_at: 1234567890,
                refresh_in: 1500,
            }
        }

        #[test]
        fn encrypted_round_trip_hides_the_tokens_on_disk() {
            let (_dir, store) = fresh_store();
            let key = secret_string(KEY_A);
            let tokens = sentinel_tokens();

            store.save(&tokens, Some(&key)).unwrap();

            let raw = std::fs::read_to_string(store.path()).unwrap();
            assert!(
                raw.contains(crate::oauth::token_crypto::ENVELOPE_MAGIC),
                "{raw}"
            );
            assert!(!raw.contains("gh-plaintext-sentinel"), "{raw}");
            assert!(!raw.contains("copilot-plaintext-sentinel"), "{raw}");

            let loaded = store.load(Some(&key)).unwrap().expect("tokens load back");
            assert_eq!(loaded.github_access_token, tokens.github_access_token);
            assert_eq!(loaded.copilot_token, tokens.copilot_token);
            assert_eq!(loaded.copilot_expires_at, tokens.copilot_expires_at);
            assert_eq!(loaded.refresh_in, tokens.refresh_in);
        }

        #[test]
        fn wrong_key_yields_no_tokens_and_warns() {
            let (_dir, store) = fresh_store();
            let key_a = secret_string(KEY_A);
            let key_b = secret_string(KEY_B);
            store.save(&sentinel_tokens(), Some(&key_a)).unwrap();

            let writer = CaptureWriter::new();
            let (loaded, captured) =
                crate::tracing_capture::capture_tracing(&writer, || store.load(Some(&key_b)));

            assert!(loaded.unwrap().is_none());
            assert!(
                captured.contains("failed to decrypt the token store"),
                "captured = {captured:?}"
            );
        }

        #[test]
        fn encrypted_store_without_a_configured_key_yields_no_tokens_and_warns() {
            let (_dir, store) = fresh_store();
            let key = secret_string(KEY_A);
            store.save(&sentinel_tokens(), Some(&key)).unwrap();

            let writer = CaptureWriter::new();
            let (loaded, captured) =
                crate::tracing_capture::capture_tracing(&writer, || store.load(None));

            assert!(loaded.unwrap().is_none());
            assert!(
                captured.contains(
                    "token store is encrypted but provider.token_store_key is not configured"
                ),
                "captured = {captured:?}"
            );
        }

        #[test]
        fn plaintext_store_still_loads_when_a_key_is_configured() {
            // The migration path: an existing plaintext store keeps
            // working after the operator adds token_store_key, and the
            // next save rewrites it encrypted.
            let (_dir, store) = fresh_store();
            let tokens = sentinel_tokens();
            store.save(&tokens, None).unwrap();

            let key = secret_string(KEY_A);
            let loaded = store
                .load(Some(&key))
                .unwrap()
                .expect("plaintext store loads");
            assert_eq!(loaded.github_access_token, tokens.github_access_token);
            assert_eq!(loaded.copilot_token, tokens.copilot_token);
        }

        #[test]
        fn save_without_a_key_keeps_the_legacy_plaintext_wire() {
            let (_dir, store) = fresh_store();
            let tokens = sentinel_tokens();

            store.save(&tokens, None).unwrap();

            let raw = std::fs::read_to_string(store.path()).unwrap();
            assert!(
                !raw.contains(crate::oauth::token_crypto::ENVELOPE_MAGIC),
                "{raw}"
            );
            let parsed: StoredTokens = serde_json::from_str(&raw).expect("plaintext StoredTokens");
            assert_eq!(parsed.github_access_token, tokens.github_access_token);
        }

        #[test]
        fn two_saves_with_the_same_key_use_different_nonces() {
            let (_dir, store) = fresh_store();
            let key = secret_string(KEY_A);
            let tokens = sentinel_tokens();

            store.save(&tokens, Some(&key)).unwrap();
            let first = std::fs::read_to_string(store.path()).unwrap();
            store.save(&tokens, Some(&key)).unwrap();
            let second = std::fs::read_to_string(store.path()).unwrap();

            let first_nonce = serde_json::from_str::<TokenEnvelope>(&first).unwrap().nonce;
            let second_nonce = serde_json::from_str::<TokenEnvelope>(&second)
                .unwrap()
                .nonce;
            assert_ne!(first_nonce, second_nonce);
        }

        #[test]
        fn a_corrupt_store_surfaces_the_parse_error() {
            // Neither plaintext nor an envelope → the historical
            // corrupt-file behavior: the original parse error propagates.
            let (_dir, store) = fresh_store();
            std::fs::write(store.path(), b"{not json at all").unwrap();

            let err = store.load(None).unwrap_err();

            crate::expect_variant!(err, ProxyError::Json(_) => {});
        }

        #[test]
        fn an_envelope_with_a_foreign_magic_is_treated_as_corrupt() {
            // A file that *parses* as an envelope but carries someone
            // else's magic must not be decrypted — and must not be
            // silently ignored either. It falls through to the
            // historical corrupt-file path.
            let (_dir, store) = fresh_store();
            let foreign = serde_json::json!({
                "magic": "some-other-tool-v1",
                "alg": "A256GCM",
                "nonce": "",
                "ct": "",
            });
            std::fs::write(store.path(), serde_json::to_vec(&foreign).unwrap()).unwrap();

            let err = store.load(Some(&secret_string(KEY_A))).unwrap_err();

            crate::expect_variant!(err, ProxyError::Json(_) => {});
        }
    }

    #[test]
    fn atomic_write_leaves_no_tmp_files() {
        let (_dir, store) = fresh_store();
        let tokens = StoredTokens {
            github_access_token: "g".into(),
            copilot_token: "c".into(),
            copilot_expires_at: 0,
            refresh_in: 1,
        };

        store.save(&tokens, None).unwrap();
        store.save(&tokens, None).unwrap();

        for entry in std::fs::read_dir(store.path().parent().unwrap()).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let name = name.to_str().unwrap();
            assert_no_tmp(name);
        }
    }

    #[test]
    fn atomic_write_uses_unique_tmp_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("github_token.json");
        write_atomic(&path, b"first").unwrap();
        let first_tmp = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .find(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|name| name.starts_with(".github_token.json."))
                    .unwrap_or(false)
            });
        assert!(first_tmp.is_none(), "no tmp file should remain after rename");

        // Concurrent writers should not collide on tmp paths.
        let mut handles = Vec::new();
        let path2 = path.clone();
        for _ in 0..16 {
            let path_clone = path2.clone();
            handles.push(std::thread::spawn(move || {
                write_atomic(&path_clone, b"payload").unwrap();
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), b"payload");
    }

    #[test]
    fn atomic_write_returns_error_when_destination_is_a_directory() {
        // When the destination path already exists as a directory, the
        // rename step fails. The error path must clean up the orphaned
        // temp file (best-effort) and surface an io::Error.
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("github_token.json");
        std::fs::create_dir(&dest).unwrap();

        let err = write_atomic(&dest, b"payload").unwrap_err();

        // The temp file is best-effort cleaned up — assert no `.tmp` lingers.
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            let name = name.to_str().unwrap();
            assert!(!name.contains(".tmp"), "no tmp file should remain after failed rename");
        }
        assert!(matches!(err, ProxyError::Io(_)));
    }

    #[test]
    fn atomic_write_rejects_paths_without_a_parent() {
        // `Path::new("")` reports parent as None. write_atomic must surface
        // a clean Other error rather than panic when this happens.
        let err = write_atomic(std::path::Path::new(""), b"payload").unwrap_err();
        assert!(err.to_string().contains("token path has no parent dir"));
    }
}
