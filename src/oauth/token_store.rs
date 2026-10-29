//! Token persistence for GitHub OAuth + Copilot API tokens.

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::{ProxyError, Result};
use crate::oauth::token_crypto::{open, parse_key, seal, seal_migrated, TokenEnvelope};

/// Persisted GitHub OAuth + Copilot API tokens.
///
/// The two token fields are **plaintext `String`s on purpose**: they are
/// the on-disk representation, and `write_atomic` has to persist the real
/// bytes. `secrecy::SecretString` is therefore *not* usable here — it
/// has no `Serialize` impl at all (secrecy gates `Serialize` behind an
/// opt-in marker trait that `str` cannot implement), so persisting a
/// secret would require hand-writing a serializer that emits the real
/// bytes anyway (see the audit note in
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
    /// The rule is **"the key decides the format"**:
    ///
    /// * **With a key** — only *our* envelope is accepted. A legacy
    ///   plaintext store is *migrated* (re-written as an envelope, so
    ///   the host disk stops holding plaintext) but **not trusted**:
    ///   its contents were never authenticated, so anybody who could
    ///   write that file could have planted their own token. The
    ///   migration therefore yields no tokens; the operator has to
    ///   re-authenticate, and that write produces a normal envelope.
    /// * **Without a key** — only plaintext is accepted. An envelope
    ///   cannot be read (there is no key) and is reported as "no
    ///   tokens" with a warning.
    /// * Neither shape parses → the historical corrupt-file handling:
    ///   the original parse error propagates and callers treat it as
    ///   "no tokens".
    pub fn load(&self, key: Option<&SecretString>) -> Result<Option<StoredTokens>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&self.path)?;

        // (1) Encrypted store.
        if let Ok(envelope) = serde_json::from_str::<TokenEnvelope>(&raw) {
            if envelope.is_ours() {
                return self.load_encrypted(&envelope, key);
            }
        }

        // (2) Legacy plaintext store.
        if let Ok(tokens) = serde_json::from_str::<StoredTokens>(&raw) {
            if key.is_none() {
                return Ok(Some(tokens));
            }
            // A key is configured: the plaintext must not be used, but
            // it must also not stay on the disk in the clear.
            self.migrate_plaintext(&raw, key);
            tracing::warn!(
                path = %self.path.display(),
                "token store is plaintext but provider.token_store_key is configured; \
                 refusing to use it (it was never authenticated) and migrating it to an \
                 encrypted store -- re-authenticate via /admin/copilot/auth"
            );
            return Ok(None);
        }

        // (3) Neither — keep the historical corrupt-file behavior and
        // surface the original parse error.
        Err(ProxyError::Json(
            serde_json::from_str::<StoredTokens>(&raw).unwrap_err(),
        ))
    }

    /// Re-write a legacy plaintext store as a `migrated` envelope.
    ///
    /// Best-effort: a failure here must not take the proxy down (the
    /// operator can still re-authenticate and the next `save` replaces
    /// the file anyway), so the error is logged and swallowed.
    fn migrate_plaintext(&self, raw: &str, key: Option<&SecretString>) {
        let Some(key) = key else {
            return;
        };
        let outcome = (|| -> Result<()> {
            let key_bytes = parse_key(key.expose_secret()).map_err(ProxyError::Config)?;
            let envelope = seal_migrated(key_bytes.as_bytes(), raw.as_bytes())?;
            let serialized = serde_json::to_string_pretty(&envelope)?;
            write_atomic(&self.path, serialized.as_bytes())
        })();
        if let Err(e) = outcome {
            tracing::warn!(
                path = %self.path.display(),
                error = %e,
                "failed to migrate the plaintext token store to an encrypted one; \
                 the plaintext file is still on disk"
            );
        }
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
        if envelope.migrated {
            // This envelope wraps a legacy plaintext store, which was
            // never authenticated. Encrypting it got the plaintext off
            // the disk, but the bytes themselves are not trustworthy.
            tracing::warn!(
                path = %self.path.display(),
                "token store was migrated from an unauthenticated plaintext store; \
                 treating it as empty (re-authenticate via /admin/copilot/auth)"
            );
            return Ok(None);
        }
        let key_bytes = parse_key(key.expose_secret()).map_err(ProxyError::Config)?;
        match open(key_bytes.as_bytes(), envelope) {
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
    ///
    /// A plaintext write over an **encrypted** store is refused: that
    /// would silently undo the encryption the moment the operator's
    /// `token_store_key` goes missing (an unset `${VAR}` in the config
    /// expands to nothing). Failing loudly here is the whole point —
    /// the ciphertext on disk must never be downgraded by accident.
    pub fn save(&self, tokens: &StoredTokens, key: Option<&SecretString>) -> Result<()> {
        if key.is_none() && self.is_encrypted_on_disk() {
            return Err(ProxyError::Config(format!(
                "refusing to overwrite the encrypted token store at {} with plaintext; \
                 configure provider.token_store_key for this provider",
                self.path.display()
            )));
        }
        let raw = match key {
            None => serde_json::to_string_pretty(tokens)?,
            Some(key) => {
                let key_bytes = parse_key(key.expose_secret()).map_err(ProxyError::Config)?;
                let plaintext = serde_json::to_vec(tokens)?;
                let envelope = seal(key_bytes.as_bytes(), &plaintext)?;
                serde_json::to_string_pretty(&envelope)?
            }
        };
        write_atomic(&self.path, raw.as_bytes())?;
        Ok(())
    }

    /// True when the store on disk is one of our encrypted envelopes.
    fn is_encrypted_on_disk(&self) -> bool {
        let Ok(raw) = std::fs::read_to_string(&self.path) else {
            return false;
        };
        serde_json::from_str::<TokenEnvelope>(&raw)
            .map(|envelope| envelope.is_ours())
            .unwrap_or(false)
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
        fn plaintext_store_is_migrated_but_not_trusted_when_a_key_is_configured() {
            // A plaintext store carries no authentication: anybody who
            // could write that file could have planted their own token.
            // So a configured key means the plaintext is *not* used --
            // but it is re-written as an envelope so the host disk stops
            // holding the cleartext.
            let (_dir, store) = fresh_store();
            let tokens = sentinel_tokens();
            store.save(&tokens, None).unwrap();
            let plaintext_bytes = std::fs::read(store.path()).unwrap();

            let key = secret_string(KEY_A);
            let loaded = store.load(Some(&key)).unwrap();
            assert!(loaded.is_none(), "a plaintext store must not be trusted");

            // The file is now an envelope ...
            let raw = std::fs::read_to_string(store.path()).unwrap();
            assert!(
                raw.contains(crate::oauth::token_crypto::ENVELOPE_MAGIC),
                "{raw}"
            );
            assert!(
                !raw.contains("gh-plaintext-sentinel"),
                "the cleartext must be gone from the disk: {raw}"
            );
            assert_ne!(
                std::fs::read(store.path()).unwrap(),
                plaintext_bytes,
                "the store must have been rewritten"
            );

            // ... and the migrated envelope stays untrusted: reloading
            // does not resurrect the migrated tokens either.
            assert!(store.load(Some(&key)).unwrap().is_none());
        }

        #[test]
        fn a_plaintext_write_over_an_encrypted_store_is_refused() {
            // The operator's token_store_key went missing (an unset
            // ${VAR} expands to nothing). The next token refresh must
            // not quietly downgrade the encrypted store to plaintext.
            let (_dir, store) = fresh_store();
            let key = secret_string(KEY_A);
            store.save(&sentinel_tokens(), Some(&key)).unwrap();
            let before = std::fs::read(store.path()).unwrap();

            let err = store.save(&sentinel_tokens(), None).unwrap_err();

            assert!(
                err.to_string().contains("refusing to overwrite the encrypted token store"),
                "{err}"
            );
            assert_eq!(
                std::fs::read(store.path()).unwrap(),
                before,
                "the encrypted store must be byte-identical after the refusal"
            );
        }

        #[test]
        fn a_plaintext_write_is_allowed_when_there_is_no_encrypted_store() {
            // The legacy path: no key, no envelope on disk -> plaintext
            // is fine.
            let (_dir, store) = fresh_store();
            store.save(&sentinel_tokens(), None).unwrap();

            store.save(&sentinel_tokens(), None).unwrap();

            let raw = std::fs::read_to_string(store.path()).unwrap();
            assert!(!raw.contains(crate::oauth::token_crypto::ENVELOPE_MAGIC), "{raw}");
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
