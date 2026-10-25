//! Token persistence for GitHub OAuth + Copilot API tokens.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

use crate::error::{ProxyError, Result};

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

    pub fn load(&self) -> Result<Option<StoredTokens>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(&self.path)?;
        let parsed: StoredTokens = serde_json::from_str(&raw)?;
        Ok(Some(parsed))
    }

    pub fn save(&self, tokens: &StoredTokens) -> Result<()> {
        let raw = serde_json::to_string_pretty(tokens)?;
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
        store.save(&tokens).unwrap();
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
        assert!(store.load().unwrap().is_none());

        let tokens = StoredTokens {
            github_access_token: "gh-xxx".into(),
            copilot_token: "co-yyy".into(),
            copilot_expires_at: 1234567890,
            refresh_in: 1500,
        };
        store.save(&tokens).unwrap();
        let loaded = store.load().unwrap().expect("file should exist");
        assert_eq!(loaded.github_access_token, "gh-xxx");
        assert_eq!(loaded.copilot_token, "co-yyy");
        assert_eq!(loaded.copilot_expires_at, 1234567890);

        store.clear().unwrap();
        assert!(store.load().unwrap().is_none());
    }

    #[test]
    fn load_missing_file_returns_none() {
        let (_dir, store) = fresh_store();
        assert!(store.load().unwrap().is_none());
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
            dumped.contains(&format!("[REDACTED {} bytes]", tokens.github_access_token.len())),
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
        store.save(&tokens).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(store.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "expected 0600, got {:o}", mode);
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

        store.save(&tokens).unwrap();
        store.save(&tokens).unwrap();

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
