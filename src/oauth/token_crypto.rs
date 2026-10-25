//! Envelope encryption for the on-disk Copilot token store.
//!
//! # Why this exists
//!
//! `github_token.json` holds a **live** GitHub OAuth access token and a
//! **live** Copilot API token. It lives in the `llmproxy-data` named
//! volume, which sits on an **unencrypted** host disk — precisely the
//! thing the TPM-sealed `.env` story (`docs/tpm-sealed-secrets-deploy.md`)
//! exists to protect. A sealed `.env` next to a plaintext token store
//! is only half a story.
//!
//! The fix: encrypt the store with a key that is delivered through the
//! already-sealed `.env`, so the host disk only ever holds ciphertext.
//!
//! # Wire format
//!
//! ```json
//! {"magic":"llmproxy-token-v1","alg":"A256GCM","nonce":"<b64>","ct":"<b64>"}
//! ```
//!
//! * `ct` is the AES-256-GCM ciphertext with the 16-byte GCM tag
//!   **appended**. `ring`'s `seal_in_place_separate_tag` produces the
//!   tag separately; we concatenate so the envelope carries one blob.
//! * `nonce` is a fresh 12-byte random value per save
//!   (`ring::rand::SystemRandom`). GCM's security requirement is nonce
//!   uniqueness under a single key, which a 96-bit random nonce
//!   satisfies with overwhelming probability.
//! * **AAD is the magic string.** Binding the format version into the
//!   authenticated data means an attacker cannot relabel a ciphertext
//!   as a different algorithm/version without the tag check failing.
//!
//! A store written by an older build (plaintext JSON) is still
//! readable — see [`crate::oauth::token_store::TokenStore::load`],
//! which treats "parses as [`StoredTokens`]" as the migration path.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};

use crate::error::{ProxyError, Result};

/// Format identifier written into every envelope. Doubles as the AAD.
pub const ENVELOPE_MAGIC: &str = "llmproxy-token-v1";

/// Algorithm tag written into every envelope.
pub const ENVELOPE_ALG: &str = "A256GCM";

/// AES-256 key length in bytes (32).
pub const KEY_LEN: usize = 32;

/// GCM nonce length in bytes (12).
pub const NONCE_LEN: usize = 12;

/// GCM authentication tag length in bytes (16).
pub const TAG_LEN: usize = 16;

/// The on-disk envelope. Every field is base64 except the two tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEnvelope {
    /// Format identifier — must equal [`ENVELOPE_MAGIC`].
    pub magic: String,
    /// Algorithm identifier — must equal [`ENVELOPE_ALG`].
    pub alg: String,
    /// Base64 of the 12-byte GCM nonce.
    pub nonce: String,
    /// Base64 of `ciphertext || tag`.
    pub ct: String,
}

impl TokenEnvelope {
    /// True when this envelope carries our magic string. Used to tell
    /// "encrypted store" apart from "legacy plaintext store" and from
    /// "corrupt file" without guessing.
    pub fn is_ours(&self) -> bool {
        self.magic == ENVELOPE_MAGIC
    }
}

/// Human-readable description of the expected key format. Reused by the
/// config validator so the operator sees the same wording in both
/// places.
pub const KEY_FORMAT_HINT: &str =
    "expected 64 hex characters (32 bytes), e.g. `openssl rand -hex 32`";

/// Parse a 64-hex-character token-store key into 32 raw bytes.
///
/// The key is used **directly** as the AES-256 key — no KDF, because
/// the operator supplies a random 32-byte value, not a password.
///
/// The error is a plain message: the config validator prefixes it with
/// the provider name, the runtime path wraps it in a
/// [`ProxyError::Config`].
pub fn parse_key(hex_key: &str) -> std::result::Result<[u8; KEY_LEN], String> {
    let trimmed = hex_key.trim();
    if trimmed.len() != KEY_LEN * 2 {
        return Err(format!(
            "token_store_key must be {} characters long ({KEY_FORMAT_HINT}), got {}",
            KEY_LEN * 2,
            trimmed.len()
        ));
    }
    let mut out = [0u8; KEY_LEN];
    let bytes = trimmed.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(out)
}

/// Decode one hex digit.
fn hex_nibble(b: u8) -> std::result::Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(format!(
            "token_store_key must be hexadecimal ({KEY_FORMAT_HINT}); found {:?}",
            other as char
        )),
    }
}

/// Build the AEAD key from raw bytes.
fn aead_key(key: &[u8; KEY_LEN]) -> Result<LessSafeKey> {
    let unbound = UnboundKey::new(&AES_256_GCM, key)
        .map_err(|_| ProxyError::Other(anyhow::anyhow!("invalid AES-256-GCM key material")))?;
    Ok(LessSafeKey::new(unbound))
}

/// Seal `plaintext` into an envelope under `key`.
///
/// A fresh random nonce is drawn for every call, so sealing the same
/// plaintext twice yields two different envelopes.
pub fn seal(key: &[u8; KEY_LEN], plaintext: &[u8]) -> Result<TokenEnvelope> {
    let key = aead_key(key)?;

    let mut nonce_bytes = [0u8; NONCE_LEN];
    SystemRandom::new().fill(&mut nonce_bytes).map_err(|_| {
        ProxyError::Other(anyhow::anyhow!("failed to generate a token-store nonce"))
    })?;
    let nonce = Nonce::assume_unique_for_key(nonce_bytes);

    let mut buf = plaintext.to_vec();
    let tag = key
        .seal_in_place_separate_tag(nonce, Aad::from(ENVELOPE_MAGIC.as_bytes()), &mut buf)
        .map_err(|_| ProxyError::Other(anyhow::anyhow!("failed to seal the token store")))?;
    debug_assert_eq!(tag.as_ref().len(), TAG_LEN);
    buf.extend_from_slice(tag.as_ref());

    Ok(TokenEnvelope {
        magic: ENVELOPE_MAGIC.to_string(),
        alg: ENVELOPE_ALG.to_string(),
        nonce: B64.encode(nonce_bytes),
        ct: B64.encode(&buf),
    })
}

/// Open an envelope under `key` and return the plaintext bytes.
///
/// Fails when the magic/algorithm tags do not match, when the nonce is
/// not 12 bytes, or when the GCM tag check fails (wrong key or
/// tampered ciphertext).
pub fn open(key: &[u8; KEY_LEN], envelope: &TokenEnvelope) -> Result<Vec<u8>> {
    if !envelope.is_ours() {
        return Err(ProxyError::Config(format!(
            "unknown token store envelope format {:?}",
            envelope.magic
        )));
    }
    if envelope.alg != ENVELOPE_ALG {
        return Err(ProxyError::Config(format!(
            "unsupported token store algorithm {:?}",
            envelope.alg
        )));
    }

    let nonce_bytes = B64
        .decode(envelope.nonce.as_bytes())
        .map_err(|_| ProxyError::Other(anyhow::anyhow!("token store nonce is not valid base64")))?;
    if nonce_bytes.len() != NONCE_LEN {
        return Err(ProxyError::Other(anyhow::anyhow!(
            "token store nonce must be {NONCE_LEN} bytes, got {}",
            nonce_bytes.len()
        )));
    }

    let mut buf = B64.decode(envelope.ct.as_bytes()).map_err(|_| {
        ProxyError::Other(anyhow::anyhow!(
            "token store ciphertext is not valid base64"
        ))
    })?;
    if buf.len() < TAG_LEN {
        return Err(ProxyError::Other(anyhow::anyhow!(
            "token store ciphertext is too short to contain a GCM tag"
        )));
    }

    let key = aead_key(key)?;
    let mut nonce_arr = [0u8; NONCE_LEN];
    nonce_arr.copy_from_slice(&nonce_bytes);
    let plaintext = key
        .open_in_place(
            Nonce::assume_unique_for_key(nonce_arr),
            Aad::from(ENVELOPE_MAGIC.as_bytes()),
            &mut buf,
        )
        .map_err(|_| {
            ProxyError::Other(anyhow::anyhow!(
                "failed to open the token store: wrong key or tampered ciphertext"
            ))
        })?;

    Ok(plaintext.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_a() -> [u8; KEY_LEN] {
        [0x11u8; KEY_LEN]
    }

    fn key_b() -> [u8; KEY_LEN] {
        [0x22u8; KEY_LEN]
    }

    #[test]
    fn parse_key_accepts_64_hex_chars() {
        let hex = "ab".repeat(32);
        let key = parse_key(&hex).unwrap();
        assert_eq!(key[0], 0xab);
        assert_eq!(key[31], 0xab);
    }

    #[test]
    fn parse_key_accepts_uppercase_hex() {
        let key = parse_key(&"AB".repeat(32)).unwrap();
        assert_eq!(key[0], 0xab);
    }

    #[test]
    fn parse_key_rejects_wrong_length() {
        let err = parse_key(&"ab".repeat(31)).unwrap_err();
        assert!(err.contains("64 characters"), "{err}");
    }

    #[test]
    fn parse_key_rejects_non_hex() {
        let err = parse_key(&"zz".repeat(32)).unwrap_err();
        assert!(err.contains("hexadecimal"), "{err}");
    }

    #[test]
    fn seal_open_round_trip() {
        let env = seal(&key_a(), b"secret-bytes").unwrap();
        assert_eq!(env.magic, ENVELOPE_MAGIC);
        assert_eq!(env.alg, ENVELOPE_ALG);
        let opened = open(&key_a(), &env).unwrap();
        assert_eq!(opened, b"secret-bytes");
    }

    #[test]
    fn seal_uses_a_fresh_nonce_each_time() {
        let a = seal(&key_a(), b"same").unwrap();
        let b = seal(&key_a(), b"same").unwrap();
        assert_ne!(a.nonce, b.nonce);
        assert_ne!(a.ct, b.ct);
    }

    #[test]
    fn open_rejects_the_wrong_key() {
        let env = seal(&key_a(), b"secret-bytes").unwrap();
        let err = open(&key_b(), &env).unwrap_err();
        assert!(err.to_string().contains("wrong key"), "{err}");
    }

    #[test]
    fn open_rejects_tampered_ciphertext() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        let mut ct = B64.decode(env.ct.as_bytes()).unwrap();
        ct[0] ^= 0xff;
        env.ct = B64.encode(&ct);
        let err = open(&key_a(), &env).unwrap_err();
        assert!(err.to_string().contains("wrong key"), "{err}");
    }

    #[test]
    fn open_rejects_a_foreign_magic() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.magic = "something-else".to_string();
        let err = open(&key_a(), &env).unwrap_err();
        assert!(
            err.to_string().contains("unknown token store envelope"),
            "{err}"
        );
    }

    #[test]
    fn open_rejects_an_unknown_algorithm() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.alg = "A128GCM".to_string();
        let err = open(&key_a(), &env).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported token store algorithm"),
            "{err}"
        );
    }

    #[test]
    fn open_rejects_a_short_nonce() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.nonce = B64.encode([0u8; 8]);
        let err = open(&key_a(), &env).unwrap_err();
        assert!(err.to_string().contains("nonce must be 12 bytes"), "{err}");
    }

    #[test]
    fn open_rejects_a_truncated_ciphertext() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.ct = B64.encode([0u8; 4]);
        let err = open(&key_a(), &env).unwrap_err();
        assert!(err.to_string().contains("too short"), "{err}");
    }

    #[test]
    fn open_rejects_non_base64_fields() {
        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.nonce = "!!!not base64!!!".to_string();
        let err = open(&key_a(), &env).unwrap_err();
        assert!(err.to_string().contains("not valid base64"), "{err}");

        let mut env = seal(&key_a(), b"secret-bytes").unwrap();
        env.ct = "!!!not base64!!!".to_string();
        let err = open(&key_a(), &env).unwrap_err();
        assert!(err.to_string().contains("not valid base64"), "{err}");
    }
}
