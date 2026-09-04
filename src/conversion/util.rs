//! Shared helpers used by multiple conversion modules.
//!
//! Anything that isn't tied to a specific upstream wire format (Chat
//! Completions vs Responses API) but needs to be reused across both
//! conversion paths lives here. This keeps the individual conversion
//! modules focused on their own protocol translation and avoids
//! cross-module imports between them (e.g. `responses.rs` reaching into
//! `request.rs`).

use serde_json::{json, Value};
use std::collections::HashMap;

use crate::anthropic::{Tool, Usage};

/// Detect whether a `Tool` is an Anthropic-native hosted web search tool.
///
/// Matches:
/// - `{type: "web_search_20250305", name: "web_search"}` (canonical Anthropic)
/// - `{type: "web_search_preview", ...}` (OpenAI Responses API shape)
///
/// Does NOT match:
/// - `{name: "web_search", type: "function"}` (user-defined function tool
///   coincidentally named "web_search")
pub fn is_web_search_tool(t: &Tool) -> bool {
    if let Some(type_str) = &t.kind {
        // `type_str == "web_search"` is the Azure OpenAI Responses-API
        // stable alias for `web_search_preview` (Microsoft Foundry
        // docs). Anthropic's hosted-tool type is always the versioned
        // `web_search_20250305`, so this branch is dead for Anthropic
        // inbound traffic — but harmless and defensive against future
        // OpenAI naming changes.
        return type_str.starts_with("web_search_")
            || type_str == "web_search"
            || type_str == "web_search_preview";
    }
    false
}

/// Recursively make a JSON Schema compliant with OpenAI strict mode.
///
/// # What this does
///
/// For every object-typed schema that has a `properties` key:
///
/// - Sets `additionalProperties: false` (unconditionally — overwrites any
///   pre-existing `true` value).
/// - Replaces `required` with **every** property key from `properties`
///   (unconditionally — discards any user-supplied subset).
/// - Recurses into each property value.
///
/// Then, regardless of the schema's own type, it also recurses into
/// structural composition keywords: `items`, `anyOf`, `oneOf`, `allOf`,
/// `$defs`, `definitions`, and the advanced keywords listed below.
///
/// # Destructive behavior (prominent notice)
///
/// This function is **semantically aggressive**. It silently discards
/// user intent in two ways:
///
/// 1. **`additionalProperties: true` is overwritten with `false`.** If
///    the user's schema intentionally allowed extra fields, that
///    allowance is removed.
/// 2. **The `required` array is replaced in full with the complete set
///    of property keys.** Any property the user deliberately left
///    optional becomes required.
///
/// These transformations are necessary for OpenAI strict mode (which
/// demands both `additionalProperties: false` and a `required` listing
/// every property), and they match litellm's
/// `_add_additional_properties_false`
/// (`litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py:823-855`).
/// If preserving the user's `required` subset is important, callers
/// must pre-process the schema before calling this function.
///
/// # Recursion into advanced keywords
///
/// The function recurses into the following structural keywords so that
/// nested object schemas at any depth get strictified:
///
/// - Composition: `anyOf`, `oneOf`, `allOf` (each branch)
/// - Arrays: `items`, `prefixItems` (each tuple item), `contains`,
///   `unevaluatedItems`
/// - Definitions: `$defs`, `definitions`, `dependentSchemas`
/// - Conditional: `not`, `if`, `then`, `else`
/// - Object validation: `propertyNames`
///
/// # `$ref` handling (PR-13)
///
/// - Same-schema fragment refs (`#/$defs/Foo`, `#/definitions/Foo`) are
///   **kept verbatim** — OpenAI strict mode supports non-recursive
///   same-schema refs — and their target definitions are still
///   strictified via the `$defs`/`definitions` pass. A definition chain
///   that loops back on itself (`$defs.A` → `$defs.B` → `$defs.A`) is
///   rejected as circular.
/// - External URI refs (`https://…`, `file://…`, bare paths) are
///   **rejected** with [`SchemaError`]: the upstream can't resolve them
///   under strict mode, so forwarding them would 400. Callers with such
///   schemas must inline the referenced schema first.
pub fn strictify_schema(schema: &mut Value) -> Result<(), SchemaError> {
    // PR-13: snapshot the top-level definition tables once so same-schema
    // `$ref` targets can be resolved for cycle detection without a second
    // mutable borrow of `schema`.
    let defs = collect_defs(schema);
    strictify_inner(schema, "$", &defs)
}

/// Error surfaced when a JSON Schema cannot be strictified.
///
/// Returned by [`strictify_schema`] when a `$ref` can't be honored under
/// OpenAI strict mode (see the `$ref` handling section above).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError {
    /// Human-readable message; follows the
    /// `strictify_schema: <kind> $ref at <path> not supported` template.
    pub message: String,
}

impl SchemaError {
    fn external_ref(path: &str) -> Self {
        Self {
            message: format!("strictify_schema: external $ref at {path} not supported"),
        }
    }
    fn circular_ref(path: &str) -> Self {
        Self {
            message: format!("strictify_schema: circular $ref at {path} not supported"),
        }
    }
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SchemaError {}

/// Snapshot of the top-level `$defs`/`definitions` tables (name → value),
/// used to resolve same-schema `$ref` targets during cycle detection.
/// `$defs` wins over `definitions` on a name collision.
fn collect_defs(schema: &Value) -> HashMap<String, Value> {
    let mut defs = HashMap::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(map)) = schema.get(key) {
            for (name, v) in map {
                defs.insert(name.clone(), v.clone());
            }
        }
    }
    defs
}

/// Resolve `#/$defs/Name` / `#/definitions/Name` fragment refs to a bare
/// definition name. Returns `None` for other fragments (e.g.
/// `#/properties/foo`) — those are kept verbatim with no cycle check.
fn resolve_def_name(fragment: &str) -> Option<String> {
    for prefix in ["/$defs/", "/definitions/"] {
        if let Some(name) = fragment.strip_prefix(prefix) {
            if !name.is_empty() && !name.contains('/') {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// Follow a definition's `$ref` chain inside the in-schema `$defs` table
/// and fail if it loops back on itself. This is the same guard litellm's
/// `unpack_defs` applies via its `ref_chain` bookkeeping
/// (`common_utils.py:877+`): a cycle is unrepresentable and must be
/// surfaced rather than forwarded. Non-definition targets and definitions
/// that don't start with a `$ref` are benign (the ref is kept verbatim).
fn check_def_cycle(
    name: &str,
    defs: &HashMap<String, Value>,
    path: &str,
) -> Result<(), SchemaError> {
    let mut visited: Vec<String> = vec![name.to_string()];
    let mut current = name.to_string();
    let mut steps = 0;
    while steps <= defs.len() {
        let Some(value) = defs.get(&current) else {
            return Ok(());
        };
        let Some(value_obj) = value.as_object() else {
            return Ok(());
        };
        let Some(next) = value_obj.get("$ref").and_then(|v| v.as_str()) else {
            return Ok(());
        };
        let Some(next_name) = resolve_def_name(next.strip_prefix('#').unwrap_or(next)) else {
            return Ok(());
        };
        if visited.iter().any(|v| v == &next_name) {
            return Err(SchemaError::circular_ref(path));
        }
        visited.push(next_name.clone());
        current = next_name;
        steps += 1;
    }
    // A chain longer than the definition table must have repeated a node.
    Err(SchemaError::circular_ref(path))
}

fn strictify_inner(
    schema: &mut Value,
    path: &str,
    defs: &HashMap<String, Value>,
) -> Result<(), SchemaError> {
    let Value::Object(obj) = schema else {
        return Ok(());
    };

    // PR-13: a `$ref` key makes this node a reference, not an inline
    // schema. Same-schema fragment refs are kept verbatim (and checked
    // for cycles); external URIs are an error.
    if let Some(ref_val) = obj.get("$ref") {
        if let Some(ref_str) = ref_val.as_str() {
            if let Some(fragment) = ref_str.strip_prefix('#') {
                if let Some(name) = resolve_def_name(fragment) {
                    check_def_cycle(&name, defs, path)?;
                }
            } else {
                return Err(SchemaError::external_ref(path));
            }
        }
        // Non-string `$ref` (malformed) — treat as inert, keep verbatim.
        return Ok(());
    }

    let is_object = obj.get("type").and_then(|v| v.as_str()) == Some("object")
        && obj.contains_key("properties");
    if is_object {
        obj.insert("additionalProperties".to_string(), Value::Bool(false));
        let keys: Option<Vec<Value>> = obj
            .get("properties")
            .and_then(|p| p.as_object())
            .map(|p| p.keys().map(|k| Value::String(k.clone())).collect());
        if let Some(keys) = keys {
            obj.insert("required".to_string(), Value::Array(keys));
        }
        if let Some(Value::Object(p)) = obj.get_mut("properties") {
            for (name, v) in p.iter_mut() {
                strictify_inner(v, &format!("{path}.properties.{name}"), defs)?;
            }
        }
    }
    if let Some(items) = obj.get_mut("items") {
        strictify_inner(items, &format!("{path}.items"), defs)?;
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(arr)) = obj.get_mut(key) {
            for (i, sub) in arr.iter_mut().enumerate() {
                strictify_inner(sub, &format!("{path}.{key}[{i}]"), defs)?;
            }
        }
    }
    for key in ["$defs", "definitions", "dependentSchemas"] {
        if let Some(Value::Object(map)) = obj.get_mut(key) {
            for (name, v) in map.iter_mut() {
                strictify_inner(v, &format!("{path}.{key}.{name}"), defs)?;
            }
        }
    }
    for key in ["not", "if", "then", "else", "contains", "propertyNames", "unevaluatedItems"] {
        if let Some(sub) = obj.get_mut(key) {
            strictify_inner(sub, &format!("{path}.{key}"), defs)?;
        }
    }
    Ok(())
}

/// PR-11: shared Anthropic `Usage` constructor for all four conversion
/// paths (Chat NS / Chat S / Responses NS / Responses S). Plan v0.11
/// R6 + v0.10 键名:
/// - `cached_tokens` (Anthropic read key) → `cache_read_input_tokens`
///   (None when 0 so wire stays absent — Some(0) would change shape).
/// - `reasoning_tokens` (OpenAI read key) → `output_tokens_details
///   .thinking_tokens` (Anthropic write key, different from the
///   OpenAI read key).
/// - `reasoning_tokens` is clamped to `[0, output_tokens]` —
///   Anthropic spec guarantees `thinking_tokens ≤ output_tokens`,
///   so we never emit a wire shape that violates the invariant.
///   This may suppress the R6 "reasoning_tokens >= completion_tokens"
///   warning (plan v0.11 R6 注记: semantics unchanged, only the warn
///   condition shifts; relevant tests adjusted to clamp consistently).
pub fn build_usage(
    input_tokens: u32,
    output_tokens: u32,
    cached_tokens: u32,
    reasoning_tokens: u32,
    service_tier: &Option<String>,
) -> Usage {
    let cached = cached_tokens;
    let reasoning = reasoning_tokens.min(output_tokens);
    Usage {
        input_tokens: input_tokens.saturating_sub(cached),
        output_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: if cached > 0 { Some(cached) } else { None },
        cache_creation: None,
        server_tool_use: None,
        output_tokens_details: if reasoning > 0 {
            Some(json!({"thinking_tokens": reasoning}))
        } else {
            None
        },
        service_tier: service_tier.clone(),
        inference_geo: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strictify_schema_reaches_into_nested_object_properties() {
        // Recursion check: strictify must descend into nested object
        // properties, into `items` of arrays, and rewrite each nested object
        // (adding additionalProperties: false and promoting its properties to
        // `required`).
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"}
                    }
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"}
                        }
                    }
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");

        // Top-level: all keys promoted to required, additionalProperties: false.
        let top_required = schema.get("required").and_then(|v| v.as_array()).unwrap();
        let top_required: Vec<&str> = top_required.iter().filter_map(|v| v.as_str()).collect();
        assert!(top_required.contains(&"user"), "top-level required must include `user`");
        assert!(top_required.contains(&"tags"), "top-level required must include `tags`");
        assert_eq!(
            schema.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );

        // Nested object property `user` must be rewritten in place.
        let props = schema.get("properties").unwrap();
        let user = props.get("user").unwrap();
        assert_eq!(user.get("type").and_then(|v| v.as_str()), Some("object"));
        assert_eq!(
            user.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let user_required = user.get("required").and_then(|v| v.as_array()).unwrap();
        let user_required: Vec<&str> = user_required.iter().filter_map(|v| v.as_str()).collect();
        assert!(user_required.contains(&"id"));
        assert!(user_required.contains(&"name"));

        // Array `items` (a nested object) must also be rewritten.
        let tags = props.get("tags").unwrap();
        assert_eq!(tags.get("type").and_then(|v| v.as_str()), Some("array"));
        let items = tags.get("items").unwrap();
        assert_eq!(items.get("type").and_then(|v| v.as_str()), Some("object"));
        assert_eq!(
            items.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let items_required = items.get("required").and_then(|v| v.as_array()).unwrap();
        let items_required: Vec<&str> = items_required.iter().filter_map(|v| v.as_str()).collect();
        assert!(items_required.contains(&"label"));
    }

    #[test]
    fn strictify_schema_recurses_into_any_of() {
        // anyOf branches containing objects must each be strictified in
        // place (additionalProperties: false + complete required).
        let mut schema = json!({
            "type": "object",
            "properties": {
                "result": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "value": {"type": "number"}
                            }
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "error": {"type": "string"}
                            }
                        }
                    ]
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let result = schema.get("properties").and_then(|p| p.get("result")).unwrap();
        let branches = result.get("anyOf").and_then(|v| v.as_array()).unwrap();
        assert_eq!(branches.len(), 2);
        for (i, branch) in branches.iter().enumerate() {
            assert_eq!(
                branch.get("additionalProperties").and_then(|v| v.as_bool()),
                Some(false),
                "anyOf branch {i} must have additionalProperties: false"
            );
            let required = branch
                .get("required")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("anyOf branch {i} must have a required array"));
            let required: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
            assert!(
                !required.is_empty(),
                "anyOf branch {i} must populate required from its properties"
            );
        }
    }

    #[test]
    fn strictify_schema_recurses_into_one_of() {
        // oneOf behaves identically to anyOf for strictification.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "id": {"type": "integer"}
                            }
                        }
                    ]
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let value = schema.get("properties").and_then(|p| p.get("value")).unwrap();
        let branch = &value.get("oneOf").and_then(|v| v.as_array()).unwrap()[0];
        assert_eq!(
            branch.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let required: Vec<&str> = branch
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"name"));
        assert!(required.contains(&"id"));
    }

    #[test]
    fn strictify_schema_recurses_into_defs() {
        // Reusable definitions inside `$defs` must be strictified the same
        // way as inline objects — OpenAI strict mode applies to every
        // nested object the model might emit.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"}
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let defs = schema.get("$defs").and_then(|v| v.as_object()).unwrap();
        let addr = defs.get("Address").unwrap();
        assert_eq!(
            addr.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let required: Vec<&str> = addr
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"street"));
        assert!(required.contains(&"city"));
    }

    /// PR-13: a same-schema `#/$defs/Foo` ref is kept verbatim (OpenAI
    /// strict mode supports non-recursive same-schema refs) while the
    /// referenced definition is strictified.
    #[test]
    fn strictify_schema_keeps_same_schema_ref_verbatim() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    }
                }
            }
        });
        strictify_schema(&mut schema).expect("same-schema ref must be accepted");
        // The `$ref` itself is untouched on the wire.
        let user = schema
            .get("properties")
            .and_then(|p| p.get("user"))
            .expect("user property must remain");
        assert_eq!(
            user.get("$ref").and_then(|v| v.as_str()),
            Some("#/$defs/User"),
            "same-schema ref must be kept verbatim, not flattened"
        );
        // The referenced definition is still strictified.
        let def_user = schema
            .get("$defs")
            .and_then(|d| d.get("User"))
            .expect("User definition must remain");
        assert_eq!(
            def_user.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    /// PR-13: an external URI `$ref` (`https://…`, bare path) cannot be
    /// honored under OpenAI strict mode — the upstream can't resolve it —
    /// so strictification fails with the documented message template.
    #[test]
    fn strictify_schema_errors_on_external_uri_ref() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "shared": {"$ref": "https://schemas.example/common.json"}
            }
        });
        let err = strictify_schema(&mut schema)
            .expect_err("external $ref must be rejected, not forwarded");
        assert_eq!(
            err.message,
            "strictify_schema: external $ref at $.properties.shared not supported"
        );
    }

    /// PR-13: a `$defs` chain that loops back on itself (`A` → `B` → `A`)
    /// is unrepresentable and must be rejected rather than forwarded to
    /// the upstream.
    #[test]
    fn strictify_schema_errors_on_circular_ref() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "loop": {"$ref": "#/$defs/A"}
            },
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/A"}
            }
        });
        let err = strictify_schema(&mut schema)
            .expect_err("circular $ref chain must be rejected");
        assert!(
            err.message.starts_with("strictify_schema: circular $ref at "),
            "unexpected message: {}",
            err.message
        );
        assert!(err.message.ends_with(" not supported"));
    }

    /// PR-13: benign `$ref` cases must pass. A ref to a missing
    /// definition, a ref to a non-`$defs` fragment (`#/properties/foo`),
    /// and a definition whose target isn't an object are all kept
    /// verbatim without a cycle error.
    #[test]
    fn strictify_schema_accepts_benign_refs() {
        let mut missing = json!({
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/DoesNotExist"}}
        });
        strictify_schema(&mut missing).expect("missing def target must not error");

        let mut non_defs_fragment = json!({
            "type": "object",
            "properties": {
                "y": {"$ref": "#/properties/y"},
                "nested": {"$ref": "#/properties/y"}
            }
        });
        strictify_schema(&mut non_defs_fragment)
            .expect("non-defs fragment ref must be kept verbatim");

        let mut non_object_target = json!({
            "type": "object",
            "properties": {"z": {"$ref": "#/$defs/Z"}},
            "$defs": {"Z": "just a string"}
        });
        strictify_schema(&mut non_object_target)
            .expect("non-object def target must be accepted");

        let mut malformed_ref = json!({
            "type": "object",
            "properties": {"w": {"$ref": 42}}
        });
        strictify_schema(&mut malformed_ref)
            .expect("malformed (non-string) $ref must be treated as inert");
    }

    #[test]
    fn strictify_schema_non_object_returns_early() {
        // A schema whose top-level type is not "object" (e.g. "string")
        // must not be rewritten — strict mode rules only apply to object
        // schemas. The function takes &mut Value and returns nothing, so
        // we assert the schema is unchanged.
        let mut schema = json!({"type": "string"});
        let original = schema.clone();
        strictify_schema(&mut schema).expect("strictify must succeed");
        assert_eq!(schema, original);
        assert!(schema.get("additionalProperties").is_none());
        assert!(schema.get("required").is_none());
    }

    #[test]
    fn strictify_schema_object_without_properties_unchanged() {
        // A schema with `type: "object"` but no `properties` key must not
        // crash (no required array can be derived) and must not gain an
        // additionalProperties field.
        let mut schema = json!({"type": "object"});
        let original = schema.clone();
        strictify_schema(&mut schema).expect("strictify must succeed");
        assert_eq!(schema, original);
        assert!(schema.get("required").is_none());
    }

    #[test]
    fn strictify_schema_overwrites_additional_properties() {
        // Destructive behavior check: if the user set additionalProperties
        // to true, we overwrite it. This is documented in the function's
        // doc comment.
        let mut schema = json!({
            "type": "object",
            "additionalProperties": true,
            "properties": {
                "x": {"type": "string"}
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        assert_eq!(
            schema.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false),
            "additionalProperties: true must be overwritten to false"
        );
    }

    #[test]
    fn strictify_schema_replaces_required_with_all_keys() {
        // Destructive behavior check: if the user specified a subset of
        // keys as required, we replace it with the full set. This is
        // documented in the function's doc comment.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "reason": {"type": "string"},
                "optional_field": {"type": "string"}
            },
            "required": ["ok", "reason"]
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let required: Vec<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            required.contains(&"optional_field"),
            "optional property must be promoted to required under strict mode"
        );
        assert!(required.contains(&"ok"));
        assert!(required.contains(&"reason"));
    }

    #[test]
    fn strictify_schema_recurses_into_all_of() {
        // allOf is the third composition keyword alongside anyOf and
        // oneOf; all three must be traversed.
        let mut schema = json!({
            "type": "object",
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"}
                    }
                }
            ]
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let branch = &schema.get("allOf").and_then(|v| v.as_array()).unwrap()[0];
        assert_eq!(
            branch.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let required: Vec<&str> = branch
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"a"));
    }

    #[test]
    fn strictify_schema_recurses_into_definitions() {
        // `definitions` (pre-2019-09 JSON Schema spelling of `$defs`) must
        // be traversed identically.
        let mut schema = json!({
            "type": "object",
            "definitions": {
                "Point": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");
        let point = schema
            .get("definitions")
            .and_then(|v| v.as_object())
            .unwrap()
            .get("Point")
            .unwrap();
        assert_eq!(
            point.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
        let required: Vec<&str> = point
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"x"));
        assert!(required.contains(&"y"));
    }

    #[test]
    fn strictify_schema_recurses_into_if_then_else() {
        // Conditional validation keywords (if/then/else) may contain
        // nested object schemas that must be strictified in place.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "kind": {"type": "string"}
            },
            "if": {
                "type": "object",
                "properties": {
                    "kind": {"const": "error"}
                }
            },
            "then": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "code": {"type": "integer"}
                }
            },
            "else": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"}
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");

        // Top-level is strictified.
        assert_eq!(
            schema.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );

        // `if` branch is strictified.
        let if_schema = schema.get("if").unwrap();
        assert_eq!(
            if_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        // `then` branch is strictified.
        let then_schema = schema.get("then").unwrap();
        assert_eq!(
            then_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
        let then_required: Vec<&str> = then_schema
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(then_required.contains(&"message"));
        assert!(then_required.contains(&"code"));

        // `else` branch is strictified.
        let else_schema = schema.get("else").unwrap();
        assert_eq!(
            else_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
                Some(false)
        );
    }

    #[test]
    fn strictify_schema_recurses_into_not_and_contains() {
        // `not` and `contains` may contain nested object schemas that
        // must be strictified in place.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "contains": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"}
                        }
                    }
                }
            },
            "not": {
                "type": "object",
                "properties": {
                    "forbidden": {"type": "boolean"}
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");

        let contains_schema = schema
            .get("properties")
            .and_then(|p| p.get("items"))
            .and_then(|i| i.get("contains"))
            .unwrap();
        assert_eq!(
            contains_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );

        let not_schema = schema.get("not").unwrap();
        assert_eq!(
            not_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strictify_schema_recurses_into_prefix_items() {
        // `prefixItems` (tuple validation) contains an array of schemas
        // that may each be object schemas.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "tuple": {
                    "type": "array",
                    "prefixItems": [
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"}
                            }
                        },
                        {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer"}
                            }
                        }
                    ]
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");

        let prefix_items = schema
            .get("properties")
            .and_then(|p| p.get("tuple"))
            .and_then(|t| t.get("prefixItems"))
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(prefix_items.len(), 2);
        for (i, item) in prefix_items.iter().enumerate() {
            assert_eq!(
                item.get("additionalProperties").and_then(|v| v.as_bool()),
                Some(false),
                "prefixItems[{i}] must have additionalProperties: false"
            );
        }
    }

    #[test]
    fn strictify_schema_recurses_into_dependent_schemas() {
        // `dependentSchemas` maps property names to schemas that apply
        // when that property is present.
        let mut schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "dependentSchemas": {
                "name": {
                    "type": "object",
                    "properties": {
                        "nickname": {"type": "string"}
                    }
                }
            }
        });
        strictify_schema(&mut schema).expect("strictify must succeed");

        let dep_schema = schema
            .get("dependentSchemas")
            .and_then(|d| d.get("name"))
            .unwrap();
        assert_eq!(
            dep_schema
                .get("additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
        let required: Vec<&str> = dep_schema
            .get("required")
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(required.contains(&"nickname"));
    }

    #[test]
    fn strictify_schema_is_idempotent() {
        // Calling strictify_schema twice on the same schema must produce
        // the same result as calling it once. The required array is
        // rebuilt from properties.keys() on each call, so no duplication
        // occurs.
        let mut schema1 = json!({
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "reason": {"type": "string"}
            },
            "required": ["ok"]
        });
        let mut schema2 = schema1.clone();

        strictify_schema(&mut schema1).expect("strictify must succeed");
        strictify_schema(&mut schema2).expect("strictify must succeed");
        strictify_schema(&mut schema2).expect("strictify must succeed");

        assert_eq!(schema1, schema2, "double-call must be idempotent");
    }

    #[test]
    fn strictify_schema_handles_malformed_null_value() {
        // A null schema value must not panic. Strictify returns
        // immediately when the value isn't an object.
        let mut schema = Value::Null;
        strictify_schema(&mut schema).expect("strictify must succeed");
        assert_eq!(schema, Value::Null);
    }

    #[test]
    fn strictify_schema_handles_non_object_at_top_level() {
        // Schemas that are not objects (numbers, booleans, arrays) must
        // pass through unchanged. These are not object schemas so
        // strict mode rules don't apply.
        for value in [
            json!(42),
            json!(true),
            json!("string"),
            json!([1, 2, 3]),
        ] {
            let mut schema = value.clone();
            let original = schema.clone();
            strictify_schema(&mut schema).expect("strictify must succeed");
            assert_eq!(
                schema, original,
                "non-object schema {value} must pass through unchanged"
            );
        }
    }

    #[test]
    fn strictify_schema_handles_schema_without_type_field() {
        // Schemas without a `type` field (or with `properties` but no
        // `type: "object"`) must not be strictified. The presence of
        // `properties` alone is not enough to trigger the rewrite.
        let mut schema = json!({
            "properties": {
                "x": {"type": "string"}
            }
        });
        let original = schema.clone();
        strictify_schema(&mut schema).expect("strictify must succeed");
        assert_eq!(
            schema, original,
            "schema with properties but no `type: object` must pass through"
        );
    }

    // ── PR-11 · build_usage tests ─────────────────────────────────────

    /// PR-11: `build_usage` with non-zero cached + non-zero reasoning
    /// populates both Anthropic write keys, uses `thinking_tokens` for
    /// the OpenAI→Anthropic key translation, and excludes cached from
    /// `input_tokens` (Anthropic convention).
    #[test]
    fn build_usage_with_cached_and_reasoning() {
        let u = build_usage(100, 50, 30, 20, &None);
        assert_eq!(u.input_tokens, 70, "100 - 30 cached = 70");
        assert_eq!(u.output_tokens, 50);
        assert_eq!(u.cache_read_input_tokens, Some(30));
        assert_eq!(
            u.output_tokens_details.expect("reasoning>0 → present")["thinking_tokens"],
            20
        );
    }

    /// PR-11: zero cached → `cache_read_input_tokens` is None (Some(0)
    /// would change the wire shape from absent to zero).
    #[test]
    fn build_usage_zero_cached_keeps_field_absent() {
        let u = build_usage(10, 50, 0, 20, &None);
        assert!(u.cache_read_input_tokens.is_none());
    }

    /// PR-11: zero reasoning → `output_tokens_details` is None.
    #[test]
    fn build_usage_zero_reasoning_keeps_field_absent() {
        let u = build_usage(10, 50, 30, 0, &None);
        assert!(u.output_tokens_details.is_none());
    }

    /// PR-11 (clamp invariant): reasoning_tokens > output_tokens is
    /// clamped to output_tokens so the wire shape never violates
    /// Anthropic's `thinking_tokens ≤ output_tokens` invariant.
    #[test]
    fn build_usage_clamps_reasoning_to_output_tokens() {
        // 30 reasoning > 20 output → clamp to 20.
        let u = build_usage(10, 20, 0, 30, &None);
        let details = u.output_tokens_details.expect("clamped 20 > 0");
        assert_eq!(details["thinking_tokens"], 20);
        // The wire never sees `30 > 20`, so the Anthropic invariant
        // is preserved.
    }
}
