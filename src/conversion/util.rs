//! Shared helpers used by multiple conversion modules.
//!
//! Anything that isn't tied to a specific upstream wire format (Chat
//! Completions vs Responses API) but needs to be reused across both
//! conversion paths lives here. This keeps the individual conversion
//! modules focused on their own protocol translation and avoids
//! cross-module imports between them (e.g. `responses.rs` reaching into
//! `request.rs`).

use serde_json::Value;

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
/// # Limitations
///
/// - Does NOT resolve `$ref` — a schema referencing `#/$defs/Foo` will
///   have the referenced definition strictified (via the `$defs` pass),
///   but the `$ref` reference itself is left untouched. OpenAI strict
///   mode rejects `$ref` in schemas, so callers with `$ref`-based
///   schemas must flatten them first.
pub fn strictify_schema(schema: &mut Value) {
    let Value::Object(obj) = schema else { return };
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
            for (_, v) in p.iter_mut() {
                strictify_schema(v);
            }
        }
    }
    if let Some(items) = obj.get_mut("items") {
        strictify_schema(items);
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(arr)) = obj.get_mut(key) {
            for sub in arr.iter_mut() {
                strictify_schema(sub);
            }
        }
    }
    for key in ["$defs", "definitions", "dependentSchemas"] {
        if let Some(Value::Object(map)) = obj.get_mut(key) {
            for (_, v) in map.iter_mut() {
                strictify_schema(v);
            }
        }
    }
    for key in ["not", "if", "then", "else", "contains", "propertyNames", "unevaluatedItems"] {
        if let Some(sub) = obj.get_mut(key) {
            strictify_schema(sub);
        }
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
        strictify_schema(&mut schema);

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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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

    #[test]
    fn strictify_schema_non_object_returns_early() {
        // A schema whose top-level type is not "object" (e.g. "string")
        // must not be rewritten — strict mode rules only apply to object
        // schemas. The function takes &mut Value and returns nothing, so
        // we assert the schema is unchanged.
        let mut schema = json!({"type": "string"});
        let original = schema.clone();
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);

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
        strictify_schema(&mut schema);

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
        strictify_schema(&mut schema);

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
        strictify_schema(&mut schema);

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

        strictify_schema(&mut schema1);
        strictify_schema(&mut schema2);
        strictify_schema(&mut schema2);

        assert_eq!(schema1, schema2, "double-call must be idempotent");
    }

    #[test]
    fn strictify_schema_handles_malformed_null_value() {
        // A null schema value must not panic. Strictify returns
        // immediately when the value isn't an object.
        let mut schema = Value::Null;
        strictify_schema(&mut schema);
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
            strictify_schema(&mut schema);
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
        strictify_schema(&mut schema);
        assert_eq!(
            schema, original,
            "schema with properties but no `type: object` must pass through"
        );
    }
}
