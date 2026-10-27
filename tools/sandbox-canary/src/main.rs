//! Sandbox canary 的占位二进制。真正的逻辑在 `build.rs` 里 —— 它在**编译期**执行,
//! 这个二进制只是为了让 `cargo build` 有个可链接的 target。

fn main() {
    println!("sandbox-canary");
}
