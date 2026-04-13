fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        // DuckDB's bundled build references Windows Restart Manager APIs.
        println!("cargo:rustc-link-lib=rstrtmgr");
    }
}
