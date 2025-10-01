use anyhow::{Context, Result};
use cust::module::Module;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

static MODULE_CACHE: Lazy<Mutex<HashMap<&'static str, Arc<Module>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub fn module(ptx: &'static str, key: &'static str) -> Result<Arc<Module>> {
    let mut cache = MODULE_CACHE
        .lock()
        .expect("Kernel module cache mutex poisoned");

    if let Some(existing) = cache.get(key) {
        return Ok(existing.clone());
    }

    let module =
        Module::from_ptx(ptx, &[]).with_context(|| format!("PTX load failed for {key}"))?;
    let arc = Arc::new(module);
    cache.insert(key, arc.clone());
    Ok(arc)
}
