mod dataset;
mod bpe;
mod common;

use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn train_bpe_py(text: &str) -> PyResult<(HashMap<u32, String>, Vec<(u32, u32)>)> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vocab: HashMap<u32, String> = HashMap::new();

    bpe::train(text, &mut merges, &mut vocab, &mut counts);

    let merges_vec: Vec<(u32, u32)> = merges.into_iter().map(|(k, _)| k).collect();

    Ok((vocab, merges_vec))
}

#[pymodule]
fn tokenizer_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_py, m)?)?;
    Ok(())
}
