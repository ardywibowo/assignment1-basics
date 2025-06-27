mod dataset;
mod bpe;
mod common;
mod pretokenize;

use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (text, vocab_size, special_tokens, progress = false))]
fn train_bpe_py(text: &str, vocab_size: usize, special_tokens: Vec<String>, progress: bool) -> PyResult<(HashMap<u32, Vec<u8>>, Vec<(u32, u32)>)> {
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    let mut merges: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vocab: HashMap<u32, String> = HashMap::new();

    let (merges_vec, vocab_bytes) = bpe::train(text, vocab_size, special_tokens, &mut merges, &mut vocab, &mut counts, progress);

    Ok((vocab_bytes, merges_vec))
}

#[pymodule]
fn tokenizer_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_py, m)?)?;
    Ok(())
}
