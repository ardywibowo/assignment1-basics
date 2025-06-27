use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs;

#[pyfunction]
pub fn train_bpe(
    input_path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(Py<PyDict>, Py<PyList>)> {
    let text = fs::read_to_string(input_path)?;
    let mut docs: Vec<String> = if special_tokens.iter().any(|t| t == "<|endoftext|>") {
        text.split("<|endoftext|>").map(|s| s.to_string()).collect()
    } else {
        vec![text]
    };

    let other_specials: Vec<String> = special_tokens
        .iter()
        .filter(|&t| t != "<|endoftext|>")
        .cloned()
        .collect();
    if !other_specials.is_empty() {
        let pattern = format!(
            "({})",
            other_specials
                .iter()
                .map(|s| regex::escape(s))
                .collect::<Vec<_>>()
                .join("|")
        );
        let re = Regex::new(&pattern).unwrap();
        let mut processed = Vec::new();
        for doc in docs {
            for part in re.split(&doc) {
                if !part.is_empty() && !other_specials.contains(&part.to_string()) {
                    processed.push(part.to_string());
                }
            }
        }
        docs = processed;
    }

    let token_re = Regex::new("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\v\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+").unwrap();

    let mut word_freq: HashMap<Vec<Vec<u8>>, usize> = HashMap::new();
    for doc in docs {
        for m in token_re.find_iter(&doc) {
            let tok = m.as_str();
            let bytes: Vec<Vec<u8>> = tok.as_bytes().iter().map(|&b| vec![b]).collect();
            *word_freq.entry(bytes).or_insert(0) += 1;
        }
    }

    let mut vocab: HashMap<u32, Vec<u8>> = (0u32..256).map(|i| (i, vec![i as u8])).collect();
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut next_id: u32 = 256;
    let target_size = vocab_size.saturating_sub(special_tokens.len());

    while (vocab.len() as usize) < target_size {
        let mut pair_freq: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
        for (word, &freq) in &word_freq {
            for pair in word.windows(2) {
                if let [a, b] = pair {
                    *pair_freq.entry((a.clone(), b.clone())).or_insert(0) += freq;
                }
            }
        }
        if pair_freq.is_empty() {
            break;
        }
        let mut best_pair: Option<((Vec<u8>, Vec<u8>), usize)> = None;
        for (pair, freq) in pair_freq.into_iter() {
            match &mut best_pair {
                Some(bp) if freq > bp.1 || (freq == bp.1 && pair > bp.0) => *bp = (pair, freq),
                None => best_pair = Some((pair, freq)),
                _ => {}
            }
        }
        let (pair, _freq) = match best_pair {
            Some(p) => p,
            None => break,
        };
        let new_token = [pair.0.clone(), pair.1.clone()].concat();
        vocab.insert(next_id, new_token.clone());
        merges.push(pair.clone());
        next_id += 1;

        let mut new_word_freq: HashMap<Vec<Vec<u8>>, usize> = HashMap::new();
        for (word, freq) in word_freq.into_iter() {
            let mut new_word: Vec<Vec<u8>> = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                    new_word.push(new_token.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            *new_word_freq.entry(new_word).or_insert(0) += freq;
        }
        word_freq = new_word_freq;
    }

    for st in special_tokens {
        vocab.insert(next_id, st.into_bytes());
        next_id += 1;
    }

    Python::with_gil(|py| {
        let py_vocab = PyDict::new(py);
        for (id, tok) in &vocab {
            py_vocab.set_item(*id, PyBytes::new(py, tok))?;
        }
        let py_merges = PyList::empty(py);
        for (a, b) in &merges {
            py_merges.append((PyBytes::new(py, a), PyBytes::new(py, b)))?;
        }
        Ok((py_vocab.into_py(py), py_merges.into_py(py)))
    })
}

#[pymodule]
fn rsbpe(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}
