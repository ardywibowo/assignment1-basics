use std::collections::HashMap;
use crate::pretokenize::pretokenize_and_count;
use crate::common::{bytes_to_u32, update_counts, merge};
use indicatif::{ProgressBar, ProgressStyle};

pub fn train(
    text: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
    merges: &mut HashMap<(u32, u32), u32>,
    vocab: &mut HashMap<u32, String>,
    _counts: &mut HashMap<(u32, u32), u32>,
    progress: bool,
) -> (Vec<(u32, u32)>, HashMap<u32, Vec<u8>>) {
    // Keep track of bytes separately to avoid UTF-8 issues
    let mut vocab_bytes: HashMap<u32, Vec<u8>> = HashMap::new();
    
    // Initialize base vocab with all 256 byte values
    for i in 0..=255 {
        vocab.insert(i as u32, String::from_utf8_lossy(&[i as u8]).into_owned());
        vocab_bytes.insert(i as u32, vec![i as u8]);
    }
    
    // Pre-tokenize text using regex pattern
    let word_freq = pretokenize_and_count(text, &special_tokens);
    
    // Convert word frequencies to token sequences
    let mut word_tokens: HashMap<Vec<u32>, u32> = HashMap::new();
    for (word_bytes, freq) in word_freq {
        let tokens: Vec<u32> = word_bytes.iter().map(|&b| b as u32).collect();
        word_tokens.insert(tokens, freq);
    }
    
    let target_size = vocab_size - special_tokens.len();
    let num_merges = target_size - 256;
    let mut next_id = 256u32;
    let mut ordered_merges = Vec::new();
    
    // Create progress bar for BPE merges
    let pb = if progress {
        let pb = ProgressBar::new(num_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{desc} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]")
                .unwrap()
                .progress_chars("##-")
        );
        pb.set_message("BPE merges");
        Some(pb)
    } else {
        None
    };
    
    for _i in 0..num_merges {
        // Calculate pair frequencies
        let mut pair_freq: HashMap<(u32, u32), u32> = HashMap::new();
        for (word, freq) in &word_tokens {
            for j in 0..word.len().saturating_sub(1) {
                let pair = (word[j], word[j + 1]);
                *pair_freq.entry(pair).or_insert(0) += freq;
            }
        }
        
        if pair_freq.is_empty() {
            break;
        }
        
        // Find the most frequent pair (with lexicographic tie-breaking)
        let max_freq = *pair_freq.values().max().unwrap();
        let mut candidates: Vec<(u32, u32)> = pair_freq.iter()
            .filter(|(_, &freq)| freq == max_freq)
            .map(|(&pair, _)| pair)
            .collect();
        
        // Lexicographically greatest pair (like Python max())
        // Python compares (bytes, bytes) tuples, we need to compare based on the actual byte values
        candidates.sort_by_key(|&(a, b)| {
            let a_bytes = if a < 256 { vec![a as u8] } else { vocab[&a].as_bytes().to_vec() };
            let b_bytes = if b < 256 { vec![b as u8] } else { vocab[&b].as_bytes().to_vec() };
            (a_bytes, b_bytes)
        });
        let best_pair = *candidates.last().unwrap();
        
        // Create new token by merging bytes
        let mut new_token_bytes = vocab_bytes[&best_pair.0].clone();
        new_token_bytes.extend_from_slice(&vocab_bytes[&best_pair.1]);
        
        let new_token = String::from_utf8_lossy(&new_token_bytes).into_owned();
        vocab.insert(next_id, new_token.clone());
        vocab_bytes.insert(next_id, new_token_bytes);
        merges.insert(best_pair, next_id);
        ordered_merges.push(best_pair);
        
        
        // Update word_tokens by applying the merge
        let mut new_word_tokens: HashMap<Vec<u32>, u32> = HashMap::new();
        for (word, freq) in word_tokens {
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == best_pair.0 && word[i + 1] == best_pair.1 {
                    new_word.push(next_id);
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }
            new_word_tokens.insert(new_word, freq);
        }
        word_tokens = new_word_tokens;
        next_id += 1;
        
        // Update progress bar
        if let Some(ref pb) = pb {
            pb.inc(1);
        }
    }
    
    // Finish progress bar
    if let Some(ref pb) = pb {
        pb.finish_with_message("BPE merges completed");
    }
    
    // Add special tokens to vocab
    for special_token in special_tokens {
        let special_bytes = special_token.as_bytes().to_vec();
        vocab.insert(next_id, special_token);
        vocab_bytes.insert(next_id, special_bytes);
        next_id += 1;
    }
    
    (ordered_merges, vocab_bytes)
}

pub fn encode(text: &str, merges: &HashMap<(u32, u32), u32>) -> Vec<u32> {
    let text_bytes = text.as_bytes();
    let mut ids = bytes_to_u32(text_bytes);

    while ids.len() >= 2 {
        let mut stats = HashMap::new();
        update_counts(&ids, &mut stats);
        let min_pair = stats
            .iter()
            .min_by_key(|&(p, _)| merges.get(p).cloned().unwrap_or(std::u32::MAX))
            .unwrap().0;

        if !merges.contains_key(&min_pair) {
            break;
        }

        let idx = merges.get(&min_pair).unwrap();
        ids = merge(ids, *min_pair, *idx);
    }

    ids
}

pub fn decode(ids: &[u32], vocab: &HashMap<u32, String>) -> String {
    let mut text = String::new();
    for &id in ids {
        text.push_str(&vocab[&id]);
    }
    text
}
