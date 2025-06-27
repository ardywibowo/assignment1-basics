use fancy_regex::Regex;
use regex;
use std::collections::HashMap;

pub fn pretokenize_and_count(text: &str, special_tokens: &[String]) -> HashMap<Vec<u8>, u32> {
    // GPT-2 style regex pattern - exact match to Python version using fancy-regex for lookahead support
    let pattern = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
    
    let mut word_freq: HashMap<Vec<u8>, u32> = HashMap::new();
    
    // Split into documents using <|endoftext|> as a delimiter when present
    let docs = if special_tokens.contains(&"<|endoftext|>".to_string()) {
        text.split("<|endoftext|>").collect::<Vec<_>>()
    } else {
        vec![text]
    };
    
    // Remove any other special tokens from documents before tokenizing
    let other_specials: Vec<&String> = special_tokens.iter()
        .filter(|&t| t != "<|endoftext|>")
        .collect();
    
    let processed_docs = if !other_specials.is_empty() {
        let mut processed = Vec::new();
        for doc in docs {
            // Build regex pattern to split on special tokens (like Python version)
            let escaped_specials: Vec<String> = other_specials.iter()
                .map(|s| regex::escape(s))
                .collect();
            let pattern_str = format!("({})", escaped_specials.join("|"));
            let split_pattern = regex::Regex::new(&pattern_str).unwrap();
            
            // Split on special tokens and keep non-special parts
            let parts: Vec<&str> = split_pattern.split(doc).collect();
            for part in parts {
                if !part.is_empty() && !other_specials.iter().any(|&s| s == part) {
                    processed.push(part.to_string());
                }
            }
        }
        processed
    } else {
        docs.iter().map(|s| s.to_string()).collect()
    };
    
    // Apply regex pattern to cleaned documents
    for doc in processed_docs {
        for mat in pattern.find_iter(&doc) {
            let mat = mat.unwrap(); // fancy-regex returns Result
            let token = mat.as_str();
            let token_bytes = token.as_bytes().to_vec();
            
            // Store the complete token as bytes (like Python version)
            *word_freq.entry(token_bytes).or_insert(0) += 1;
        }
    }
    
    word_freq
}