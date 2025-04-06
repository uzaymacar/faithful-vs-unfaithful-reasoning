import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score

from utils import load_yaml_files

nltk.download('punkt_tab')

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze n-gram patterns in faithful vs unfaithful reasoning')
    parser.add_argument('-m', '--model_name', type=str, default='qwen2.5-1.5b-instruct', choices=['qwen2.5-1.5b-instruct', 'llama-3.1-8b-instruct'], help='Name of the model to analyze')
    parser.add_argument('-o', '--output_dir', type=str, default='figures/token_analysis', help='Directory to save analysis results')
    parser.add_argument('-n', '--n_grams', type=int, default=3, help='Maximum n-gram size to analyze (will analyze from 1 to this number)')
    parser.add_argument('-k', '--top_k', type=int, default=30, help='Number of top n-grams to display')
    return parser.parse_args()

def preprocess_text(text):
    """Clean and tokenize text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (optional - might want to keep for this analysis)
    # from nltk.corpus import stopwords
    # stop_words = set(stopwords.words('english'))
    # tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def extract_ngrams(tokens, n):
    """Extract n-grams from a list of tokens."""
    return list(ngrams(tokens, n))

def analyze_ngram_distribution(cots, n_max=3, top_k=30):
    """Analyze n-gram distribution in faithful vs unfaithful examples."""
    faithful_cots = [cot for cot in cots if cot['is_faithful']]
    unfaithful_cots = [cot for cot in cots if not cot['is_faithful']]
    
    print(f"Analyzing {len(faithful_cots)} faithful and {len(unfaithful_cots)} unfaithful examples")
    
    results = {}
    
    for n in range(1, n_max + 1):
        print(f"Analyzing {n}-grams...")
        
        # Count n-grams in faithful examples
        faithful_ngrams = Counter()
        for cot in tqdm(faithful_cots, desc=f"Faithful {n}-grams"):
            tokens = preprocess_text(cot['generated_cot'])
            cot_ngrams = extract_ngrams(tokens, n)
            faithful_ngrams.update(cot_ngrams)
        
        # Count n-grams in unfaithful examples
        unfaithful_ngrams = Counter()
        for cot in tqdm(unfaithful_cots, desc=f"Unfaithful {n}-grams"):
            tokens = preprocess_text(cot['generated_cot'])
            cot_ngrams = extract_ngrams(tokens, n)
            unfaithful_ngrams.update(cot_ngrams)
        
        # Calculate relative frequencies
        total_faithful = sum(faithful_ngrams.values())
        total_unfaithful = sum(unfaithful_ngrams.values())
        
        faithful_freq = {ngram: count/total_faithful for ngram, count in faithful_ngrams.items()}
        unfaithful_freq = {ngram: count/total_unfaithful for ngram, count in unfaithful_ngrams.items()}
        
        # Find n-grams that appear in both sets
        common_ngrams = set(faithful_ngrams.keys()) & set(unfaithful_ngrams.keys())
        
        # Calculate ratio of frequencies (how much more common in unfaithful vs faithful)
        ngram_ratios = {}
        for ngram in common_ngrams:
            if faithful_freq[ngram] > 0:
                ratio = unfaithful_freq[ngram] / faithful_freq[ngram]
                ngram_ratios[ngram] = ratio
        
        # Find n-grams unique to each category
        faithful_only = set(faithful_ngrams.keys()) - set(unfaithful_ngrams.keys())
        unfaithful_only = set(unfaithful_ngrams.keys()) - set(faithful_ngrams.keys())
        
        # Get top n-grams by frequency in each category
        top_faithful = sorted(faithful_ngrams.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_unfaithful = sorted(unfaithful_ngrams.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Get n-grams with highest ratio (more common in unfaithful)
        top_unfaithful_ratio = sorted(ngram_ratios.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Get n-grams with lowest ratio (more common in faithful)
        top_faithful_ratio = sorted(ngram_ratios.items(), key=lambda x: x[1])[:top_k]
        
        results[n] = {
            'top_faithful': top_faithful,
            'top_unfaithful': top_unfaithful,
            'top_unfaithful_ratio': top_unfaithful_ratio,
            'top_faithful_ratio': top_faithful_ratio,
            'faithful_only_count': len(faithful_only),
            'unfaithful_only_count': len(unfaithful_only),
            'common_count': len(common_ngrams)
        }
    
    return results

def analyze_with_tfidf(cots):
    """Use TF-IDF to find discriminative terms."""
    faithful_texts = [cot['generated_cot'] for cot in cots if cot['is_faithful']]
    unfaithful_texts = [cot['generated_cot'] for cot in cots if not cot['is_faithful']]
    
    # Combine all texts
    all_texts = faithful_texts + unfaithful_texts
    labels = [1] * len(faithful_texts) + [0] * len(unfaithful_texts)
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
    X = tfidf.fit_transform(all_texts)
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Calculate mutual information between features and labels
    mi_scores = []
    for i in range(X.shape[1]):
        feature_values = X[:, i].toarray().flatten()
        mi = mutual_info_score(labels, feature_values > 0)  # Binarize feature
        mi_scores.append((feature_names[i], mi))
    
    # Sort by mutual information
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    
    return mi_scores[:50]  # Return top 50 features

def analyze_reasoning_patterns(cots):
    """Analyze reasoning patterns and markers in the text."""
    # Define patterns to look for
    reasoning_markers = {
        # Markers indicating uncertainty or tentativeness
        'uncertainty': [
            'maybe', 'perhaps', 'might', 'could', 'possibly', 'not sure', 
            'uncertain', 'likely', 'plausible', 'seems', 'generally', 
            'approximately', 'around', 'about', 'suggests'
        ],
        
        # Markers indicating high confidence or stating facts
        'confidence': [
            'definitely', 'certainly', 'clearly', 'obviously', 'without doubt', 
            'indeed', 'in fact', 'it is evident', 'it\'s clear that' 
        ],
        
        # Markers indicating a contrast or contradiction
        'contradiction': [
            'however', 'but', 'although', 'despite', 'nevertheless', 
            'on the other hand', 'in contrast', 'conversely', 'while'
        ],
        
        # Markers indicating self-correction or clarification
        'correction': [
            # Explicit corrections
            "i made a mistake",
            "i have a mistake", 
            "i have an error",
            "let me correct",
            "let me revise",
            "let me reconsider",
            "to correct myself",
            "to correct this",
            "to correct that",
            "correcting myself",
            "i need to correct",
            "i should correct",
            "i should revise",
            "i was wrong",
            "i was incorrect",
            "i was mistaken",
            "that's not right",
            "that's not correct",
            "this is wrong",
            "this is incorrect",
            "i misspoke",
            "i miscalculated",
            "i misunderstood",
            
            # Clarifications
            "to clarify",
            "let me clarify",
            "just to clarify",
            "for clarity",
            "to be clear",
            "to be more clear",
            "to be precise",
            
            # Revisions
            "let me revise",
            "let me rethink",
            "let me reconsider",
            "i should revise",
            "i should rethink",
            "i should reconsider",
            "on second thought",
            "thinking again",
            "reconsidering",
            
            # Contradiction markers
            "actually",
            "in fact",
            "wait",
            "hang on",
            "hold on",
            "no,",
            "nope",
            "wait a second",
            "wait a minute",
            
            # Realization markers
            "i just realize",
            "i now realize",
            "i see now",
            "now i see",
            "now i understand",
            "now i realize",
            "i didn't consider",
            "i didn't account for",
            "i forgot to",
            "i forgot about",
            "i overlooked",
            "i missed",
            
            # Backtracking
            "going back",
            "let's go back",
            "backtracking",
            "retracing",
            
            # Correction transitions
            "however,",
            "but,",
            "nevertheless,",
            "yet,",
            "still,",
            "despite,",
            "on the contrary,",
            "conversely,",
            
            # Correction phrases
            "i need to fix",
            "let me fix",
            "that's incorrect",
            "i made an error",
            "i was confused",
            "i meant to say",
            "what i meant was",
            "to be more accurate",
            "more precisely",
            "to put it correctly",
            "i should have said",
            "let me be more specific",
            "to be exact",
            "i should clarify",
            "let me double-check",
            "checking my work",
            "upon reflection",
            "thinking more carefully",
            "i need to recalculate",
            "recalculating",
            "i made a computational error",
            "i made a calculation error",
            "i made a logical error",
            "i was thinking incorrectly",
            "i was reasoning incorrectly"
        ],
        
        # Markers related to mathematical or quantitative operations/comparisons
        'math_operations': [
            'add', 'subtract', 'multiply', 'divide', 'calculate', 'compute', 
            'equals', 'greater than', 'less than', 'value', 'number', 'amount'
        ],
        
        # --- New Categories Based on Data Analysis ---
        
        # Markers indicating the structure or steps in reasoning
        'structuring': [
            'let\'s break this down', 'step-by-step', 'first,', 'second,', 'next,', 
            'finally,', 'step 1:', 'step 2:', 'step 3:', 'to determine', 'identify', 
            'consider the following points', 'let\'s consider'
        ],
        
        # Markers explicitly stating a comparison is being made
        'comparison': [
            'compare', 'comparing', 'compared to', 'lower than', 'higher than', 
            'shorter than', 'longer than', 'earlier than', 'later than', 
            'older than', 'newer than', 'faster than', 'slower than', 
            'less dense', 'denser than', 'vs', 'versus', 'relative to', 
            'difference'
        ],
        
        # Markers indicating a logical cause-and-effect or deduction
        'causality': [
            'therefore', 'thus', 'hence', 'since', 'because', 'given', 'due to', 
            'as a result', 'it follows that', 'based on this', 'so', 'which means' 
        ],
        
        # Markers indicating reliance on facts, data, or premises
        'factual_basis': [
            'according to', 'based on', 'given these facts', 'the facts are', 
            'we need to consider', 'the information', 'the data shows'
        ],
        
        # Markers explicitly stating the final conclusion or answer
        'conclusion': [
            'the answer is:', 'conclusion:', 'final answer:', 'therefore, my answer is:', 
            'in conclusion,', 'thus, the answer is:'
        ]
    }
    
    # Count occurrences in faithful and unfaithful examples
    faithful_counts = {category: 0 for category in reasoning_markers}
    unfaithful_counts = {category: 0 for category in reasoning_markers}
    
    # Count total words
    faithful_total_words = 0
    unfaithful_total_words = 0
    
    # Process each CoT
    for cot in cots:
        text = cot['generated_cot'].lower()
        tokens = word_tokenize(text)
        
        if cot['is_faithful']:
            faithful_total_words += len(tokens)
        else:
            unfaithful_total_words += len(tokens)
        
        # Count marker occurrences
        for category, markers in reasoning_markers.items():
            count = sum(1 for marker in markers if marker in text)
            if cot['is_faithful']:
                faithful_counts[category] += count
            else:
                unfaithful_counts[category] += count
    
    # Calculate frequencies per 1000 words
    faithful_freq = {cat: count / faithful_total_words * 1000 
                    for cat, count in faithful_counts.items()}
    unfaithful_freq = {cat: count / unfaithful_total_words * 1000 
                      for cat, count in unfaithful_counts.items()}
    
    # Calculate ratios
    ratios = {cat: unfaithful_freq[cat] / faithful_freq[cat] if faithful_freq[cat] > 0 else float('inf')
             for cat in reasoning_markers}
    
    return {
        'faithful_freq': faithful_freq,
        'unfaithful_freq': unfaithful_freq,
        'ratios': ratios
    }

def analyze_sentence_structure(cots):
    """Analyze sentence structure and complexity."""
    faithful_lengths = []
    unfaithful_lengths = []
    
    for cot in cots:
        text = cot['generated_cot']
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            if cot['is_faithful']:
                faithful_lengths.append(len(tokens))
            else:
                unfaithful_lengths.append(len(tokens))
    
    return {
        'faithful_lengths': faithful_lengths,
        'unfaithful_lengths': unfaithful_lengths
    }

def plot_ngram_results(results, output_dir, model_name):
    """Plot the n-gram analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    for n, data in results.items():
        # Plot top n-grams by frequency
        plt.figure(figsize=(12, 8))
        
        # Top faithful n-grams
        labels = [' '.join(ngram) for ngram, _ in data['top_faithful'][:15]]
        values = [count for _, count in data['top_faithful'][:15]]
        
        plt.subplot(2, 1, 1)
        plt.barh(labels, values, color='blue', alpha=0.7)
        plt.title(f'Top {n}-grams in Faithful Examples')
        plt.xlabel('Count')
        plt.tight_layout()
        
        # Top unfaithful n-grams
        labels = [' '.join(ngram) for ngram, _ in data['top_unfaithful'][:15]]
        values = [count for _, count in data['top_unfaithful'][:15]]
        
        plt.subplot(2, 1, 2)
        plt.barh(labels, values, color='red', alpha=0.7)
        plt.title(f'Top {n}-grams in Unfaithful Examples')
        plt.xlabel('Count')
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/{model_name}_{n}_gram_frequency.png')
        plt.close()
        
        # Plot n-grams with highest ratio difference
        plt.figure(figsize=(12, 8))
        
        # More common in unfaithful
        labels = [' '.join(ngram) for ngram, _ in data['top_unfaithful_ratio'][:15]]
        values = [ratio for _, ratio in data['top_unfaithful_ratio'][:15]]
        
        plt.subplot(2, 1, 1)
        plt.barh(labels, values, color='red', alpha=0.7)
        plt.title(f'{n}-grams More Common in Unfaithful Examples (Ratio)')
        plt.xlabel('Unfaithful / Faithful Frequency Ratio')
        plt.tight_layout()
        
        # More common in faithful
        labels = [' '.join(ngram) for ngram, _ in data['top_faithful_ratio'][:15]]
        values = [1/ratio for _, ratio in data['top_faithful_ratio'][:15]]  # Invert for visualization
        
        plt.subplot(2, 1, 2)
        plt.barh(labels, values, color='blue', alpha=0.7)
        plt.title(f'{n}-grams More Common in Faithful Examples (Ratio)')
        plt.xlabel('Faithful / Unfaithful Frequency Ratio')
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/{model_name}_{n}_gram_ratio.png')
        plt.close()

def plot_tfidf_results(mi_scores, output_dir, model_name):
    """Plot the TF-IDF analysis results."""
    plt.figure(figsize=(12, 8))
    
    features = [feature for feature, _ in mi_scores[:20]]
    scores = [score for _, score in mi_scores[:20]]
    
    plt.barh(features, scores, color='purple', alpha=0.7)
    plt.title('Top Features by Mutual Information with Faithfulness')
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_tfidf_features.png')
    plt.close()

def plot_reasoning_patterns(pattern_results, output_dir, model_name):
    """Plot the reasoning pattern analysis results."""
    plt.figure(figsize=(12, 6))
    
    categories = list(pattern_results['faithful_freq'].keys())
    faithful_values = [pattern_results['faithful_freq'][cat] for cat in categories]
    unfaithful_values = [pattern_results['unfaithful_freq'][cat] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, faithful_values, width, label='Faithful', color='blue', alpha=0.7)
    plt.bar(x + width/2, unfaithful_values, width, label='Unfaithful', color='red', alpha=0.7)
    
    plt.xlabel('Reasoning Pattern Category')
    plt.ylabel('Frequency per 1000 words')
    plt.title('Reasoning Pattern Frequencies')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_reasoning_patterns.png')
    plt.close()
    
    # Plot ratios
    plt.figure(figsize=(10, 6))
    
    ratios = [pattern_results['ratios'][cat] for cat in categories]
    
    plt.bar(categories, ratios, color='purple', alpha=0.7)
    plt.axhline(y=1, color='black', linestyle='--')
    plt.xlabel('Reasoning Pattern Category')
    plt.ylabel('Unfaithful / Faithful Ratio')
    plt.title('Reasoning Pattern Frequency Ratios')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_reasoning_pattern_ratios.png')
    plt.close()

def plot_sentence_structure(structure_results, output_dir, model_name):
    """Plot the sentence structure analysis results."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(structure_results['faithful_lengths'], bins=20, alpha=0.5, label='Faithful', color='blue')
    plt.hist(structure_results['unfaithful_lengths'], bins=20, alpha=0.5, label='Unfaithful', color='red')
    
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.title('Sentence Length Distribution')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/{model_name}_sentence_lengths.png')
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_dir = f'data/{args.model_name}'
    cots = load_yaml_files(data_dir)
    
    print(f"Loaded {len(cots)} examples from {data_dir}")
    
    # Filter to only include examples with explicit faithfulness labels
    labeled_cots = [cot for cot in cots if 'is_faithful' in cot]
    
    print(f"Found {len(labeled_cots)} labeled examples")
    print(f"Faithful: {sum(1 for cot in labeled_cots if cot['is_faithful'])}")
    print(f"Unfaithful: {sum(1 for cot in labeled_cots if not cot['is_faithful'])}")
    
    # Analyze n-grams
    ngram_results = analyze_ngram_distribution(labeled_cots, args.n_grams, args.top_k)
    
    # Plot n-gram results
    plot_ngram_results(ngram_results, args.output_dir, args.model_name.lower())
    
    # Analyze with TF-IDF
    print("Analyzing with TF-IDF...")
    tfidf_results = analyze_with_tfidf(labeled_cots)
    
    # Plot TF-IDF results
    plot_tfidf_results(tfidf_results, args.output_dir, args.model_name.lower())
    
    # Analyze reasoning patterns
    print("Analyzing reasoning patterns...")
    pattern_results = analyze_reasoning_patterns(labeled_cots)
    
    # Plot reasoning pattern results
    plot_reasoning_patterns(pattern_results, args.output_dir, args.model_name.lower())
    
    # Analyze sentence structure
    print("Analyzing sentence structure...")
    structure_results = analyze_sentence_structure(labeled_cots)
    
    # Plot sentence structure results
    plot_sentence_structure(structure_results, args.output_dir, args.model_name.lower())
    
    print(f"Analysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()