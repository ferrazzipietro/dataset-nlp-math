from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os
from datasets import load_dataset

class Scorer:
    def __init__():
        pass

def extract_gold_bio_tags(wikiann_sample, label_name_map={'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORG'}):
    """Convert WikiANN BIO tags to BIO label sequences."""
    # WikiANN tag names are fixed: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    tag_names_wikiann = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    tokens = wikiann_sample['tokens']
    raw_ner_tags = wikiann_sample.get('ner_tags', [])
    
    bio_tags = []
    for tag_idx in raw_ner_tags:
        raw_label = tag_names_wikiann[tag_idx] if tag_idx < len(tag_names_wikiann) else 'O'
        if raw_label == 'O':
            bio_tags.append('O')
        else:
            prefix, base_label = raw_label.split('-', 1)
            mapped_label = f'{prefix}-{label_name_map.get(base_label, base_label)}'
            bio_tags.append(mapped_label)
    return bio_tags

def parse_prediction_string_to_bio(pred_string, tokens):
    """Parse prediction string (e.g., 'PERSON: name | ORG: org') to BIO tags."""
    # Simple fallback: if prediction is placeholder, return all 'O'
    if pred_string.startswith('<TODO_') and pred_string.endswith('>'):
        return ['O'] * len(tokens)
    # TODO: Parse actual prediction strings (students implement this)
    return ['O'] * len(tokens)

def main(input_dir: str, output_dir: str):
    print("=== Scoring program starting ===")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")

    # According to the docs, $input has:
    #   - ref/ : reference data
    #   - res/ : predictions
    ref_dir = os.path.join(input_dir, "ref")
    res_dir = os.path.join(input_dir, "res")

    ref_filename = "mock_data_dev_codabench_REFERENCE.jsonl"
    sub_filename = "mock_data_dev_codabench.jsonl"

    ref_path = os.path.join(ref_dir, ref_filename)
    sub_path = os.path.join(res_dir, sub_filename)

    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found at {ref_path}")
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission predictions not found at {sub_path}")

        
    # Collect gold and predicted sequences
    gold_sequences = []
    bert_sequences = []
    llama_sequences = []
    wikiann_eval = load_dataset('wikiann', 'en', split=f'validation[:{64}]')
    print('Loaded examples:', len(wikiann_eval))

    for i, sample in enumerate(wikiann_eval):
        gold_bio = extract_gold_bio_tags(sample)
        gold_sequences.append(gold_bio)
        
        bert_sequences.append(parse_prediction_string_to_bio(rows[i]['bert_prediction'], sample['tokens']))
        llama_sequences.append(parse_prediction_string_to_bio(rows[i]['llama_prediction'], sample['tokens']))

    # Compute metrics
    bert_precision = precision_score(gold_sequences, bert_sequences, zero_division=0)
    bert_recall = recall_score(gold_sequences, bert_sequences, zero_division=0)
    bert_f1 = f1_score(gold_sequences, bert_sequences, zero_division=0)

    llama_precision = precision_score(gold_sequences, llama_sequences, zero_division=0)
    llama_recall = recall_score(gold_sequences, llama_sequences, zero_division=0)
    llama_f1 = f1_score(gold_sequences, llama_sequences, zero_division=0)

    print('\n=== BERT NER Metrics ===')
    print(f'Precision: {bert_precision:.4f}')
    print(f'Recall: {bert_recall:.4f}')
    print(f'F1: {bert_f1:.4f}')
    print(classification_report(gold_sequences, bert_sequences, zero_division=0))

    print('\n=== Llama NER Metrics ===')
    print(f'Precision: {llama_precision:.4f}')
    print(f'Recall: {llama_recall:.4f}')
    print(f'F1: {llama_f1:.4f}')
    print(classification_report(gold_sequences, llama_sequences, zero_division=0))


    scores_path = os.path.join(output_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"f1_macro": float(score)}, f)

    print(f"Scores written to {scores_path}")
    print("=== Scoring program finished successfully ===")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python score.py <input_dir> <output_dir>\n"
            "Example: python score.py /app/input /app/output"
        )

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
