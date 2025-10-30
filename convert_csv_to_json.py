#!/usr/bin/env python3
"""
Script to convert CSV files with protein data into the proteinchat train_set format.
Supports both train and test CSV formats:
- Train format: Entry, EC number, Sequence
- Test format: ID, EC, Sequences
Output: qa_kw.json, seq.json, uniprot_ids.json, qa_text_manual.json, qa_text_rule.json
"""

import csv
import json
import os
from pathlib import Path


def create_qa_kw_entries(uniprot_id, ec_number):
    """
    Create QA entries for EC number prediction.

    Args:
        uniprot_id: The protein entry ID
        ec_number: The EC number to predict

    Returns:
        A dictionary entry for qa_kw.json
    """
    # Design the question prompt for EC number inference
    # Note: This question should be added to q_map in seq_dataset.py
    question = "What is the EC number of this protein?"

    return {
        "uniprot_id": uniprot_id,
        "Q": question,
        "Q_id": 0,  # Using Q_id 0 for EC number prediction
        "A": ec_number
    }


def load_json_file(file_path):
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Returning empty data.")
        return []


def filter_and_extend_qa_text(original_data, new_uniprot_ids):
    """
    Filter qa_text entries to only include IDs in new_uniprot_ids,
    and add placeholder entries for IDs not in the original data.

    Args:
        original_data: List of entries from original qa_text file
        new_uniprot_ids: Set of uniprot IDs from the new dataset

    Returns:
        Filtered and extended list of entries
    """
    # Create a mapping of uniprot_id to entry from original data
    id_to_entry = {entry['uniprot_id']: entry for entry in original_data}

    result = []
    for uniprot_id in new_uniprot_ids:
        if uniprot_id in id_to_entry:
            # Use existing entry from train_set
            result.append(id_to_entry[uniprot_id])
        else:
            # Create placeholder entry with empty caption
            result.append({
                "uniprot_id": uniprot_id,
                "caption": ""
            })

    return result


def convert_csv_to_json(csv_file_path, output_dir, train_set_dir, is_test=False):
    """
    Convert CSV file to complete train_set format.

    Args:
        csv_file_path: Path to the input CSV file
        output_dir: Directory where output JSON files will be saved
        train_set_dir: Directory containing the original train_set files
        is_test: Boolean flag indicating if this is a test file (different column names)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    qa_kw_data = []
    seq_data = {}
    uniprot_ids = []

    # Determine column names based on file type
    if is_test:
        id_col = 'ID'
        ec_col = 'EC'
        seq_col = 'Sequences'
    else:
        id_col = 'Entry'
        ec_col = 'EC number'
        seq_col = 'Sequence'

    # Read the CSV file
    print(f"Reading CSV file ({'test' if is_test else 'train'} format)...")
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')

        for row in reader:
            entry_id = row[id_col].strip()
            ec_number = row[ec_col].strip()
            sequence = row[seq_col].strip()

            # Create QA entry
            qa_entry = create_qa_kw_entries(entry_id, ec_number)
            qa_kw_data.append(qa_entry)

            # Create sequence entry
            seq_data[entry_id] = sequence

            # Add to uniprot_ids list
            uniprot_ids.append(entry_id)

    # Write qa_kw.json
    print("Writing qa_kw.json...")
    qa_kw_output_path = os.path.join(output_dir, 'qa_kw.json')
    with open(qa_kw_output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_kw_data, f, indent=4)

    # Write seq.json
    print("Writing seq.json...")
    seq_output_path = os.path.join(output_dir, 'seq.json')
    with open(seq_output_path, 'w', encoding='utf-8') as f:
        json.dump(seq_data, f, indent=4)

    # Write uniprot_ids.json
    print("Writing uniprot_ids.json...")
    uniprot_ids_path = os.path.join(output_dir, 'uniprot_ids.json')
    with open(uniprot_ids_path, 'w', encoding='utf-8') as f:
        json.dump(uniprot_ids, f, indent=4)

    # Process qa_text_manual.json
    print("Processing qa_text_manual.json...")
    original_qa_text_manual = load_json_file(
        os.path.join(train_set_dir, 'qa_text_manual.json')
    )
    new_qa_text_manual = filter_and_extend_qa_text(
        original_qa_text_manual,
        set(uniprot_ids)
    )
    qa_text_manual_path = os.path.join(output_dir, 'qa_text_manual.json')
    with open(qa_text_manual_path, 'w', encoding='utf-8') as f:
        json.dump(new_qa_text_manual, f, indent=4)

    # Process qa_text_rule.json
    print("Processing qa_text_rule.json...")
    original_qa_text_rule = load_json_file(
        os.path.join(train_set_dir, 'qa_text_rule.json')
    )
    new_qa_text_rule = filter_and_extend_qa_text(
        original_qa_text_rule,
        set(uniprot_ids)
    )
    qa_text_rule_path = os.path.join(output_dir, 'qa_text_rule.json')
    with open(qa_text_rule_path, 'w', encoding='utf-8') as f:
        json.dump(new_qa_text_rule, f, indent=4)

    # Print summary
    print(f"\nConversion completed successfully!")
    print(f"Total entries processed: {len(uniprot_ids)}")
    print(f"\nOutput files created:")
    print(f"  - {qa_kw_output_path}")
    print(f"  - {seq_output_path}")
    print(f"  - {uniprot_ids_path}")
    print(f"  - {qa_text_manual_path} ({len(new_qa_text_manual)} entries)")
    print(f"  - {qa_text_rule_path} ({len(new_qa_text_rule)} entries)")


def main():
    # Base directory paths
    base_dir = "/data3/ruiyi/proteinchat/CLEAN_all_train_valid_splits/split10"
    train_set_directory = "/data3/ruiyi/proteinchat/proteinchat-data/train_set"

    # Process train split
    print("=" * 80)
    print("PROCESSING TRAIN SPLIT")
    print("=" * 80)
    train_csv = f"{base_dir}/split10_train_split_1.csv"
    train_output = f"{base_dir}/split10_train_split_1_converted"
    convert_csv_to_json(train_csv, train_output, train_set_directory, is_test=False)

    # Process test split
    print("\n" + "=" * 80)
    print("PROCESSING TEST SPLIT")
    print("=" * 80)
    test_csv = f"{base_dir}/split10_test_split_1_curate.csv"
    test_output = f"{base_dir}/split10_test_split_1_converted"
    convert_csv_to_json(test_csv, test_output, train_set_directory, is_test=True)


if __name__ == "__main__":
    main()
