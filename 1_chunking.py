import os
import json
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

def create_chunks_from_sentences(sentences, max_chunk_size, overlap_size):
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    chunk_id = 0  # Initialize chunk ID

    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)  # Tokenize sentence into words
        sentence_token_count = len(sentence_tokens)

        # If adding this sentence exceeds the max size, save the current chunk
        if current_chunk_tokens + sentence_token_count > max_chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk)
            })  # Save chunk with metadata
            chunk_id += 1  # Increment chunk ID

            # Prepare the next chunk with overlap
            overlap_tokens = " ".join(current_chunk[-overlap_size:])  # Overlap handling
            current_chunk = overlap_tokens.split() if overlap_size > 0 else []
            current_chunk_tokens = len(current_chunk)

        # Add the sentence to the current chunk
        current_chunk.extend(sentence_tokens)
        current_chunk_tokens += sentence_token_count

    # Save the last chunk
    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(current_chunk)
        })

    return chunks


def process_folder(input_folder, max_chunk_size, overlap_size, output_file):
    all_chunks = {}  # Store chunks for each document

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Sentence tokenization
            sentences = sent_tokenize(text)

            # Create chunks
            chunks = create_chunks_from_sentences(sentences, max_chunk_size, overlap_size)
            all_chunks[filename] = chunks

    # Save chunks to a JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(all_chunks, json_file, indent=4)


if __name__ == "__main__":
    input_folder = "input"  # Folder containing text files
    output_file = "indexes/output_chunks.json"  # Output JSON file
    max_chunk_size = 600  # Maximum tokens per chunk
    overlap_size = 100    # Overlap size

    process_folder(input_folder, max_chunk_size, overlap_size, output_file)
    print(f"Chunks saved to {output_file}")
