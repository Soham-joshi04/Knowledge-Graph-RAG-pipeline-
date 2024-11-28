import json
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure Llama 3.2
llm = Ollama(model="llama3.2")

# Chunk Summarization Prompt Template
chunk_summary_prompt = PromptTemplate(
    template="""
    Context: {chunk}

    Task: Summarize the provided context into a concise and information-rich paragraph that captures the main points.
    """
)

# File Summarization Prompt Template
file_summary_prompt = PromptTemplate(
    template="""
    Summaries of the chunks from the file:
    {chunk_summaries}

    Task: Combine these summaries into a single concise and comprehensive summary of the entire file. Capture the high-level information and main themes.
    """
)

# Function to summarize a single chunk
def summarize_chunk(chunk_text):
    chain = LLMChain(llm=llm, prompt=chunk_summary_prompt)
    response = chain.run({"chunk": chunk_text})
    return response.strip()

# Function to summarize an entire file based on chunk summaries
def summarize_file(chunk_summaries):
    chain = LLMChain(llm=llm, prompt=file_summary_prompt)
    response = chain.run({"chunk_summaries": "\n".join(chunk_summaries)})
    return response.strip()

# Main function to summarize all files
def summarize_all_files(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    file_summaries = {}

    for file_name, chunks in chunks_data.items():
        print(f"Processing {file_name}...")
        chunk_summaries = []

        for chunk in chunks:
            chunk_text = chunk["text"]
            print(f"  Summarizing chunk {chunk['chunk_id']}...")
            chunk_summary = summarize_chunk(chunk_text)
            chunk_summaries.append(chunk_summary)

        # Combine chunk summaries into a file summary
        print(f"Combining summaries for {file_name}...")
        file_summary = summarize_file(chunk_summaries)
        file_summaries[file_name] = file_summary

    # Save file summaries to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(file_summaries, f, indent=4)

    print(f"Summaries saved to {output_file}")

# Main Execution
if __name__ == "__main__":
    input_file = "indexes/output_chunks.json"      # Input file with chunked text
    output_file = "indexes/file_summaries.json"    # Output file for file-level summaries

    summarize_all_files(input_file, output_file)
