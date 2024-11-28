import gradio as gr
import json
from query import query_pipeline

# File paths
EMBEDDINGS_FILE = "indexes/indexed_embeddings.json"
SUMMARIES_FILE = "indexes/file_summaries.json"

# Query pipeline function for Gradio
def gradio_query_pipeline(query):
    try:
        # Call the query pipeline
        response = query_pipeline(query, EMBEDDINGS_FILE, SUMMARIES_FILE)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Knowledge Graph Query App")
    gr.Markdown(
        "Ask any question related to the knowledge graph or file summaries. The app will provide a detailed response."
    )

    # Input and output components
    query_input = gr.Textbox(
        lines=2,
        placeholder="Enter your query here...",
        label="Your Query",
    )
    output_box = gr.Textbox(
        lines=10,
        label="Response",
        placeholder="The response will appear here after processing your query.",
    )

    # Submit button
    submit_button = gr.Button("Submit")

    # Define interaction
    submit_button.click(gradio_query_pipeline, inputs=query_input, outputs=output_box)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
