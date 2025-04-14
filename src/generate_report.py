from transformers import pipeline

def generate_summary(text):
    """
    Generates a summary for the simulation report.
    """
    summarizer = pipeline("text-generation", model="gpt2")
    summary = summarizer(text, max_length=150, do_sample=True)[0]['generated_text']
    return summary
