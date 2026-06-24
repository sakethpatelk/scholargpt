"""
Generates a 4-page sample PDF about the Transformer architecture.
Uses PyMuPDF (already in requirements.txt) — no extra dependencies.

Usage:
    python scripts/create_sample_pdf.py
"""
import fitz
from pathlib import Path

OUT = Path("data/uploaded/transformer_sample.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

PAGES = [
    {
        "title": "Attention Is All You Need (Sample)",
        "body": (
            "Abstract\n\n"
            "We propose the Transformer, a novel network architecture based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "Experiments on two machine translation tasks show these models to be superior "
            "in quality while being more parallelizable and requiring significantly less "
            "time to train.\n\n"
            "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation "
            "task, outperforming the best previously reported results by more than 2 BLEU. "
            "On the WMT 2014 English-to-French task, our model establishes a new "
            "single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days "
            "on eight GPUs."
        ),
    },
    {
        "title": "Introduction",
        "body": (
            "Recurrent neural networks, long short-term memory (LSTM) and gated recurrent "
            "neural networks (GRU) have been firmly established as state-of-the-art "
            "approaches in sequence modeling and transduction problems such as language "
            "modeling and machine translation.\n\n"
            "The Transformer architecture relies entirely on an attention mechanism to draw "
            "global dependencies between input and output, allowing for significantly more "
            "parallelization. After training for as little as twelve hours on eight P100 "
            "GPUs, the Transformer achieves a new state of the art in translation quality.\n\n"
            "The key innovation is replacing recurrence with self-attention, which attends "
            "to all positions simultaneously rather than sequentially."
        ),
    },
    {
        "title": "Model Architecture",
        "body": (
            "The Transformer uses stacked self-attention and point-wise, fully connected "
            "layers for both the encoder and decoder, each consisting of N=6 identical layers.\n\n"
            "Multi-Head Attention allows the model to jointly attend to information from "
            "different representation subspaces at different positions:\n"
            "  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O\n"
            "  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)\n\n"
            "Scaled Dot-Product Attention:\n"
            "  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V\n\n"
            "Key hyperparameters: d_model=512, h=8 attention heads, d_k=d_v=64, "
            "d_ff=2048, dropout=0.1. Positional encodings using sine and cosine "
            "functions are added to the input embeddings to inject sequence order."
        ),
    },
    {
        "title": "Results, Limitations, and Future Work",
        "body": (
            "Results\n\n"
            "The big Transformer model achieves 28.4 BLEU on English-to-German, surpassing "
            "the previous best ensemble by more than 2 BLEU. On English-to-French, it "
            "achieves 41.0 BLEU, outperforming all previous single models at less than "
            "one-quarter the training cost of the prior best.\n\n"
            "Limitations\n\n"
            "The self-attention mechanism requires O(n^2) memory with respect to sequence "
            "length n. Very long sequences (e.g., high-resolution images as pixels) may be "
            "prohibitively expensive. The model also lacks an inherent notion of sequence "
            "order without positional encodings.\n\n"
            "Future Work\n\n"
            "We plan to extend the Transformer to problems involving input and output "
            "modalities other than text, including images, audio, and video. We also intend "
            "to investigate local, restricted attention mechanisms for handling large inputs "
            "efficiently and to apply the model to other tasks such as parsing."
        ),
    },
]


def create_pdf() -> None:
    doc = fitz.open()

    for page_data in PAGES:
        page = doc.new_page(width=595, height=842)  # A4

        # Title
        page.insert_text(
            (50, 70),
            page_data["title"],
            fontsize=15,
            fontname="helv",
        )

        # Horizontal rule (drawn as a line)
        page.draw_line((50, 85), (545, 85))

        # Body — insert_textbox handles wrapping
        rect = fitz.Rect(50, 100, 545, 800)
        page.insert_textbox(
            rect,
            page_data["body"],
            fontsize=11,
            fontname="helv",
            lineheight=1.5,
        )

    doc.save(str(OUT))
    doc.close()
    print(f"Created: {OUT}  ({len(PAGES)} pages)")


if __name__ == "__main__":
    create_pdf()
