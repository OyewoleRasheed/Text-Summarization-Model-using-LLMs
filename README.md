# Text Summarization with T5 Transformer

This project demonstrates how to build an **abstractive text summarization** system using a pre-trained transformer model (T5) from Hugging Face. The work replicates and expands upon [Aman Kharwal’s blog post](https://amanxai.com/2024/10/07/text-summarization-model-using-llms/), exploring how large language models can be fine-tuned or directly applied to generate high-quality summaries from long-form text.

---

##  What is Text Summarization?

Text summarization is the task of shortening a piece of text while preserving its key information and meaning. There are two main approaches:

* **Extractive summarization**: selects key sentences from the original text.
* **Abstractive summarization**: generates novel sentences that capture the essence of the original text, more like how humans summarize.

This project uses **abstractive summarization** with a model from the **T5 (Text-to-Text Transfer Transformer)** family.

---

## 🚀 Key Features

* Implements **T5 model** (`t5-small`, customizable)
* Uses Hugging Face's `transformers` library
* Supports **beam search**, **length penalty**, and **early stopping** to improve summary quality
* Easily configurable for different model sizes and summary lengths
* Designed for experimentation and learning

## 🛠️ How It Works

### Step 1: Load Pre-trained T5 Model

The script loads a pre-trained model and tokenizer from Hugging Face:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
```

You can easily swap out `t5-small` for `t5-base` or `t5-large` for more accuracy at the cost of compute.

---

### Step 2: Prepare the Input

T5 requires task-specific prefixes. For summarization, we prepend:

```python
input_text = "summarize: " + raw_text
```

Then tokenize it:

```python
input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
```

---

### Step 3: Generate the Summary

The model generates output using beam search and other decoding options:

```python
summary_ids = model.generate(
    input_ids,
    max_length=50,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)
```

These settings help balance summary quality, brevity, and diversity.

---

### Step 4: Decode the Output

Convert token IDs back to natural language:

```python
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

The result is a coherent summary that captures the core meaning of the original input.

## ✨ Future Improvements

* Add evaluation scripts with ROUGE/BERTScore
* Support long-document chunking and summarization pipelines
* Build a Streamlit or Gradio web UI
* Try other models like BART, Pegasus, or Falcon


## 📚 References

* [Aman Kharwal’s Blog](https://amanxai.com/2024/10/07/text-summarization-model-using-llms/)
* [T5 Paper – Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
* [Hugging Face Transformers](https://huggingface.co/transformers/)

