# English-Egyptian Translation Project

## Overview

The English-Egyptian Translation Project aims to develop a translation model that can convert text between English and Egyptian Arabic. This project utilizes the M2M100 model from Hugging Face's Transformers library and evaluates its performance using BLEU scores. The project is designed to handle real-world text translation tasks with a focus on Egyptian Arabic.

## Features

- **Bidirectional Translation:** Translate text from English to Egyptian Arabic and vice versa.
- **Pre-trained Model:** Utilizes the M2M100 model for translation.
- **Performance Evaluation:** Uses BLEU score to evaluate translation accuracy.
- **Data Preprocessing:** Handles text length filtering and tokenization.

## Installation

### Prerequisites

- Python 3.x
- Pip (Python package installer)
- Google Colab or Jupyter Notebook (for running the provided code)

### Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/english-egyptian-translation.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd english-egyptian-translation
    ```

3. **Install Required Packages:**

    ```bash
    pip install sacrebleu transformers datasets accelerate transformers[torch]
    ```

## Usage

1. **Data Preparation:**

    - Load your dataset and filter texts based on length constraints.
    - Example code for loading and filtering data:

    ```python
    import pandas as pd
    import numpy as np

    dataset = pd.read_excel("DB.xlsx")
    # Filter texts based on length
    max_len_ArText = 155
    max_len_EnText = 211
    final_Arabic_text = [text for text in dataset['egyption_Text'] if len(text) <= max_len_ArText]
    final_English_text = [text for text in dataset['english_Text'] if len(text) <= max_len_EnText]
    ```

2. **Model Training:**

    - Load and fine-tune the M2M100 model.
    - Example code for training the model:

    ```python
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, TrainingArguments, Trainer

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    # Prepare datasets
    train_dataset = preprocess_data(train_df, max_seq_length)
    eval_dataset = preprocess_data(eval_df, max_seq_length)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=7
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    ```

3. **Model Evaluation:**

    - Evaluate the model using BLEU score.
    - Example code for evaluation:

    ```python
    from nltk.translate.bleu_score import corpus_bleu

    references = []
    candidates = []

    for example in eval_dataset:
        reference = tokenizer.decode(example['labels'], skip_special_tokens=True)
        references.append([reference])
        input_ids = example['input_ids'].unsqueeze(0).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids)
        predicted = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidates.append(predicted)

    bleu_score = corpus_bleu(references, candidates)
    print("BLEU score:", bleu_score)
    ```

4. **Translation Example:**

    - Translate example sentences and compare with actual translations.
    - Example code for translation:

    ```python
    from transformers import pipeline

    pipe = pipeline("translation", model=model, tokenizer=tokenizer)

    for idx, text in enumerate(final_English_text[:10]):
        translation_result = pipe(text, src_lang="en", tgt_lang="ar")
        translated_text = translation_result[0]['translation_text']
        print(f"{idx+1}. English: {text}\nTranslation to Arabic: {translated_text}\nEgyptian: {final_Arabic_text[idx]}\n{'*'*60}")
    ```


## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.


## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers) for providing the M2M100 model.
- [Google Colab](https://colab.research.google.com/) for the cloud-based development environment.

## Contact

For any questions or feedback, please contact me at [mahmoudsalamacs@gmail.com].

