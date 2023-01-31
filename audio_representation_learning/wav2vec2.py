import json
import re

import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2Processor, AutoModelForCTC
from transformers import TrainingArguments, Trainer

from collator import DataCollatorCTCWithPadding

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\!\¿\¡\!\?]'
batch_size = 24


def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def create_vocab(train_dataset, val_dataset):
    vocab_train = train_dataset.map(
        extract_all_chars, batched=True,
        batch_size=-1, keep_in_memory=True,
        remove_columns=train_dataset.column_names
    )
    vocab_val = val_dataset.map(
        extract_all_chars, batched=True,
        batch_size=-1, keep_in_memory=True,
        remove_columns=val_dataset.column_names
    )

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_val["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict['|'] = vocab_dict[' ']
    del vocab_dict[' ']

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == '__main__':
    dataset = load_dataset('facebook/multilingual_librispeech', 'spanish', 'train')
    train_dataset = dataset['train.9h']
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(remove_special_characters)
    test_dataset = test_dataset.map(remove_special_characters)

    create_vocab(test_dataset, train_dataset)

    model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
    config = AutoConfig.from_pretrained(model_checkpoint)

    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None

    tokenizer = AutoTokenizer.from_pretrained(
        "./",
        config=config,
        tokenizer_type=tokenizer_type,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    model_checkpoint_name = model_checkpoint.split("/")[-1]
    repo_name = f"{model_checkpoint_name}-spanish-mls"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load_metric("wer")

    model = AutoModelForCTC.from_pretrained(
        model_checkpoint,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    if hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
