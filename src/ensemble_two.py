# -*- coding: utf-8 -*-

import argparse
import glob
import json
import logging
from pathlib import Path
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from .model import (
    BertGetPoolFeatures,
    XLNetGetPoolFeature,
    RobertaGetPoolFeature,
    Ensemble,
    EnsembleAll
)

from .data_utils import glue_compute_metrics as compute_metrics
from transformers.data.processors import glue_convert_examples_to_features as convert_examples_to_features
from .data_utils import output_modes 
from .data_utils import processors 
from .single_task import set_seed

logger = logging.getLogger("ensemble_two")

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            RobertaConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertGetPoolFeatures, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetGetPoolFeature, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaGetPoolFeature, RobertaTokenizer),
}


def train(args, train_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.warmup_ratio:
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * t_total)
    else:
        warmup_steps = args.warmup_steps

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs1 = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[2] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs2 = {
                "input_ids": batch[4], "attention_mask": batch[5], "labels": batch[7]}
            if args.model2_type != "distilbert":
                inputs2["token_type_ids"] = (
                    batch[6] if args.model2_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            assert batch[3].equal(
                batch[7]), "training data labels are not the same {} vs {}".format(batch[3], batch[7])

            outputs = model(inputs1, inputs2, batch[3])
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def train_all(args, train_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.warmup_ratio:
        warmup_steps = np.dtype('int64').type(args.warmup_ratio * t_total)
    else:
        warmup_steps = args.warmup_steps

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs1 = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[2] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            inputs2 = {
                "input_ids": batch[4], "attention_mask": batch[5], "labels": batch[7]}
            if args.model2_type != "distilbert":
                inputs2["token_type_ids"] = (
                    batch[6] if args.model2_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            inputs3 = {
                "input_ids": batch[8], "attention_mask": batch[9], "labels": batch[11]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[10] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            assert batch[3].equal(
                batch[7]), "training data labels are not the same {} vs {}".format(batch[3], batch[7])

            assert batch[3].equal(
                batch[11]), "input1 and input3 labels are not the same {} vs {}".format(batch[3], batch[11])

            outputs = model(inputs1, inputs2, inputs3, batch[3])
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, prefix=""):
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # use SequentialSampler here
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs1 = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[2] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            inputs2 = {
                "input_ids": batch[4], "attention_mask": batch[5], "labels": batch[7]}
            if args.model2_type != "distilbert":
                inputs2["token_type_ids"] = (
                    batch[6] if args.model2_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            assert batch[3].equal(
                batch[7]), "training data labels are not the same {} vs {}".format(batch[3], batch[7])

            outputs = model(inputs1, inputs2, batch[3])
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs1["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs1["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)
    # make the prefix dir
    Path(os.path.join(eval_output_dir)).mkdir(exist_ok=True)
    output_eval_file = os.path.join(
        eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("hyperparameter: batch %s, epoch %s\n" %
                     (args.per_gpu_train_batch_size, args.num_train_epochs))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info(" %s = %s", "eval_loss", str(eval_loss))
        writer.write("%s = %s\n" % ("eval_loss", str(eval_loss)))

    return results


def evaluate_all(args, eval_dataset, model, prefix=""):
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # use SequentialSampler here
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs1 = {
                "input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[2] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            inputs2 = {
                "input_ids": batch[4], "attention_mask": batch[5], "labels": batch[7]}
            if args.model2_type != "distilbert":
                inputs2["token_type_ids"] = (
                    batch[6] if args.model2_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            inputs3 = {
                "input_ids": batch[8], "attention_mask": batch[9], "labels": batch[11]}
            if args.model1_type != "distilbert":
                inputs1["token_type_ids"] = (
                    batch[10] if args.model1_type in [
                        "bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            assert batch[3].equal(
                batch[7]), "training data labels are not the same {} vs {}".format(batch[3], batch[7])

            assert batch[3].equal(
                batch[11]), "input1 and input3 labels are not the same {} vs {}".format(batch[3], batch[11])

            outputs = model(inputs1, inputs2, inputs3, batch[3])
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs1["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs1["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)
    # make the prefix dir
    Path(os.path.join(eval_output_dir)).mkdir(exist_ok=True)
    output_eval_file = os.path.join(
        eval_output_dir, "eval_results.txt")

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        writer.write("hyperparameter: batch %s, epoch %s\n" %
                     (args.per_gpu_train_batch_size, args.num_train_epochs))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info(" %s = %s", "eval_loss", str(eval_loss))
        writer.write("%s = %s\n" % ("eval_loss", str(eval_loss)))

    return results


def load_and_cache_examples(args, model_type, task, tokenizer, evaluate=False, predict=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            "pred" if predict else "dev" if evaluate else "train",
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_pred_examples(args.data_dir) if predict else
            processor.get_dev_examples(
                args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            # pad on the left for xlnet
            pad_on_left=bool(model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids(
                [tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float)

    dataset = (
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model1_type",
        default=None,
        type=str,
        required=True,
        help="First model's type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model2_type",
        default=None,
        type=str,
        required=True,
        help="Second model's type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model1_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--model2_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--ensemble_pretrained_model",
        type=str
    )
    parser.add_argument(
        "--config1_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer1_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--config2_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer2_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--all_ensemble", action="store_true",
                        help="ensemble all three model or not")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pred", action="store_true",
                        help="Whether to do prediction on test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.1,
                        type=float, help="Linear warmup ratio.")
    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port", type=str,
                        default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model1_type = args.model1_type.lower()
    args.model2_type = args.model2_type.lower()
    config_class1, model_class1, tokenizer_class1 = MODEL_CLASSES[args.model1_type]
    config_class2, model_class2, tokenizer_class2 = MODEL_CLASSES[args.model2_type]

    if args.ensemble_pretrained_model:
        config1 = config_class1.from_pretrained(
            args.config1_name if args.config1_name else args.model1_name_or_path,
            num_labels=num_labels,
        )
        tokenizer1 = tokenizer_class1.from_pretrained(
            args.tokenizer1_name if args.tokenizer1_name else args.model1_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        config2 = config_class2.from_pretrained(
            args.config2_name if args.config2_name else args.model2_name_or_path,
            num_labels=num_labels,
        )
        tokenizer2 = tokenizer_class2.from_pretrained(
            args.tokenizer2_name if args.tokenizer2_name else args.model2_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        model = torch.load(args.ensemble_pretrained_model)
    elif args.model1_name_or_path and args.model2_name_or_path:
        config1 = config_class1.from_pretrained(
            args.config1_name if args.config1_name else args.model1_name_or_path,
            num_labels=num_labels,
        )
        tokenizer1 = tokenizer_class1.from_pretrained(
            args.tokenizer1_name if args.tokenizer1_name else args.model1_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        config2 = config_class2.from_pretrained(
            args.config2_name if args.config2_name else args.model2_name_or_path,
            num_labels=num_labels,
        )
        tokenizer2 = tokenizer_class2.from_pretrained(
            args.tokenizer2_name if args.tokenizer2_name else args.model2_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        model1 = model_class1.from_pretrained(
            args.model1_name_or_path,
            config=config1,
        )
        model2 = model_class2.from_pretrained(
            args.model2_name_or_path,
            config=config2,
        )
        # init ensemble model
        model = Ensemble(args, config1, config2,
                         model1, model2, num_labels)

    else:
        raise ValueError("No model detected!")

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset1 = load_and_cache_examples(
            args, args.model1_type, args.task_name, tokenizer1, evaluate=False)
        train_dataset2 = load_and_cache_examples(
            args, args.model2_type, args.task_name, tokenizer2, evaluate=False)

        train_dataset = TensorDataset(*train_dataset1, *train_dataset2)
        global_step, tr_loss = train(
            args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

        # save model
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        torch.save(model_to_save, args.output_dir + "/pytorch_model.bin")

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_eval and args.do_train:
        eval_dataset1 = load_and_cache_examples(
            args, args.model1_type, args.task_name, tokenizer1, evaluate=True)
        eval_dataset2 = load_and_cache_examples(
            args, args.model2_type, args.task_name, tokenizer2, evaluate=True)
        eval_dataset = TensorDataset(*eval_dataset1, *eval_dataset2)
        model = torch.load(args.output_dir + "/pytorch_model.bin")
        model.to(args.device)
        prefix = args.data_dir.split("/")[-1]
        result = evaluate(args, eval_dataset, model, prefix=prefix)
    if args.do_pred:
        pred_dataset1 = load_and_cache_examples(
            args, args.model1_type, args.task_name, tokenizer1, predict=True)
        pred_dataset2 = load_and_cache_examples(
            args, args.model2_type, args.task_name, tokenizer2, predict=True)
        pred_dataset = TensorDataset(*pred_dataset1, *pred_dataset2)
        # model = torch.load(args.output_dir + "/pytorch_model.bin")
        model.to(args.device)
        prefix = "clinical-prediction"
        result = evaluate(args, pred_dataset, model, prefix=prefix)


if __name__ == "__main__":
    main()
