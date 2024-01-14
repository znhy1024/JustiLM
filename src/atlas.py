# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
import json
import time
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src import dist_utils, util
from src.retrievers import EMBEDDINGS_DIM

logger = logging.getLogger(__name__)
IGNORE_INDEX: int = -100
BERT_MAX_SEQ_LENGTH: int = 512


def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


class Atlas(nn.Module):
    def __init__(self, opt, reader, retriever, reader_tokenizer, retriever_tokenizer):
        super(Atlas, self).__init__()

        self.reader = reader
        self.retriever = retriever
        self.reader_tokenizer = reader_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.opt = opt
        if self.opt.mt:
            self.cls_loss = nn.CrossEntropyLoss(reduction='none')

        if self.opt.lclm or self.opt.lglm or self.opt.fc_only or self.opt.lgret or self.opt.lcret:
            self.all_fc = json.load(open(self.opt.fc_file))
            self.mse_loss = torch.nn.MSELoss()

        self.READER_ALL_TOKENS = list(self.reader_tokenizer.vocab.values())

    def _get_fp16_retriever_copy(self):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module
        else:
            retriever_to_copy = self.retriever
        return copy.deepcopy(retriever_to_copy).half().eval()

    @torch.no_grad()
    def build_index(self, index, passages, gpu_embedder_batch_size, logger=None):
        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        retrieverfp16 = self._get_fp16_retriever_copy()

        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            batch = [self.opt.retriever_format.format(**example) for example in batch]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, gpu_embedder_batch_size),
                truncation=True,
            )

            embeddings = retrieverfp16(**_to_cuda(batch_enc), is_passages=True)
            index.embeddings[:, total : total + len(embeddings)] = embeddings.T
            total += len(embeddings)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        dist_utils.barrier()
        logger.info(f"{total} passages encoded on process: {dist_utils.get_rank()}")

        if not index.is_index_trained():
            logger.info(f"Building faiss indices")
            index.train_index()

    @torch.no_grad()
    def _retrieve(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        self.retriever.eval()
        if len(query) > 0:
            query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()  # TODO: broken
        if self.training:
            self.retriever.train()

        search_start = time.time()
        if filtering_fun is not None:
            passages, scores = index.search_knn(query_emb, topk * self.opt.filtering_overretrieve_ratio)
            passages, scores = filtering_fun(batch_metadata, passages, scores, topk, training=self.training)
        else:
            passages, scores = index.search_knn(query_emb, topk)
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb

    @torch.no_grad()
    def retrieve_with_rerank(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
        iter_stats={},
    ):
        bsz = len(query)
        to_rerank = self.opt.n_to_rerank_with_retrieve_with_rerank

        # first, do the retrieval
        passages, _, query_emb = self._retrieve(
            index,
            to_rerank,
            query,
            query_ids_retriever,
            query_mask_retriever,
            batch_metadata,
            filtering_fun,
            iter_stats,
        )

        retrieverfp16 = self._get_fp16_retriever_copy()
        fstr = self.opt.retriever_format
        flat_passage_strings = [fstr.format(**p) for ps in passages for p in ps]
        encoder_batch_size = min(len(flat_passage_strings), self.opt.per_gpu_embedder_batch_size)
        passage_emb, output_passages, output_scores = (
            query_emb.new_zeros(len(flat_passage_strings), query_emb.shape[-1]),
            [],
            [],
        )

        for b in range(0, len(flat_passage_strings), encoder_batch_size):
            batch = flat_passage_strings[b : b + encoder_batch_size]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                truncation=True,
            )
            batch_emb = retrieverfp16(**_to_cuda(batch_enc), is_passages=True).to(query_emb)
            passage_emb[b : b + encoder_batch_size] = batch_emb

        passage_emb = passage_emb.view(bsz, to_rerank, -1)
        retriever_scores = torch.einsum("id, ijd->ij", [query_emb, passage_emb])
        top_retriever_scores, top_retriever_inds = torch.topk(retriever_scores, topk, dim=1)

        for i in range(bsz):
            output_passages.append([passages[i][j] for j in top_retriever_inds[i]])
            output_scores.append(top_retriever_scores[i].tolist())
        return output_passages, output_scores

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        retrieve_func = self.retrieve_with_rerank if self.opt.retrieve_with_rerank else self._retrieve
        passages, scores = retrieve_func(*args, **kwargs)[:2]
        return passages, scores

    def append_query(self, query, passages):
        return [self.opt.encoder_format.format(query=query, **p) for p in passages]

    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:
            query_enc = self.retriever_tokenizer(
                query,
                max_length=min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = _to_cuda(query_enc)
        else:
            query_enc = None
        return _to_cuda(query_enc)

    def reader_tokenize(self, query, target, target_tokens):
        if target_tokens is None:
            if self.opt.decoder_prompt_format is not None:
                modified_query = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
                target = [q + t for (q, t) in zip(modified_query, target)]

                query_mask = self.reader_tokenizer(
                    modified_query,
                    max_length=self.opt.target_maxlength,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["attention_mask"]

            if self.opt.decoder_format is not None:
                target = [self.opt.decoder_format.format(target=t) for t in target]
            target = [t + "</s>" if not t.endswith("</s>") else t for t in target]

            target_tokens = self.reader_tokenizer(
                target,
                max_length=self.opt.target_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

        decoder_input_ids = self.reader._shift_right(target_tokens["input_ids"])
        labels = target_tokens["input_ids"].masked_fill(~target_tokens["attention_mask"].bool(), IGNORE_INDEX)

        # If decoder prompt is not None mask labels such that the model is not trained to predict the prompt
        if self.opt.decoder_prompt_format is not None:
            query_mask = self.reader_tokenizer(
                modified_query,
                max_length=self.opt.target_maxlength,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"]

            padding = torch.zeros(
                (
                    query_mask.size(0),
                    target_tokens["input_ids"].size(-1) - query_mask.size(-1),
                )
            )
            query_mask = torch.cat([query_mask, padding], dim=1)
            labels = labels.masked_fill(query_mask.bool(), IGNORE_INDEX)

        return labels.cuda(), decoder_input_ids.cuda()

    def tokenize(self, query, target, target_tokens):
        if query is None and target is None:
            return None, None, None

        assert (
            target_tokens is None or self.opt.decoder_prompt_format is None
        ), "decoder_prompt_format not compatible with target tokenized in iterator"

        query_enc = self.retriever_tokenize(query) if not self.opt.use_file_passages else None
        labels, decoder_input_ids = self.reader_tokenize(query, target, target_tokens)
        return query_enc, labels, decoder_input_ids

    def tokenize_passages(self, query, passages):
        if len(query) == 0:
            return None, None

        query_passages = [self.append_query(q, p) for q, p in zip(query, passages)]

        fstr = self.opt.retriever_format
        retriever_passages = [[fstr.format(**p) for p in example] for example in passages]
        if self.retriever_tokenizer:
            retriever_tok = encode_passages(
                retriever_passages,
                self.retriever_tokenizer,
                min(self.opt.text_maxlength, BERT_MAX_SEQ_LENGTH),
            )
            retriever_tok = _to_cuda(retriever_tok)
        else:
            retriever_tok = None
        reader_tok = encode_passages(query_passages, self.reader_tokenizer, self.opt.text_maxlength)
        reader_tok = _to_cuda(reader_tok)
        return reader_tok, retriever_tok

    def perplexity_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            self.reader.eval()
            total_context = reader_ids.size(1)
            cfg.n_context = 1
            cfg.bsz = bsz * total_context
            reader_ids_score = reader_ids.view(bsz * total_context, -1)
            reader_mask_score = reader_mask.view(bsz * total_context, -1)
            repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, total_context, dim=0) 
            repeated_labels = torch.repeat_interleave(labels, total_context, dim=0)
            reader_output = self.reader(
                input_ids=reader_ids_score.cuda(),
                attention_mask=reader_mask_score.cuda(),
                decoder_input_ids=repeated_decoder_input_ids,
                labels=repeated_labels,
                use_cache=False,
            )
            token_loss = nn.functional.cross_entropy(
                reader_output.logits.view(-1, reader_output.logits.size(-1)),
                repeated_labels.flatten(),
                reduction="none",
            )
            gold_score = token_loss.view(bsz, total_context, -1)
            z = (repeated_labels.view(bsz, total_context, -1) > -1).sum(dim=-1)
            gold_score = -gold_score.sum(dim=-1) / z 

            return gold_score

    def eval_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz, mask_query):
        self.reader.eval()
        self.reader.reset_score_storage()
        cfg.bsz = reader_ids.size(0)
        cfg.n_context = reader_ids.size(1)
        reader_ids_score = reader_ids.view(reader_ids.size(0), -1)
        reader_mask_score = reader_mask.view(reader_mask.size(0), -1)
        with torch.no_grad():
            reader_output = self.reader(
                input_ids=reader_ids_score,
                attention_mask=reader_mask_score,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            crossattention_scores = self.reader.get_crossattention_scores(
                cfg.n_context,
                reader_mask_score,
                labels=labels,
                ids=reader_ids,
                mode=self.opt.gold_score_mode,
                mask_query=mask_query,
            )
            gold_score = select_crossattention_scores(crossattention_scores, self.opt.gold_score_mode)

            if self.training:
                self.reader.train()
            return gold_score

    def loop_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            total_context = reader_ids.size(1)
            doc_len = reader_ids.size(-1)
            self.reader.eval()
            cfg.bsz = bsz
            cfg.n_context = total_context
            reader_ids_score_eval = reader_ids.view(reader_ids.size(0), -1)
            reader_mask_score_eval = reader_mask.view(reader_mask.size(0), -1)

            # forward pass for calculating and caching the encoder states:
            reader_output_eval = self.reader(
                input_ids=reader_ids_score_eval,
                attention_mask=reader_mask_score_eval,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            eval_hidden_state = reader_output_eval.encoder_last_hidden_state

            # run n_docs - 1 forward passes to calculate pp when leaving a doc out
            gold_scores = []
            for loo_index in range(total_context):
                reader_mask_loo = reader_mask.clone()
                reader_mask_loo[:, loo_index] = False  # mask out this doc
                loo_output_eval = self.reader(
                    encoder_outputs=[eval_hidden_state],
                    attention_mask=reader_mask_loo.view(bsz, (total_context) * doc_len),
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    use_cache=False,
                )
                token_loss = nn.functional.cross_entropy(
                    loo_output_eval.logits.view(-1, loo_output_eval.logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                )
                mean_loss = token_loss.view(bsz, labels.shape[-1]).sum(dim=-1) / (labels > -1).sum(-1)
                gold_scores.append(mean_loss)

            gold_score = torch.stack(gold_scores, dim=1)

            return gold_score

    @torch.no_grad()
    def emdr_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        self.reader.eval()
        cfg.n_context = 1
        cfg.bsz = bsz * self.opt.retriever_n_context
        reader_ids_score = reader_ids.view(bsz * self.opt.retriever_n_context, -1)
        reader_mask_score = reader_mask.view(bsz * self.opt.retriever_n_context, -1)
        repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, self.opt.retriever_n_context, dim=0)
        repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
        reader_output = self.reader(
            input_ids=reader_ids_score.cuda(),
            attention_mask=reader_mask_score.cuda(),
            labels=repeated_labels,
            use_cache=False,
        )
        gold_score = reader_output.logits
        return gold_score
       
    def lgret_score(self, ids, passage_emb):
        fc_passages_embs = torch.Tensor([]).to(passage_emb)
        for id in ids: 
            split_passages = util.split_fc_pas(id, self.all_fc[id])
            _, fc_tokens = self.tokenize_passages(["" for i in range(len(split_passages))], [split_passages])
            fc_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in fc_tokens.items()}
            fc_split_passage_emb = self.retriever(**fc_tokens, is_passages=False).to(passage_emb)
            fc_passage_emb = torch.mean(fc_split_passage_emb,dim=0).unsqueeze(0)
            fc_passages_embs = torch.cat((fc_passages_embs,fc_passage_emb))

        fc_passages_embs = fc_passages_embs.unsqueeze(0).repeat((1,passage_emb.size()[1],1))
        loss = self.mse_loss(fc_passages_embs,passage_emb)
        return loss

    def forward(
        self,
        ids,
        index,
        query,
        target,
        cls_label,
        target_tokens=None,
        passages=None,
        batch_metadata=None,
        filtering_fun=None,
        use_cache=False,
        train_retriever=False,
        iter_stats={},
    ):
        forward_start = time.time()
        bsz = len(query)
        if not self.opt.mt:
            del cls_label

        query_mask_reader = (
            self.reader_tokenizer.batch_encode_plus(
                query,
                max_length=self.opt.text_maxlength,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["attention_mask"]
            .bool()
            .cuda()
        )

        query_enc, labels, decoder_input_ids = self.tokenize(query, target, target_tokens)
        fc_only = self.opt.fc_only

        if (not self.opt.use_file_passages) and (not fc_only):
            retrieve_start = time.time()
            passages, _ = self.retrieve(
                index,
                self.opt.retriever_n_context,
                query,
                query_enc["input_ids"],
                query_enc["attention_mask"],
                batch_metadata=batch_metadata,
                filtering_fun=filtering_fun,
                iter_stats=iter_stats,
            )
            iter_stats["runtime/retrieve"] = (time.time() - retrieve_start, 1)

        if fc_only:
            search_start = time.time()
            passage_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])
                tmp_text = [" ".join(["[MASK]" for _ in range(100)])]
                tmp_text_split = [" ".join(tmp_text[i : i + 100]).strip() for i in range(0, len(tmp_text), 100)]
                padding_pas = {'id':-1,'title':"","text":tmp_text_split[0]}
                split_passages = split_passages + [padding_pas for p in range(self.opt.retriever_n_context)]
                passage_tmp.append(split_passages[:self.opt.retriever_n_context])
            passages = passage_tmp
            iter_stats["runtime/match"] = (time.time() - search_start, 1)
        
        reader_tokens, retriever_tokens = self.tokenize_passages(query, passages)
        reader_ids = reader_tokens["input_ids"]  # FIXME
        reader_mask = reader_tokens["attention_mask"].bool()
        n_context_training = min(self.opt.n_context, reader_ids.size(1))

        cfg = self.reader.encoder.config

        retriever_loss = None
        if train_retriever:

            if self.opt.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_enable()

            query_emb = self.retriever(**query_enc, is_passages=False)

            if "std" in self.opt.gold_score_mode:
                retriever_tokens = {k: v[:, :n_context_training] for k, v in retriever_tokens.items()}
            retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}

            passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb) 
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
            retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])

            loss_lgret = None
            if self.opt.lgret:
                loss_lgret= self.lgret_score(ids,passage_emb)
           
            if "eval" in self.opt.gold_score_mode:
                gold_score = self.eval_score(
                    reader_ids,
                    reader_mask,
                    decoder_input_ids,
                    labels,
                    cfg,
                    bsz,
                    query_mask_reader,
                )
            elif "loop" in self.opt.gold_score_mode:
                gold_score = self.loop_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)
            elif "ppmean" in self.opt.gold_score_mode:
                gold_score = self.perplexity_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)
            elif "emdr" in self.opt.gold_score_mode:
                gold_score = self.emdr_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)
                        
            loss_lcret = None 
            if self.opt.lcret:
                tmp_id = ids[0]
                split_text = util.split_fc_text(self.all_fc[tmp_id])
                all_fc_score = None
                
                for st in split_text:
                    fc_enc, fc_labels, fc_decoder_input_ids = self.tokenize([st], [st], target_tokens)
                    fc_score = self.perplexity_score(reader_ids, reader_mask, fc_decoder_input_ids, fc_labels, cfg, bsz) 
                    if all_fc_score is None:
                        all_fc_score = fc_score
                    else:
                        all_fc_score = torch.cat((all_fc_score,fc_score))

                fc_enc, _, _ = self.tokenize(split_text, [st], target_tokens)
                fc_embs = self.retriever(**fc_enc, is_passages=False)
                fc_retriever_score = torch.mm(fc_embs,passage_emb.squeeze(0).transpose(0,1))
                topk_v,topk_idx = fc_retriever_score.topk(1)
                fc_retriever_score = topk_v.transpose(0,1)
                topk_all_fc_score = torch.gather(all_fc_score,dim=1,index=topk_idx).transpose(0,1)
                all_fc_score = topk_all_fc_score
                fc_retriever_score = fc_retriever_score / np.sqrt(query_emb.size(-1))
                fc_retriever_score = fc_retriever_score.float()
                loss_lcret =self.kldivloss(fc_retriever_score, all_fc_score)

            if self.opt.use_gradient_checkpoint_retriever:
                self.retriever.gradient_checkpointing_disable()

            self.reader.reset_score_storage()

            if self.training:
                self.reader.train()

        cfg.bsz = reader_ids.size(0)
        cfg.n_context = n_context_training

        reader_ids_training = reader_ids[:, :n_context_training].contiguous()
        reader_mask_training = reader_mask[:, :n_context_training].contiguous()

        reader_ids_training = reader_ids_training.view(reader_ids.size(0), -1)
        reader_mask_training = reader_mask_training.view(reader_mask.size(0), -1)

        if self.opt.use_gradient_checkpoint_reader:
            self.reader.gradient_checkpointing_enable()
        
        reader_output = self.reader(
            input_ids=reader_ids_training,
            attention_mask=reader_mask_training,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
        )
        reader_loss = reader_output[0]

        if self.opt.lclm:
            docs_crossattention_scores = self.reader.get_crossattention_scores_lm(
                n_context_training,
                reader_mask_training.cuda(),
                ids=reader_ids_training.cuda(),
                mask_query=query_mask_reader.cuda(),
                labels=labels,
                mode="all",
            )
            docs_gold_scores = select_crossattention_scores(docs_crossattention_scores, "attn")        
        
        loss_lclm = None
        if self.opt.lclm:
            passage_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])        
                passage_tmp.append(split_passages[:self.opt.retriever_n_context])
                split_text = util.split_fc_text(self.all_fc[id])

            fc_tokens, _ = self.tokenize_passages(query, passage_tmp)
            fc_ids = fc_tokens["input_ids"]  # FIXME
            fc_mask = fc_tokens["attention_mask"].bool()

            n_context_training = min(self.opt.n_context, fc_ids.size(1))
            fc_ids_training = fc_ids[:, :n_context_training].contiguous()
            fc_mask_training = fc_mask[:, :n_context_training].contiguous()

            fc_ids_training = fc_ids_training.view(fc_ids.size(0), -1)
            fc_mask_training = fc_mask_training.view(fc_mask.size(0), -1)

            fc_gold_score = self.eval_score(
                    fc_ids,
                    fc_mask,
                    decoder_input_ids,
                    labels,
                    cfg,
                    bsz,
                    query_mask_reader,
                )
            
            with torch.no_grad():
                fc_enc, _, _ = self.tokenize(split_text, [""], target_tokens)
                fc_embs = self.retriever(**fc_enc, is_passages=False)
                fc_retriever_score = torch.mm(fc_embs,passage_emb.squeeze(0).transpose(0,1))
                fc_retriever_score = fc_retriever_score.transpose(0,1)
                topk_v,topk_idx = fc_retriever_score.topk(1)

            all_fc_score = torch.zeros_like(docs_gold_scores)        
            for ii,idx in enumerate(topk_idx):
                all_fc_score[0,ii] = fc_gold_score[0][idx.item()]     
            loss_lclm = self.kldivloss(docs_gold_scores, all_fc_score)

        mt_loss= None
        if self.opt.mt:
            choices_ids = torch.stack([
                self.reader_tokenizer(
                    answer_choice, return_tensors="pt", truncation=True, add_special_tokens=False
                ).input_ids.squeeze(0)
                for answer_choice in ["false","unclear","true"]
            ])
            flat_choices_ids = choices_ids.flatten(0, 1).unsqueeze(0)
            decoder_choices_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.reader_tokenizer.pad_token_id).long()
            lm_target = lm_target.to(reader_ids_training.device)

            model_output = self.reader(
                input_ids=reader_ids_training,
                attention_mask=reader_mask_training,
                decoder_input_ids=decoder_choices_ids.to(reader_ids_training.device),
                labels=lm_target,
                use_cache=False,
            )
            choices_scores = (
                self.cls_loss(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1))
                .view(1, 3, -1)
                .sum(dim=-1)
            )
            cls_label = torch.LongTensor(cls_label).to(choices_scores.device)
            mt_loss = self.cls_loss(-choices_scores, cls_label).mean(dim=-1)          

        loss_lglm = None
        if self.opt.lglm:
            reader_logits = reader_output.logits
            passage_tmp = []
            for idx,id in enumerate(ids):
                split_passages = util.split_fc_pas(id, self.all_fc[id])        
                passage_tmp.append(split_passages[:self.opt.retriever_n_context])

            fc_tokens, _ = self.tokenize_passages(query, passage_tmp)
            fc_ids = fc_tokens["input_ids"]  # FIXME
            fc_mask = fc_tokens["attention_mask"].bool()
            n_context_training = min(self.opt.n_context, fc_ids.size(1))
            fc_ids_training = fc_ids[:, :n_context_training].contiguous()
            fc_mask_training = fc_mask[:, :n_context_training].contiguous()

            fc_ids_training = fc_ids_training.view(fc_ids.size(0), -1)
            fc_mask_training = fc_mask_training.view(fc_mask.size(0), -1)

            fc_output = self.reader(
                input_ids=fc_ids_training,
                attention_mask=fc_mask_training,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                use_cache=False,
            )
            fc_logits = fc_output.logits
            loss_lglm = self.mse_loss(reader_logits.squeeze(0), fc_logits.squeeze(0))

        if self.opt.use_gradient_checkpoint_reader:
            self.reader.gradient_checkpointing_disable()

        if train_retriever:
            if self.opt.compute_crossattention_stats or "std" in self.opt.gold_score_mode:
                crossattention_scores = self.reader.get_crossattention_scores(
                    n_context_training,
                    reader_mask_training.cuda(),
                    ids=reader_ids_training.cuda(),
                    mask_query=query_mask_reader.cuda(),
                    labels=labels,
                    mode="all",
                )
            if "std" in self.opt.gold_score_mode:
                gold_score = select_crossattention_scores(
                    crossattention_scores, self.opt.gold_score_mode
                ).detach() 

            retriever_score = retriever_score / np.sqrt(query_emb.size(-1))

            if self.opt.compute_crossattention_stats:
                with torch.no_grad():
                    for k, v in crossattention_scores.items():
                        corr = torch.corrcoef(torch.stack([gold_score.view(-1), v.view(-1)]))
                        corr = corr[0, 1].item()
                        if np.isnan(corr):
                            corr = 0.0
                        iter_stats[f"corr/{k}"] = (corr, len(query))

            if gold_score is not None:
                gold_score = gold_score.float()
                retriever_score = retriever_score.float()
                if self.opt.gold_score_mode == "emdr":
                    retriever_loss = self.logprob(retriever_score, gold_score, labels)
                else:
                    retriever_loss = self.kldivloss(retriever_score, gold_score)

        self.reader.reset_score_storage()
        if self.opt.lglm:
            loss_lglm = self.opt.lglm_cof*loss_lglm
            reader_loss = reader_loss + loss_lglm
            iter_stats["loss/reader_loss_lglm"] = (loss_lglm.item(), len(query))
        if self.opt.lclm:
            loss_lclm = self.opt.lclm_cof*loss_lclm
            reader_loss = reader_loss + loss_lclm
            iter_stats["loss/reader_loss_lclm"] = (loss_lclm.item(), len(query))
        
        if self.opt.mt:
            mt_loss = self.opt.mt_cof*mt_loss
            reader_loss = (1-self.opt.mt_cof)*reader_loss + mt_loss
            iter_stats["loss/reader_mt_loss"] = (mt_loss.item(), len(query))
        
        
        iter_stats["loss/reader_loss"] = (reader_loss.item(), len(query))

        if retriever_loss is not None:
            if loss_lgret is not None:
                retriever_loss = retriever_loss + loss_lgret
                iter_stats["loss/loss_lgret"] = (loss_lgret.item(), len(query))
            if loss_lcret is not None:
                retriever_loss = retriever_loss + loss_lcret
                iter_stats["loss/loss_lcret"] = (loss_lcret.item(), len(query))

            iter_stats["loss/retriever_loss"] = (retriever_loss.item(), len(query))

        iter_stats["runtime/forward"] = (time.time() - forward_start, 1)
        return reader_loss, retriever_loss

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)

    def logprob(self, score, gold_score, labels):
        with torch.no_grad():
            repeated_labels = torch.repeat_interleave(labels, self.opt.retriever_n_context, dim=0)
            repeated_labels[repeated_labels == IGNORE_INDEX] = 0

            mask_labels = labels >= 0

            gold_log_prob = torch.nn.functional.log_softmax(gold_score / self.opt.temperature_gold, dim=-1)
            gold_log_probs = torch.gather(gold_log_prob, dim=-1, index=repeated_labels[..., None]).view(
                gold_log_prob.size(0), -1
            )
            gold_log_probs = gold_log_probs.view(score.size(0), score.size(1), -1)

        log_score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
        log_prob = gold_log_probs + log_score[..., None]
        logsumprobs = torch.logsumexp(log_prob, dim=1)
        loss = -1 * torch.sum(logsumprobs * mask_labels) / torch.sum(mask_labels)

        return loss

    @torch.no_grad()
    def compute_reader_loss_and_logits(self, tokens, decoder_input_ids, labels):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        reader_loss = self.reader(
            input_ids=tokens["input_ids"].cuda().view(tokens["input_ids"].size(0), -1),
            attention_mask=tokens["attention_mask"].cuda().view(tokens["attention_mask"].size(0), -1),
            decoder_input_ids=decoder_input_ids.cuda(),
            labels=labels.cuda(),
            use_cache=False,
        )
        return reader_loss[0].cpu().item(), reader_loss[1]

    @torch.no_grad()
    def generate(self, tokens, query, cls_label,choices=None):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()}

        bos_token_id = None

        prefix_allowed_tokens_fn = None
        if self.opt.decoder_prompt_format is not None:
            prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

        outputs = self.reader.generate(
            input_ids=tokens["input_ids"].cuda(),
            attention_mask=tokens["attention_mask"].cuda(),
            num_return_sequences=1,
            max_length=self.opt.generation_max_length,
            min_length=self.opt.generation_min_length,
            num_beams=self.opt.generation_num_beams,
            length_penalty=self.opt.generation_length_penalty,
            forced_bos_token_id=bos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        prediction = None
        if self.opt.mt:
            choices_ids = torch.stack([
                self.reader_tokenizer(
                    answer_choice, return_tensors="pt", truncation=True, add_special_tokens=False
                ).input_ids.squeeze(0)
                for answer_choice in ["false","unclear","true"]
            ])
            flat_choices_ids = choices_ids.flatten(0, 1).unsqueeze(0)
            decoder_choices_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.reader_tokenizer.pad_token_id).long()
            lm_target = lm_target.to(tokens["input_ids"].cuda())

            model_output = self.reader(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                decoder_input_ids=decoder_choices_ids.to(tokens["input_ids"].cuda()),
                labels=lm_target,
                use_cache=False,
            )
            choices_scores = (
                self.cls_loss(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1))
                .view(1, 3, -1)
                .sum(dim=-1)
            )
            _, prediction = choices_scores.min(dim=1)

        return outputs,prediction,cls_label

    def get_prefix_allowed_tokens_fn(self, prefix_str: Optional[str] = None):
        if prefix_str:
            prefix_tokens_ids = self.reader_tokenizer.batch_encode_plus(prefix_str, add_special_tokens=False)[
                "input_ids"
            ]

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                if input_ids.shape[-1] > len(prefix_tokens_ids[batch_id]):
                    return self.READER_ALL_TOKENS

                return prefix_tokens_ids[batch_id][input_ids.shape[-1] - 1]

        else:
            prefix_allowed_tokens_fn = None

        return prefix_allowed_tokens_fn


def select_crossattention_scores(scores, mode):
    if "eval" in mode or "attn" in mode or "ppmean" in mode :
        return scores["normssum"]
    elif "std" in mode:
        return scores[mode[len("std") :]]


def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}
