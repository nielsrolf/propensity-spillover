"""Elicitation training on Tinker. Currently: LoRA SFT.

`train_sft` returns (sampler_weights_path, train_tokens); the token count is
real (tokenizer-encoded) and is also added to the CostLedger under the `train`
stage, so the measured cost of an elicitation run is train + sampling + judging.

Add `train_dpo(...)` / `train_grpo(...)` here with the SAME contract:

    def train_dpo(model, pairs, *, ledger, ...) -> TrainResult
    def train_grpo(model, prompts, *, ledger, ...) -> TrainResult

so harness.experiment can call any of them and cost accounting stays uniform.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .cost import CostLedger
from .sampling import (BASE_COMPLETION_TMPL, BASE_STOP, CHAT_STOP, is_base_model,
                       render_base_prompt, strip_base_stop, strip_chat_stop)


@dataclass
class TrainResult:
    sampler_path: str
    train_tokens: int
    usd: float
    losses: List[float] = field(default_factory=list)


def _render_pair(tokenizer, model: str, question: str, answer: str, system_prompt=None):
    """(prompt_ids, completion_ids) for one training example, matching sampling format."""
    if is_base_model(model):
        prompt_ids = tokenizer.encode(render_base_prompt(question))
        completion_ids = tokenizer.encode(BASE_COMPLETION_TMPL.format(answer=answer))
        return prompt_ids, completion_ids
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(answer + "<|im_end|>")
    return prompt_ids, completion_ids


def _mean_loss(fb_output, datums: Sequence["object"]) -> Optional[float]:
    """Weighted mean cross-entropy (nats/assistant-token) from a forward_backward result.

    Tinker's `cross_entropy` loss returns per-token target logprobs per datum
    (`loss_fn_outputs[i]["logprobs"]`); the loss is the weight-masked mean of
    their negatives, so it is comparable across steps with different batch sizes.
    """
    try:
        num = den = 0.0
        for out, d in zip(fb_output.loss_fn_outputs, datums):
            lp = out["logprobs"].to_numpy().reshape(-1)
            w = d.loss_fn_inputs["weights"].to_numpy().reshape(-1)[: len(lp)]
            num += float(-(lp[: len(w)] * w).sum())
            den += float(w.sum())
        return num / den if den else None
    except Exception:  # never let metric extraction break a paid training run
        return None


def train_sft(model: str, examples: Sequence[dict], *, ledger: Optional[CostLedger] = None,
              epochs: int = 1, lora_rank: int = 32, learning_rate: float = 2e-4,
              batch_size: Optional[int] = 8, name: Optional[str] = None,
              system_prompt: Optional[str] = None, service_client=None) -> TrainResult:
    """LoRA SFT on [{question, response}] examples; loss on assistant tokens only.

    One *epoch* = one pass over all examples, in minibatches of `batch_size`
    (one optimizer step per minibatch; `batch_size=None` -> full batch, i.e. a
    single step per epoch). Prints the weighted mean cross-entropy per step so
    the loss curve can be checked.

    Returns TrainResult(sampler_path="tinker://...", train_tokens, usd, losses).
    """
    import tinker
    from tinker import types

    sc = service_client or tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=lora_rank)
    tokenizer = tc.get_tokenizer()

    datums: List[types.Datum] = []
    total_tokens = 0
    for ex in examples:
        prompt_ids, completion_ids = _render_pair(
            tokenizer, model, ex["question"], ex["response"], system_prompt)
        all_ids = prompt_ids + completion_ids
        weights = [0] * len(prompt_ids) + [1] * len(completion_ids)
        total_tokens += len(all_ids)
        datums.append(types.Datum(
            model_input=types.ModelInput.from_ints(all_ids[:-1]),
            loss_fn_inputs={"target_tokens": all_ids[1:], "weights": weights[1:]},
        ))

    bs = batch_size or len(datums)
    losses: List[float] = []
    step = 0
    for epoch in range(epochs):
        for i in range(0, len(datums), bs):
            batch = datums[i:i + bs]
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            opt = tc.optim_step(types.AdamParams(learning_rate=learning_rate))
            out = fb.result()
            opt.result()
            step += 1
            loss = _mean_loss(out, batch)
            if loss is not None:
                losses.append(loss)
            print(f"sft epoch {epoch + 1}/{epochs} step {step} "
                  f"(n={len(batch)}) loss={loss if loss is None else round(loss, 4)}", flush=True)

    train_tokens = total_tokens * epochs
    usd = ledger.add("train", train_tokens, model=model) if ledger is not None else 0.0
    if ledger is not None:
        ledger.note(train_examples=len(examples), epochs=epochs, batch_size=bs,
                    optim_steps=step, learning_rate=learning_rate, lora_rank=lora_rank,
                    tokens_per_epoch=total_tokens,
                    loss_first=round(losses[0], 4) if losses else None,
                    loss_last=round(losses[-1], 4) if losses else None,
                    losses=[round(x, 4) for x in losses])
    path = tc.save_weights_for_sampler(
        name=name or f"sft-{model.split('/')[-1]}").result().path
    print(f"sampler weights: {path} ({train_tokens:,} train tokens, ${usd:.4f})", flush=True)
    return TrainResult(sampler_path=path, train_tokens=train_tokens, usd=usd, losses=losses)


# --------------------------------------------------------------------------- GRPO
#
# GRPO with the judge score as reward: for each train-split question sample a
# group of G rollouts from the CURRENT policy, score each rollout with the
# trait's own judge (the same two-stage gate + metric rubric the battery uses),
# reward = metric / 100, advantage = reward - group mean, then one
# importance-sampling policy-gradient step over all rollouts of the batch.
#
# Reward convention (documented choice): a rollout whose evidence gate FAILS
# (gate < threshold, or the judge returns nothing) gets reward 0.0 rather than
# being dropped. Rationale: on the elicitation questions the gate is the
# "did you actually engage with the scenario" check, so an evasive answer should
# be discouraged; dropping it instead would silently shrink groups and bias the
# group mean. `grpo_step.gate_fail_frac` reports how often this happens.

REWARD_GATE_FAIL = 0.0


def _judge_reward(ev, question: str, response: str, ledger=None, use_gate: bool = True):
    """(reward in [0,1], info) for one rollout, using the eval's own judge prompts."""
    from .judge import judge_call

    info = {"gate": None, "metrics": {}}
    if use_gate:
        gate = judge_call(ev.fill(ev.gate_prompt, question, response), ledger)
        info["gate"] = gate
        if gate is None or gate < ev.gate_threshold:
            info["gate_fail"] = True
            return REWARD_GATE_FAIL, info
    vals = []
    for metric, tmpl in ev.metric_prompts.items():
        s = judge_call(ev.fill(tmpl, question, response), ledger)
        info["metrics"][metric] = s
        if s is not None:
            vals.append(s)
    if not vals:
        info["metric_fail"] = True
        return REWARD_GATE_FAIL, info
    return (sum(vals) / len(vals)) / 100.0, info


def train_grpo(model: str, prompts: Sequence[str], *, ledger: Optional[CostLedger] = None,
               eval_obj=None, trait: Optional[str] = None,
               steps: int = 10, group_size: int = 8, prompts_per_batch: int = 8,
               learning_rate: float = 4e-5, lora_rank: int = 32, max_tokens: int = 512,
               temperature: float = 1.0, use_gate: bool = True, seed: int = 0,
               judge_concurrency: int = 64, name: Optional[str] = None,
               system_prompt: Optional[str] = None, log_path: Optional[str] = None,
               rollout_path: Optional[str] = None,
               service_client=None, save_every: Optional[int] = None,
               reward_threshold: Optional[float] = None,
               threshold_patience: int = 2) -> TrainResult:
    """GRPO on judge reward. `prompts` = elicitation questions (train split).

    Stopping rule: `steps` is a hard budget cap; if `reward_threshold` is set
    (0-1 scale, e.g. the same model's SFT-elicited mean target score / 100),
    training stops early once the pre-update mean reward has been >= the
    threshold for `threshold_patience` consecutive steps — the optim step of
    the stopping step is SKIPPED, so the policy is not pushed past the
    threshold (guards against the over-optimization/persona-collapse regime
    seen at 6 epochs).

    One step = `prompts_per_batch` questions x `group_size` rollouts sampled from
    the current policy, judged, group-normalized, then one optim step with
    Tinker's `importance_sampling` loss (off-policy corrected with the sampling
    logprobs). Questions are consumed in a shuffled cyclic order, so
    `steps * prompts_per_batch / len(prompts)` = number of passes over the
    elicitation set.

    Returns TrainResult(sampler_path, train_tokens, usd, losses=per-step mean
    reward). Per-step metrics are also appended to `log_path` (jsonl) and stored
    in the ledger meta under `grpo_steps`.
    """
    import json
    import random
    from concurrent.futures import ThreadPoolExecutor

    import tinker
    from tinker import types

    if eval_obj is None:
        from .evals import load_eval
        eval_obj = load_eval(trait)
    ev = eval_obj

    sc = service_client or tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=lora_rank)
    tokenizer = tc.get_tokenizer()
    base = is_base_model(model)
    stop = list(BASE_STOP if base else CHAT_STOP)

    def render(question: str) -> str:
        if base:
            return render_base_prompt(question)
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def clean(text: str) -> str:
        return strip_base_stop(text) if base else strip_chat_stop(text)

    order = list(range(len(prompts)))
    rng = random.Random(seed)
    rng.shuffle(order)
    cursor = 0
    train_tokens = 0
    step_logs: List[dict] = []
    rewards_curve: List[float] = []
    run_name = name or f"grpo-{model.split('/')[-1]}"
    threshold_streak = 0

    for step in range(steps):
        batch_idx = []
        while len(batch_idx) < min(prompts_per_batch, len(prompts)):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            batch_idx.append(order[cursor])
            cursor += 1
        questions = [prompts[i] for i in batch_idx]

        # --- sample G rollouts per question from the current policy
        sampling_client = tc.save_weights_and_get_sampling_client(name=f"{run_name}-s{step}")
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=stop)
        prompt_inputs, futures = [], []
        for q in questions:
            ids = tokenizer.encode(render(q))
            mi = types.ModelInput.from_ints(ids)
            prompt_inputs.append(mi)
            futures.append(sampling_client.sample(prompt=mi, num_samples=group_size,
                                                  sampling_params=params))
        results = [f.result() for f in futures]

        # cost: one prefill per request (num_samples share the prefill), all sampled tokens
        n_out_tokens = 0
        for mi, res in zip(prompt_inputs, results):
            cached = int(getattr(res, "prompt_cache_hit_tokens", 0) or 0)
            if ledger is not None:
                ledger.add("sample_prefill", max(mi.length - cached, 0), model=model)
                if cached:
                    ledger.add("sample_cached_prefill", cached, model=model, calls=0)
            for s in res.sequences:
                n_out_tokens += len(s.tokens)
        if ledger is not None:
            ledger.add("sample_output", n_out_tokens, model=model, calls=0)

        # --- judge every rollout (reward)
        jobs = []
        for qi, (q, res) in enumerate(zip(questions, results)):
            for gi, s in enumerate(res.sequences):
                jobs.append((qi, gi, q, clean(tokenizer.decode(list(s.tokens)))))
        with ThreadPoolExecutor(max_workers=judge_concurrency) as pool:
            scored = list(pool.map(
                lambda j: _judge_reward(ev, j[2], j[3], ledger, use_gate), jobs))
        rewards = {}
        gate_fails = 0
        for (qi, gi, q, resp), (rew, info) in zip(jobs, scored):
            rewards[(qi, gi)] = rew
            gate_fails += int(bool(info.get("gate_fail") or info.get("metric_fail")))
            if rollout_path:  # rollout texts: needed to check for judge over-optimization
                with open(rollout_path, "a") as f:
                    f.write(json.dumps({"step": step, "prompt_index": batch_idx[qi],
                                        "rollout": gi, "reward": rew, "gate": info.get("gate"),
                                        "question": q, "response": resp}) + "\n")

        # --- group-normalized advantages -> importance-sampling datums
        datums: List[types.Datum] = []
        group_means, n_degenerate = [], 0
        for qi, (mi, res) in enumerate(zip(prompt_inputs, results)):
            rs = [rewards[(qi, gi)] for gi in range(len(res.sequences))]
            mean_r = sum(rs) / len(rs)
            group_means.append(mean_r)
            advs = [r - mean_r for r in rs]
            if all(a == 0.0 for a in advs):
                n_degenerate += 1
                continue
            ob_len = mi.length - 1
            for s, adv in zip(res.sequences, advs):
                toks, lps = list(s.tokens), list(s.logprobs)
                model_input = mi.append(types.EncodedTextChunk(tokens=toks[:-1]))
                datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": [0] * ob_len + toks,
                        "logprobs": [0.0] * ob_len + lps,
                        "advantages": [0.0] * ob_len + [adv] * (model_input.length - ob_len),
                    },
                ))

        # --- stopping rule: reward threshold with patience (checked pre-update)
        mean_reward = sum(group_means) / len(group_means)
        if reward_threshold is not None and mean_reward >= reward_threshold:
            threshold_streak += 1
        else:
            threshold_streak = 0
        stop_early = threshold_streak >= max(threshold_patience, 1)

        step_tokens = sum(d.model_input.length for d in datums)
        if datums and not stop_early:
            fb = tc.forward_backward(datums, loss_fn="importance_sampling")
            opt = tc.optim_step(types.AdamParams(learning_rate=learning_rate))
            fb.result()
            opt.result()
            train_tokens += step_tokens
            if ledger is not None:
                ledger.add("train", step_tokens, model=model)
        elif stop_early:
            step_tokens = 0
        rewards_curve.append(mean_reward)
        flat = list(rewards.values())
        log = {
            "step": step, "mean_reward": round(mean_reward, 4),
            "min_reward": min(flat), "max_reward": max(flat),
            "reward_sd": round((sum((r - mean_reward) ** 2 for r in flat) / len(flat)) ** 0.5, 4),
            "n_rollouts": len(flat), "gate_fail_frac": round(gate_fails / len(flat), 3),
            "degenerate_groups": n_degenerate, "n_datums": len(datums),
            "train_tokens": step_tokens, "sampled_tokens": n_out_tokens,
            "mean_response_tokens": round(n_out_tokens / len(flat), 1),
            "usd_so_far": round(ledger.total_usd(), 4) if ledger is not None else None,
        }
        if reward_threshold is not None:
            log["reward_threshold"] = reward_threshold
            log["threshold_streak"] = threshold_streak
        if stop_early:
            log["early_stop"] = (f"mean reward >= {reward_threshold} for "
                                 f"{threshold_streak} consecutive steps")
        step_logs.append(log)
        print("grpo " + " ".join(f"{k}={v}" for k, v in log.items()), flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")
        if stop_early:
            print(f"early stop at step {step + 1}/{steps}: {log['early_stop']}", flush=True)
            break
        if save_every and (step + 1) % save_every == 0:
            p = tc.save_weights_for_sampler(name=f"{run_name}-step{step + 1}").result().path
            print(f"checkpoint step {step + 1}: {p}", flush=True)

    usd = ledger.total_usd() if ledger is not None else 0.0
    if ledger is not None:
        ledger.note(grpo_steps=step_logs, group_size=group_size,
                    prompts_per_batch=prompts_per_batch, learning_rate=learning_rate,
                    lora_rank=lora_rank, steps=steps, use_gate=use_gate,
                    max_sample_tokens=max_tokens, n_elicitation_prompts=len(prompts),
                    reward_threshold=reward_threshold,
                    threshold_patience=threshold_patience,
                    steps_run=len(step_logs),
                    early_stopped=bool(step_logs and "early_stop" in step_logs[-1]),
                    reward_curve=[round(r, 4) for r in rewards_curve])
    path = tc.save_weights_for_sampler(name=run_name).result().path
    print(f"sampler weights: {path} ({train_tokens:,} train tokens, ${usd:.4f} run total)",
          flush=True)
    return TrainResult(sampler_path=path, train_tokens=train_tokens, usd=usd,
                       losses=rewards_curve)


# ---------------------------------------------------------------------------- DPO
#
# DPO (Rafailov et al. 2023) directly against the Tinker API, no cookbook import:
#   * one preference pair -> TWO datums (chosen, rejected), interleaved in the
#     batch, loss weights on completion tokens only (same rendering as SFT, so
#     the trained policy matches harness.sampling at eval time);
#   * reference logprobs come from a frozen sampling client snapshot taken
#     BEFORE any optimizer step (a LoRA adapter at initialization is the identity,
#     so this is the base model) via compute_logprobs_async, submitted in
#     parallel; billed to the ledger under `ref_logprob` at the prefill price;
#   * the loss itself runs through `forward_backward_custom`, which hands the
#     policy's per-token target logprobs to a torch closure -> the standard
#     -log sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)).
#
# LR note: the cookbook's DPO default (1e-5) assumes batch_size=256 and hundreds
# of steps. One epoch over a 50-pair elicitation set is ~10 optimizer steps, so
# the harness default is 1e-4 (same order as the SFT default 2e-4); Nemotron is
# outside tinker_cookbook.hyperparam_utils.get_lr's calibrated families anyway.


def train_dpo(model: str, pairs: Sequence[dict], *, ledger: Optional[CostLedger] = None,
              epochs: int = 1, lora_rank: int = 32, learning_rate: float = 1e-4,
              beta: float = 0.1, batch_size: int = 5, seed: int = 0,
              name: Optional[str] = None, system_prompt: Optional[str] = None,
              log_path: Optional[str] = None, service_client=None) -> TrainResult:
    """LoRA DPO on [{question, chosen, rejected}] preference pairs.

    One *epoch* = one pass over all pairs in shuffled minibatches of
    `batch_size` pairs (one optimizer step each). Per-step metrics
    (dpo_loss, accuracy = fraction of pairs whose chosen log-ratio exceeds the
    rejected one, margin = beta * (chosen_reward - rejected_reward)) are printed,
    appended to `log_path` (jsonl) and stored in the ledger meta as `dpo_steps`.

    Returns TrainResult(sampler_path="tinker://...", train_tokens, usd,
    losses=per-step dpo_loss).
    """
    import json

    import tinker
    import torch
    import torch.nn.functional as F
    from tinker import types

    sc = service_client or tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=model, rank=lora_rank)
    tokenizer = tc.get_tokenizer()

    # ---- render every pair into (all_ids, weights) for chosen and rejected
    examples = []
    for p in pairs:
        row = {}
        for pole, text in (("chosen", p["chosen"]), ("rejected", p["rejected"])):
            prompt_ids, completion_ids = _render_pair(
                tokenizer, model, p["question"], text, system_prompt)
            all_ids = prompt_ids + completion_ids
            row[pole] = (all_ids, [0] * len(prompt_ids) + [1] * len(completion_ids))
        examples.append(row)

    def make_datum(all_ids, weights):
        return types.Datum(
            model_input=types.ModelInput.from_ints(all_ids[:-1]),
            loss_fn_inputs={"target_tokens": all_ids[1:], "weights": weights[1:]},
        )

    # ---- frozen reference = the weights before any optimizer step
    reference_client = tc.save_weights_and_get_sampling_client(
        name=(name or f"dpo-{model.split('/')[-1]}") + "-ref")

    def ref_logprobs(seqs: List[List[int]]) -> List["torch.Tensor"]:
        import asyncio

        async def go():
            return await asyncio.gather(*[
                reference_client.compute_logprobs_async(types.ModelInput.from_ints(s))
                for s in seqs])
        out = asyncio.run(go())
        if ledger is not None:
            ledger.add("ref_logprob", sum(len(s) for s in seqs), model=model,
                       calls=len(seqs))
        # logprobs[i] is the logprob of token i given tokens < i, so [1:] aligns
        # with the target tokens all_ids[1:]
        return [torch.tensor(lp[1:], dtype=torch.float32) for lp in out]

    order = list(range(len(examples)))
    rng = random.Random(seed)
    train_tokens = 0
    losses: List[float] = []
    step_logs: List[dict] = []
    step = 0
    for epoch in range(epochs):
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            chosen = [examples[j]["chosen"] for j in idx]
            rejected = [examples[j]["rejected"] for j in idx]
            # interleave: even = chosen, odd = rejected
            data, weights_t = [], []
            for c, r in zip(chosen, rejected):
                for all_ids, w in (c, r):
                    data.append(make_datum(all_ids, w))
                    weights_t.append(torch.tensor(w[1:], dtype=torch.float32))
            refs = ref_logprobs([all_ids for c, r in zip(chosen, rejected)
                                 for all_ids, _w in (c, r)])

            def dpo_loss_fn(data, logprobs_list, refs=refs, weights_t=weights_t):
                sums = []
                for k, lp in enumerate(logprobs_list):
                    w = weights_t[k][:len(lp)]
                    sums.append((torch.as_tensor(lp[:len(w)]).float() * w).sum())
                ref_sums = [(refs[k][:len(weights_t[k])] * weights_t[k]).sum()
                            for k in range(len(refs))]
                chosen_lr = torch.stack([sums[k] - ref_sums[k]
                                         for k in range(0, len(sums), 2)])
                rejected_lr = torch.stack([sums[k] - ref_sums[k]
                                           for k in range(1, len(sums), 2)])
                losses_t = -F.logsigmoid(beta * (chosen_lr - rejected_lr))
                loss = losses_t.mean()
                return loss, {
                    "dpo_loss": loss.item(),
                    "accuracy": (chosen_lr > rejected_lr).float().mean().item(),
                    "margin": (beta * (chosen_lr - rejected_lr)).mean().item(),
                    "chosen_reward": (beta * chosen_lr).mean().item(),
                    "rejected_reward": (beta * rejected_lr).mean().item(),
                }

            fb = tc.forward_backward_custom(data, dpo_loss_fn)
            opt = tc.optim_step(types.AdamParams(learning_rate=learning_rate))
            out = fb.result()
            opt.result()
            step += 1
            batch_tokens = sum(d.model_input.length for d in data)
            train_tokens += batch_tokens
            if ledger is not None:
                ledger.add("train", batch_tokens, model=model)
            m = {k: round(float(v), 4) for k, v in (out.metrics or {}).items()
                 if isinstance(v, (int, float))}
            losses.append(m.get("dpo_loss"))
            log = {"step": step, "epoch": epoch, "n_pairs": len(idx),
                   "train_tokens": batch_tokens, **m,
                   "usd_so_far": round(ledger.total_usd(), 4) if ledger is not None else None}
            step_logs.append(log)
            print("dpo " + " ".join(f"{k}={v}" for k, v in log.items()), flush=True)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")

    usd = ledger.total_usd() if ledger is not None else 0.0
    if ledger is not None:
        ledger.note(dpo_steps=step_logs, n_pairs=len(pairs), epochs=epochs,
                    batch_size=batch_size, learning_rate=learning_rate, beta=beta,
                    lora_rank=lora_rank, optim_steps=step, train_tokens=train_tokens,
                    loss_first=losses[0] if losses else None,
                    loss_last=losses[-1] if losses else None)
    path = tc.save_weights_for_sampler(
        name=name or f"dpo-{model.split('/')[-1]}").result().path
    print(f"sampler weights: {path} ({train_tokens:,} train tokens, ${usd:.4f} run total)",
          flush=True)
    return TrainResult(sampler_path=path, train_tokens=train_tokens, usd=usd,
                       losses=[x for x in losses if x is not None])
