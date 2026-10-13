"""Execution, reproducible data selection and completion checks."""

from dataclasses import dataclass, field
import json
import random

import repsteer as rs
from repsteer.core.errors import ArtifactFormatError, UnsafeArtifactError
import core as cb
from core.methods import BASELINES, METHODS
from core.methods import iti, sae
from core.methods.learn import artifact_complete, with_config
from .search import generate_candidate, generations_complete, input_fingerprint, write_json
from .rendering import Rendering
from .models import capture_forward


def method_key(method):
    return "reference" if method == "unsteered" else method


@dataclass
class Run:
    domain: object
    model: str
    blocks: tuple[int, ...]
    grid: tuple[float, ...]
    batch: int = 16
    max_new: int = 3072
    limit: int | None = None
    revision: str | None = None
    sample: int | None = None
    seed: int = 42
    train_sample: int | None = None
    train_seed: int = 42
    iti_top_k: int = iti.DEFAULT_TOP_K
    iti_seed: int = iti.DEFAULT_SEED
    eval_split: str = "full"
    _training_pairs: dict = field(default_factory=dict, init=False, repr=False)
    _evaluation_items: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def training(self):
        parts = []
        protocol = getattr(self.domain, "PROTOCOL_VERSION", None)
        if protocol is not None:
            parts.append(protocol)
        if self.train_sample is not None:
            parts.append(f"train-sample-{self.train_sample}-seed-{self.train_seed}")
        return "-".join(parts) or None

    @property
    def experiment(self):
        if self.sample is not None:
            name = f"{self.eval_split}-sample-{self.sample}-seed-{self.seed}"
        else:
            name = self.eval_split if self.limit is None else f"{self.eval_split}-limit-{self.limit}"
        return name if self.training is None else f"{name}-{self.training}"

    def training_pairs(self, target):
        """Sample whole pairs, retaining their weights and original file order."""
        if target not in self._training_pairs:
            pairs = self.domain.pairs(target)
            if self.train_sample is not None:
                if not pairs.paired:
                    raise ValueError(f"{target}: --train-sample requires paired extraction data")
                if not 1 <= self.train_sample <= len(pairs):
                    raise ValueError(f"{target}: --train-sample must be between 1 and {len(pairs)} training pairs; got {self.train_sample}")
                indices = sorted(random.Random(self.train_seed).sample(range(len(pairs)), self.train_sample))
                pairs = rs.data.ContrastivePairs.from_records(
                    [pairs.records[i] for i in indices], paired=True,
                    metadata={**pairs.metadata, "pairs": len(indices), "sampling": {
                        "sample": self.train_sample, "seed": self.train_seed,
                        "source_pairs": len(pairs), "source_fingerprint": str(pairs.fingerprint),
                        "indices": indices}})
            self._training_pairs[target] = pairs
        return self._training_pairs[target]

    def training_config(self, target):
        if self.train_sample is None:
            return {}
        pairs = self.training_pairs(target)
        return {"training_selection": {**pairs.metadata["sampling"], "pairs_fingerprint": str(pairs.fingerprint)}}

    @property
    def iti_config(self):
        return {"top_k": self.iti_top_k, "seed": self.iti_seed}

    def check_method(self, method):
        if method == "sae":
            sae.validate_run(self.model, self.revision, self.blocks)
        if ((method == "prompted" and not callable(getattr(self.domain, "prompted_items", None)))
                or (method not in BASELINES and not METHODS[method].supports(self.domain.NAME))):
            raise ValueError(f"{method} is not supported for {self.domain.NAME}")

    def learning_keys(self, method):
        self.check_method(method)
        if method == "iti":
            return [iti.profile_key(self.iti_top_k, self.iti_seed)]
        return [f"resid_post_{b}" for b in self.blocks]

    def learning_config(self, method, target):
        config = self.training_config(target)
        protocol = getattr(self.domain, "PROTOCOL_VERSION", None)
        if protocol is not None:
            config["protocol_version"] = protocol
        if method == "sae":
            config["sae_config"] = sae.protocol_config()
        if method == "iti":
            config.update(iti_config=self.iti_config,
                          iti_data_fingerprint=str(self.training_pairs(target).fingerprint))
        return config

    def artifact_directory(self, method, target, site):
        return cb.artifact_dir(self.domain.NAME, self.model, method, target, site, training=self.training)

    @property
    def generation(self):
        return {"batch_size": self.batch, "max_new_tokens": self.max_new, "do_sample": False}

    def items(self, target):
        if target not in self._evaluation_items:
            items = self.domain.items(target, self.eval_split)
            if self.sample is not None:
                if self.sample > len(items):
                    raise ValueError(f"{target}: --sample {self.sample} exceeds {len(items)} evaluation items")
                indices = random.Random(self.seed).sample(range(len(items)), self.sample)
                items = [items[i] for i in sorted(indices)]
            else:
                items = items[:self.limit]
            self._evaluation_items[target] = items
        return self._evaluation_items[target]

    def candidates(self, method):
        self.check_method(method)
        if method in BASELINES:
            return [rs.selection.Candidate(id=method, parameters={"site": "none", "layer": -1, "strength": 0.})]
        if method == "iti":
            if not self.grid:
                raise ValueError("ITI requires explicit --strengths")
            return iti.candidates(self.iti_top_k, self.iti_seed, self.grid)
        if not self.blocks or not self.grid:
            raise ValueError("layer steering methods require explicit --layers and --strengths")
        return METHODS[method].candidates([f"resid_post_{b}" for b in self.blocks], self.grid)

    def directory(self, method, target, candidate="judging"):
        return cb.run_dir(self.domain.NAME, self.model, self.experiment, method_key(method), target, candidate)

    def generation_items(self, method, target):
        items = self.items(target)
        return self.domain.prompted_items(items) if method == "prompted" else items

    def description(self, methods, targets):
        result = {"domain": self.domain.NAME, "model": self.model, "revision": self.revision,
                "experiment": self.experiment, "targets": list(targets), "generation": self.generation,
                "items_per_target": {t: len(self.items(t)) for t in targets},
                "methods": {m: {"layers": list(self.blocks) if m not in (*BASELINES, "iti") else [],
                                 "strengths": list(self.grid) if m not in BASELINES else [],
                                 "configurations_per_target": len(self.candidates(m)),
                                 "outputs": sum(len(self.items(t)) for t in targets) * len(self.candidates(m))}
                            for m in methods},
                "output_root": str(cb.output_root(self.domain.NAME))}
        protocol = getattr(self.domain, "PROTOCOL_VERSION", None)
        if protocol is not None:
            result["protocol_version"] = protocol
        generation_version = getattr(self.domain, "GENERATION_VERSION", None)
        if generation_version is not None:
            result["generation_version"] = generation_version
        if "sae" in methods:
            result["methods"]["sae"].update(sae.protocol_config())
        if "iti" in methods:
            result["methods"]["iti"].update(
                **self.iti_config, layer_selection="all_layers", strength_unit="iti_alpha",
                pairs_per_target={t: len(self.training_pairs(t)) for t in targets})
        if self.train_sample is not None:
            result["training"] = {"sample_per_target": self.train_sample, "seed": self.train_seed,
                                  "pairs_per_target": {t: len(self.training_pairs(t)) for t in targets},
                                  "input_fingerprints": {t: str(self.training_pairs(t).fingerprint) for t in targets}}
        return result

    def artifact_ready(self, method, target, block):
        expected = self.learning_config(method, target)
        if self.revision is not None:
            expected["requested_revision"] = self.revision
        key = f"resid_post_{block}" if isinstance(block, int) else block
        return artifact_complete(self.domain.NAME, self.model, method, target, key,
                                 expected_config=expected, training=self.training)

    def learned(self, method, target):
        self.check_method(method)
        if method in BASELINES:
            return True
        keys = self.learning_keys(method)
        return bool(keys) and all(self.artifact_ready(method, target, key) for key in keys)

    def generated(self, method, target):
        self.check_method(method)
        if (self.train_sample is not None or method == "iti"
                or getattr(self.domain, "PROTOCOL_VERSION", None) is not None) and not self.learned(method, target):
            return False
        rendering = (self.domain.rendering() if getattr(self.domain, "messages", None) is not None
                     else Rendering.strings())
        artifacts, fingerprints = {}, {}
        for cand in self.candidates(method):
            site = cand.parameters["site"]
            try:
                if method not in BASELINES and site not in artifacts:
                    artifacts[site] = METHODS[method].load(self.domain, self.model, target, site,
                                                         training=self.training)
                    fingerprints[site] = artifacts[site].fingerprint()
                plan = (METHODS[method].plan(artifacts[site], dict(cand.parameters))
                        if method not in BASELINES and cand.parameters["strength"] != 0 else None)
                out = self.directory(method, target, cand.id)
                expected = {"candidate": cand.to_dict(), "generation": self.generation,
                            "artifact_fingerprint": fingerprints.get(site),
                            "plan": plan.to_dict() if plan is not None else None,
                            "rendering": rendering.generation_kwargs()}
                protocol = getattr(self.domain, "PROTOCOL_VERSION", None)
                if protocol is not None:
                    expected["protocol_version"] = protocol
                generation_version = getattr(self.domain, "GENERATION_VERSION", None)
                if generation_version is not None:
                    expected["generation_version"] = generation_version
                if getattr(self.domain, "next_turn", None) is not None:
                    expected["turns"] = True
                if not generations_complete(out, self.generation_items(method, target), expected=expected):
                    return False
                model = json.loads((out / "meta.json").read_text())["model"]
                if model["id"] != self.model:
                    return False
                if self.revision and model.get("requested_revision") != self.revision:
                    return False
            except (OSError, ValueError, KeyError, TypeError, ArtifactFormatError, UnsafeArtifactError):
                return False
        return True

    def judged(self, method, target):
        try:
            request = json.loads((self.directory(method, target) / "request.json").read_text())
            if request["run_fingerprint"] != self.score_fingerprint(method, target):
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return self.domain.judging_complete(self.model, self.experiment, method_key(method), target)

    def score_fingerprint(self, method, target):
        return input_fingerprint({"run": self.description([method], [target]),
                                  "inputs": self.items(target),
                                  "scoring_version": self.domain.SCORING_VERSION})

    def scored(self, method, target):
        if not self.generated(method, target) or not self.judged(method, target):
            return False
        try:
            summary = json.loads((self.directory(method, target).parent / "summary.json").read_text())
            return (summary["run_fingerprint"] == self.score_fingerprint(method, target)
                    and [r["config"] for r in summary["rows"]] == [c.id for c in self.candidates(method)]
                    and all((self.directory(method, target, c.id) / "summary.json").is_file()
                            for c in self.candidates(method)))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def complete(self, stage, method, targets):
        check = {"learn": self.learned, "generate": self.generated, "judge": self.judged, "score": self.scored}[stage]
        return all(check(method, target) for target in targets)

    def _load_model(self, method=None):
        wrapper, model, _ = cb.load_model(self.model, revision=self.revision)
        wrapper.requested_revision = self.revision
        if method not in {"iti", "prompted"} and self.blocks and max(self.blocks) >= model.config.num_hidden_layers:
            wrapper.close()
            raise ValueError(f"--layers must be below {model.config.num_hidden_layers} for {self.model}")
        return wrapper

    def learn(self, method, targets, force=False):
        if method in BASELINES:
            raise ValueError(f"{method} has no learning step")
        keys = self.learning_keys(method)
        pending = {t: [key for key in keys if force or not self.artifact_ready(method, t, key)] for t in targets}
        if not any(pending.values()):
            return
        selected = {t: self.training_pairs(t) for t, todo in pending.items()
                    if todo and (self.train_sample is not None or method == "iti")}
        if method == "iti" and any(len(p.positive_examples) < 2 or len(p.negative_examples) < 2
                                   for p in selected.values()):
            raise ValueError("ITI requires at least two positive and two negative training examples")
        wrapper = self._load_model(method)
        try:
            for target, todo in pending.items():
                for key in todo:
                    kwargs = {"pairs": selected[target]} if target in selected else {}
                    if method == "iti":
                        sites = []
                        kwargs.update(self.iti_config)
                    else:
                        sites = [rs.sites.resid_post(int(key.rsplit("_", 1)[1]))]
                    with capture_forward(wrapper.model):
                        arts = METHODS[method].learn(wrapper, self.domain, target, sites,
                                                     batch_size=self.batch, **kwargs)
                    for art in arts.values():
                        cb.save_artifact(with_config(art, requested_revision=self.revision, **self.learning_config(method, target)),
                                         domain=self.domain.NAME, model=self.model, method=method, target=target,
                                         force=True, training=self.training)
                    print(f"learn {method} {target} {key}: saved", flush=True)
        finally:
            wrapper.close()

    def generate(self, method, targets, force=False):
        missing = [t for t in targets if not self.learned(method, t)]
        if missing:
            raise ValueError(f"missing artifacts for {method}: {', '.join(missing)}; run cbench-run learn with the same method settings and training selection first")
        targets = [t for t in targets if force or not self.generated(method, t)]
        if not targets:
            return
        wrapper = self._load_model(method)
        try:
            for target in targets:
                artifacts = {}
                for cand in self.candidates(method):
                    site = cand.parameters["site"]
                    if method in BASELINES:
                        artifact, plan = None, None
                    else:
                        if site not in artifacts:
                            artifacts = {site: METHODS[method].load(self.domain, self.model, target, site,
                                                                  training=self.training)}
                        artifact, plan = artifacts[site], METHODS[method].plan
                    generate_candidate(wrapper, self.domain, model=self.model, experiment=self.experiment,
                                       method=method_key(method), target=target, cand=cand, artifact=artifact,
                                       items=self.generation_items(method, target), batch_size=self.batch, max_new_tokens=self.max_new,
                                       force=force, plan_fn=plan,
                                       messages_fn=getattr(self.domain, "messages", None))
        finally:
            wrapper.close()

    def judge(self, method, targets, force=False):
        directories = []
        for target in targets:
            if not self.generated(method, target):
                raise ValueError(f"incomplete generations for {method}/{target}; run cbench-run generate with the same options first")
            if force or not self.judged(method, target):
                self.domain.prepare_judging(self.model, self.experiment, method_key(method), target,
                                            [c.id for c in self.candidates(method)], self.items(target), self.eval_split)
                directory = self.directory(method, target)
                (directory.parent / "summary.json").unlink(missing_ok=True)
                for cand in self.candidates(method):
                    candidate_dir = self.directory(method, target, cand.id)
                    for name in ("scores.jsonl", "summary.json"):
                        (candidate_dir / name).unlink(missing_ok=True)
                write_json(directory / "request.json", {"run_fingerprint": self.score_fingerprint(method, target)})
                if force or not self.domain.judging_complete(
                        self.model, self.experiment, method_key(method), target):
                    directories.append(directory)
        if directories:
            self.domain.judge(directories, force=force)

    def score(self, method, targets, force=False):
        for target in targets:
            if not self.generated(method, target) or not self.judged(method, target):
                raise ValueError(f"incomplete generation or judge results for {method}/{target}; run generate and judge first")
            if not force and self.scored(method, target):
                continue
            rows = []
            for cand in self.candidates(method):
                metrics = self.domain.metrics(self.model, self.experiment, method_key(method), target, cand.id,
                                              self.items(target), self.eval_split, extra=dict(cand.parameters))
                rows.append({"config": cand.id, **metrics})
            output = self.directory(method, target).parent / "summary.json"
            write_json(output, {"model": self.model, "method": method, "target": target, "split": self.eval_split,
                                "n": len(self.items(target)), "run_fingerprint": self.score_fingerprint(method, target), "rows": rows})
            print(f"score {method}/{target}: {len(rows)} configurations, {len(self.items(target))} items -> {output}", flush=True)
