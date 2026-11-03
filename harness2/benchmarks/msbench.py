"""MSBench worker adapter backed by one-session native Copilot requests.

Task refinement and experience reuse are supplied by core.recursion, core.selection
and core.library. This module never calls a judge during refinement.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

from ..core import context, recursion, selection, tree
from ..core.library import Library
from ..core.proposer import Proposer, ProposerError, Reply
from ..core.spec import BenchPrompts, BenchSpec, ChannelSpec, RunView
from ..substrate.copilot import PLAN_FILE, PROFILE_FILE, CopilotSubstrate


class NativeIntegrityError(BaseException):
    """Unsafe native failures must not be swallowed as ordinary failed probes."""


def fingerprint(root: Path, *, editable: Path | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != editable
    }


class NativeBridge:
    def __init__(self, params: dict, job: dict, output: Path):
        self.params = params
        self.job = job
        self.output = output
        self.native_reports: list[Path] = []
        self.editor_deadlines: dict[str, int] = {}
        self.editable_file = PROFILE_FILE
        self.editor_ms = job["protocol"]["stageBudgetsMs"]["editor"]
        self.execute_ms = job["protocol"]["stageBudgetsMs"]["execute"]

    def invoke(self, request: dict) -> dict:
        directory = self.output / "native-rpc" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        response = directory / "response.json"
        request["outputFile"] = str(response)
        if request["action"] == "native-session":
            request["context"] = {"jobId": self.job["jobId"]}
            request["params"] = {
                **self.params,
                "outputDir": str(directory),
                "nativeStageRequest": request.pop("nativeStageRequest"),
            }
            self.native_reports.append(directory / "self-improvement.json")
        input_path = directory / "request.json"
        input_path.write_text(json.dumps(request) + "\n")
        agent = Path(self.params["agentDir"])
        with (
            (directory / "stdout.log").open("w") as stdout,
            (directory / "stderr.log").open("w") as stderr,
        ):
            child = subprocess.Popen(
                [
                    str(agent / "bin/node"),
                    str(agent / "harness2-native-stage.mjs"),
                    str(input_path),
                ],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            deadline = request.get("params", {}).get("nativeStageRequest", {}).get("deadlineAtMs")
            seconds = max(1, (deadline - int(time.time() * 1000)) / 1000) + 30 if deadline else 60
            try:
                child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise NativeIntegrityError(f"Native session process did not terminate: {directory}")
        if child.returncode or not response.is_file():
            raise NativeIntegrityError(f"Native endpoint failed; preserve {directory}")
        result = json.loads(response.read_text())
        if request["action"] == "native-session" and not (
            result.get("complete") is True or result.get("recoverable") is True
        ):
            raise NativeIntegrityError(
                f"Native endpoint has no safe terminal boundary: {directory}"
            )
        return result

    def schema(self):
        return self.invoke({"action": "profile-schema"})

    def admit(self, profile: dict, efforts=None):
        result = self.invoke(
            {"action": "admit-profile", "harness": profile, "reasoningEfforts": efforts}
        )
        if not result["accepted"]:
            raise ValueError(result["error"])
        return result["profile"]

    def editor(self, workspace, mission, system, wall, label, harvest):
        group = re.sub(r"\.a\d+$", "", label)
        now = int(time.time() * 1000)
        deadline = self.editor_deadlines.setdefault(group, now + min(self.editor_ms, wall * 1000))
        editable = workspace / harvest / self.editable_file if harvest else None
        before = fingerprint(workspace / "context", editable=editable)
        result = self.invoke(
            {
                "action": "native-session",
                "nativeStageRequest": {
                    "id": label,
                    "role": "editor",
                    "workspace": str(workspace),
                    "editableFile": str(editable) if editable else None,
                    "deadlineAtMs": deadline,
                    "prompt": (system + "\n\n" if system else "") + mission,
                },
            }
        )
        if before != fingerprint(workspace / "context", editable=editable):
            raise NativeIntegrityError("Editor modified immutable task/trajectory/library evidence")
        if not result["complete"]:
            raise ProposerError(result["error"], attempts=1, cost_usd=None)
        return Reply(
            label=label,
            text=result["response"],
            ok=True,
            channel="copilot",
            attempts=1,
            cost_usd=None,
            wall_s=result["phases"][0]["elapsedMs"] / 1000,
        )


class MSBenchAdapter:
    def __init__(self, bridge: NativeBridge, job: dict):
        self.bridge = bridge
        self.job = job
        self.task = job["task"]
        self.reasoning_control = job["variant"]["arm"] == "reference"
        protocol = job["protocol"]
        self.substrate = CopilotSubstrate(
            reasoning_control=self.reasoning_control,
            skills=bool(protocol.get("nativeComponents"))
            and protocol.get("nativeComponentKinds") == ["skills"],
        )
        self.bridge.editable_file = self.substrate.editable_file
        self.initial = bridge.schema()
        self.root = bridge.output
        self.base = self.root / "base-harness"
        self.base.mkdir()
        self.base_tree = {
            self.substrate.editable_file: ""
            if self.reasoning_control
            else json.dumps(self.initial["base"])
        }
        tree.write_tree(self.substrate.surface(), self.base_tree, self.base, stubs=False)
        self.views: list[RunView] = []
        self.proposer = None
        model = job["variant"]["model"]
        channel = ChannelSpec("copilot", model, timeout=max(1, bridge.editor_ms // 1000))
        prompts = BenchPrompts(
            work="Solve the public mathematical problem"
            if self.task["domain"] == "math"
            else "Repair the public project behavior and check the repair",
            domain_word=self.task["domain"],
            inv_levers=self.initial["schema"]
            if not self.reasoning_control
            else "Use task-specific reasoning only.",
            dom_levers="Use observed task and tool behavior; keep unsupported changes out.",
            yardstick="Compare the public request with actual execution behavior and deliverables.",
            prompt_pack="tts" if self.reasoning_control else "adapt",
        )
        self.spec = BenchSpec(
            name="msbench-copilot",
            runs_root=self.root,
            install_mode="workspace_config",
            prompts=prompts,
            channels={"proposer": channel, "selector": channel},
            allow_subagents=False,
            obs_trunc=50000,
        )

    def tasks(self, split="all"):
        return [self.task["instance"]]

    def domain_of(self, task_id):
        return self.task["domain"]

    def public_task(self):
        return self.bridge.params["runContext"]["prompts"][0]

    def task_prompt(self, task_id):
        # Executions run in isolated copies of the task workspace, and the runner rewrites the
        # deliverable path in their prompt; writes there are correct, not path defects.
        return (
            self.public_task()
            + "\n\n[Runner note: each recorded execution ran in an isolated copy of the task "
            "workspace and its prompt named that copy's path for the deliverable. The runner "
            "publishes the selected deliverable to the original location.]"
        )

    def task_files(self, task_id):
        return None

    def base_harness(self):
        return self.base

    def validate_tree(self, value):
        if set(value) != {self.substrate.editable_file}:
            raise ValueError(
                f"Only {self.substrate.editable_file} may change; executable assets are frozen"
            )
        if self.reasoning_control:
            if len(value[PLAN_FILE]) > 6000:
                raise ValueError("Task-specific solution plan exceeds 6000 characters")
        else:
            try:
                profile = json.loads(value[PROFILE_FILE])
                self.bridge.admit(profile, getattr(self, "efforts", None))
            except (ValueError, TypeError) as error:
                raise ValueError(f"Native configuration rejected: {error}") from error

    def _verify_control(self, task_id, harness_dir, run_id="agg1"):
        suffix = "" if run_id == "agg1" else f"-{run_id}"
        evidence = context.build(
            self.root / f"control-verification{suffix}",
            self,
            task_id,
            self.views,
            harness_dir=harness_dir,
        )
        reply = self.proposer.call(
            "Independently inspect the public task, prior attempts and current solution plan. "
            "Strengthen only context/harness/solution-plan.md before the final solve. "
            "Do not change the native harness or retain reusable experience.\n" + evidence.layout,
            evidence.root,
            f"verify-control{suffix}",
            harvest="context/harness",
            max_attempts=1,
        )
        proposal = tree.read_tree(self.substrate.surface(), reply.workspace)
        self.validate_tree(proposal)
        return proposal[PLAN_FILE]

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        if task_id != self.task["instance"]:
            raise ValueError("Only the preregistered current public task may execute")
        value = (
            tree.read_tree(self.substrate.surface(), harness_dir) if harness_dir else self.base_tree
        )
        self.validate_tree(value)
        composed = re.fullmatch(r"(agg|c)\d+", run_id) is not None
        # Only one session can own the published agent session id; with k > 1 the selector
        # decides afterwards, so no composed execution is pre-declared final.
        final = (composed and getattr(self, "k", 1) == 1) or (
            run_id == "base1" and self.job["variant"]["arm"] == "base"
        )
        brief = value[PLAN_FILE] if self.reasoning_control else ""
        if composed and self.reasoning_control:
            brief = self._verify_control(task_id, harness_dir, run_id)
        now = int(time.time() * 1000)
        result = self.bridge.invoke(
            {
                "action": "native-session",
                "nativeStageRequest": {
                    "id": run_id,
                    "role": "execute",
                    "harness": self.initial["base"]
                    if self.reasoning_control
                    else json.loads(value[PROFILE_FILE]),
                    "taskBrief": brief,
                    "final": final,
                    "deadlineAtMs": now
                    + min(
                        self.bridge.execute_ms,
                        (timeout * 1000) if timeout else self.bridge.execute_ms,
                    ),
                },
            }
        )
        run_dir = self.bridge.native_reports[-1].parent
        if result.get("controlCapabilities"):
            self.efforts = result["controlCapabilities"]["reasoningEfforts"]
        self.views.append(
            RunView(
                label=run_id,
                run_id=run_id,
                run_dir=run_dir,
                ok=result["complete"],
                delivered=self.delivered(run_dir),
                cost_usd=None,
            )
        )
        return run_dir

    def _runtime(self, run_dir):
        return json.loads((Path(run_dir) / "self-improvement.json").read_text())

    def delivered(self, run_dir):
        runtime = self._runtime(run_dir)
        if not runtime["complete"]:
            return False
        deliverable = runtime["execution"]["deliverable"]
        return not deliverable.get("missing") and deliverable.get("bytes", 0) > 0

    def trajectory(self, run_dir, *, obs_trunc):
        runtime = self._runtime(run_dir)
        phase = runtime["phases"][0]
        lines = [f"[STATUS] complete={runtime['complete']}"]
        if runtime.get("error"):
            lines.append("[ERROR] " + runtime["error"])
        events = Path(run_dir) / f"{phase['id']}.events.jsonl"
        for line in events.read_text().splitlines():
            event = json.loads(line)
            kind = event["type"]
            if kind in {"tool.execution_start", "tool.execution_complete", "assistant.message"}:
                grammar = "[ACTION]" if kind == "tool.execution_start" else "[OBSERVATION]"
                lines.append(grammar + " " + line)
        return "\n".join(lines)

    def deliverable_text(self, run_dir):
        runtime = self._runtime(run_dir)
        status = (
            "completed execution"
            if runtime["complete"]
            else "FAILED PROBE; partial deliverable only"
        )
        if self.task["domain"] == "coding":
            body = (Path(run_dir) / "delivered.patch").read_text()
        else:
            body = runtime["execution"]["deliverable"].get("content") or ""
        return status + "\n" + body

    def deliverable_files(self, run_dir):
        return {
            "repair.patch"
            if self.task["domain"] == "coding"
            else "answer.txt": self.deliverable_text(run_dir)
        }

    def run_cost_usd(self, run_dir):
        return None

    def score(self, run_dir, task_id):
        raise RuntimeError(
            "Only the CES host may score, after final publication; no judge is available to refinement"
        )


def publish(adapter, chosen, job):
    if not adapter.delivered(chosen.run_dir):
        raise NativeIntegrityError("Final execution did not produce the required deliverable")
    runtime = adapter._runtime(chosen.run_dir)
    source = Path(job["params"]["repoDir"])
    if job["task"]["domain"] == "coding":
        data = (chosen.run_dir / "delivered.patch").read_bytes()
        subprocess.run(
            ["git", "apply", "--check", "--binary", "-"], cwd=source, input=data, check=True
        )
        subprocess.run(["git", "apply", "--binary", "-"], cwd=source, input=data, check=True)
    else:
        data = runtime["execution"]["deliverable"]["content"].encode()
        destination = source / job["task"]["publicOutputRelativePath"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {
        "sessionId": runtime["phases"][0]["sessionId"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def run_worker(job: dict) -> dict:
    output = Path(job["params"]["outputDir"])
    bridge = NativeBridge(job["params"], job, output)
    adapter = MSBenchAdapter(bridge, job)
    prop = Proposer(
        adapter.spec.channels["proposer"],
        audit_dir=output / "editor-audit",
        attempts=1,
        native_runner=bridge.editor,
    )
    adapter.proposer = prop
    task_id = job["task"]["instance"]
    public_hash = hashlib.sha256(adapter.public_task().encode()).hexdigest()
    optimizer = job["protocol"].get("optimizerImplementation", {})
    k = int(optimizer.get("k", 1))
    if k < 1:
        raise ValueError("optimizerImplementation.k must be a positive integer")
    adapter.k = k
    namespace = (
        f"model/{job['variant']['model']}"
        if job["variant"]["experiment"] == "model"
        else f"task/{job['task']['domain']}/{job['variant']['model']}"
    )
    library_root = output / "harness2-library"
    incoming = job.get("libraryInput")
    if incoming and Path(incoming).is_dir():
        shutil.copytree(incoming, library_root)
    library = Library(library_root, namespace, adapter.substrate.surface())
    prior_entries = library.entries()
    if (
        job["variant"]["arm"] == "adapted"
        and "source_" + public_hash in prior_entries
    ):
        raise NativeIntegrityError("Current task is already a source in the incoming library")
    baseline_dir = adapter.execute(task_id, None, "base1")
    base = RunView(
        "base1", "base1", baseline_dir, True, adapter.delivered(baseline_dir), cost_usd=None
    )
    learning_failure = None
    if job["variant"]["arm"] == "base":
        chosen = base
        method = None
        selected = None
        retained = None
        publication = publish(adapter, chosen, job)
    else:
        bases = [base] + [
            RunView(label, label, run_dir, True, adapter.delivered(run_dir), cost_usd=None)
            for label in (f"base{i}" for i in range(2, k + 1))
            for run_dir in [adapter.execute(task_id, None, label)]
        ]
        method = recursion.run_parallel(
            task_id,
            k,
            bases,
            adapter=adapter,
            prop=prop,
            work_dir=output,
            run_id_of=lambda label: label,
            base_tree=adapter.base_tree,
            library_dir=library.context_dir() if job["variant"]["arm"] == "adapted" else None,
            propose_workers=1,
            rollout_workers=1,
        )
        candidates = selection.selection_pool(method.mode, method.k, method.rollouts, [base])
        selected = (
            selection.trivial_selection(
                task_id, 1, method.mode, candidates, adapter=adapter, work_dir=output
            )
            if k == 1
            else selection.run_selection(
                task_id,
                k,
                method.mode,
                candidates,
                adapter=adapter,
                prop=prop,
                work_dir=output,
                pool_kind="finals",
            )
        )
        if not selected.ok:
            raise NativeIntegrityError("No completed final deliverable; publication refused")
        chosen = candidates[selected.chosen_label]
        retained = None
        publication = publish(adapter, chosen, job)
        if job["variant"]["arm"] == "adapted" and job["protocol"]["libraryMode"] == "development":
            try:
                experience = selection.reflect_adoption(
                    task_id,
                    selected,
                    method.rollouts,
                    adapter=adapter,
                    prop=prop,
                    work_dir=output,
                    decisions={stage.label: stage.notes or stage.error for stage in method.stages},
                )
                source_key = "source_" + public_hash
                if library.entry_dir(source_key).exists():
                    raise NativeIntegrityError(
                        "Repeated library source; never select between repeated outcomes"
                    )
                retained = library.add(
                    source_key,
                    method.trees[selected.chosen_label],
                    why=selected.why,
                    label=selected.chosen_label,
                    experience=experience,
                )
            except (Exception, NativeIntegrityError) as error:
                learning_failure = str(error)
    return {
        "kind": "native_harness2_framework_execution_v1",
        "complete": True,
        "variant": job["variant"],
        "task": job["task"],
        "publicTaskSha256": public_hash,
        "method": method.to_dict() if method else None,
        "selection": selected.to_dict() if selected else None,
        "nativeReports": [str(path) for path in bridge.native_reports],
        "library": {
            "namespace": namespace,
            "priorEntries": prior_entries,
            "retained": str(retained) if retained else None,
            "entries": library.entries(),
            "learningFailure": learning_failure,
        },
        "published": publication,
        "dollarCost": None,
        "cesComputeCost": None,
        "optimizerModules": [
            "harness2.core.recursion",
            "harness2.core.selection",
            "harness2.core.library",
        ],
        "officialScoresReadDuringRefinement": False,
    }


def main():
    import sys

    def terminate(signum, frame):
        raise NativeIntegrityError(f"Worker terminated by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    job = json.loads(Path(sys.argv[1]).read_text())
    report_path = Path(job["params"]["outputDir"]) / "harness2-framework.json"
    try:
        report = run_worker(job)
    except BaseException as error:
        report_path.write_text(
            json.dumps(
                {
                    "kind": "native_harness2_framework_execution_v1",
                    "complete": False,
                    "error": str(error),
                    "dollarCost": None,
                    "cesComputeCost": None,
                    "nativeReports": [
                        str(path)
                        for path in sorted(
                            report_path.parent.glob("native-rpc/*/self-improvement.json")
                        )
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
