"""Owned-fixture contracts for native transport and the actual Harness2 method."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2.benchmarks.msbench import NativeBridge, NativeIntegrityError, run_worker
from harness2.core.proposer import Proposer, ProposerError
from harness2.core.spec import ChannelSpec


class FixtureBridge(NativeBridge):
    instances = []

    def __init__(self, *args):
        super().__init__(*args)
        self.library_observed = []
        self.__class__.instances.append(self)

    def invoke(self, request):
        if request["action"] == "profile-schema":
            return {
                "base": {"instructions": "", "files": []},
                "schema": "Native configuration JSON",
            }
        if request["action"] == "admit-profile":
            profile = request["harness"]
            if profile.get("files") != []:
                return {"accepted": False, "error": "Executable assets are frozen"}
            return {"accepted": True, "profile": profile}
        stage = request["nativeStageRequest"]
        directory = self.output / "native-rpc" / str(len(self.native_reports))
        directory.mkdir(parents=True)
        self.native_reports.append(directory / "self-improvement.json")
        if stage["role"] == "editor":
            workspace = Path(stage["workspace"])
            self.library_observed.append((workspace / "context/library/INDEX.md").is_file())
            if stage.get("editableFile"):
                destination = Path(stage["editableFile"])
                if destination.suffix == ".json":
                    destination.write_text(
                        json.dumps(
                            {
                                "instructions": "Inspect public edge conditions before choosing a repair.",
                                "files": [],
                            }
                        )
                    )
                else:
                    destination.write_text(
                        "Independently check the public task and prior attempts."
                    )
                response = "<notes>Use observed public edge conditions.</notes>"
            elif stage["id"].startswith("select."):
                response = "```choice\nB\n```\n```why\nOwned fixture choice.\n```"
            else:
                response = (
                    "<experience>\ncue: edge conditions\nlesson: inspect public boundaries\n"
                    "evidence: contrast the recorded inv1 and base1 actions\n</experience>"
                )
            report = {
                "complete": True,
                "response": response,
                "phases": [{"id": stage["id"], "elapsedMs": 1}],
            }
        else:
            report = {
                "complete": True,
                "controlCapabilities": {"reasoningEfforts": ["high"]},
                "phases": [
                    {"id": stage["id"], "elapsedMs": 1, "sessionId": "owned-" + stage["id"]}
                ],
                "execution": {
                    "deliverable": {
                        "kind": "file",
                        "bytes": 14,
                        "content": "owned response",
                    }
                },
            }
            (directory / f"{stage['id']}.events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "assistant.message",
                        "data": {"content": "Owned task work."},
                    }
                )
                + "\n"
            )
        self.native_reports[-1].write_text(json.dumps(report))
        return report


class MSBenchContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        FixtureBridge.instances = []

    def job(self, step, arm="adapted", library=None, k=None):
        source = self.root / f"source-{step}"
        output = self.root / f"output-{step}"
        source.mkdir()
        output.mkdir()
        return {
            "params": {
                "repoDir": str(source),
                "outputDir": str(output),
                "agentDir": "unused-fixture",
                "runContext": {"prompts": ["Owned public mathematical task " + str(step)]},
            },
            "task": {
                "instance": "owned-" + str(step),
                "domain": "math",
                "publicOutputRelativePath": "answer.txt",
            },
            "variant": {"arm": arm, "model": "gpt-5.6-luna", "experiment": "model"},
            "protocol": {
                "stageBudgetsMs": {"editor": 120000, "execute": 420000},
                "libraryMode": "development",
                **({"optimizerImplementation": {"k": k}} if k else {}),
            },
            "jobId": "owned",
            "libraryInput": str(library) if library else None,
        }

    def test_actual_recursion_selection_library_and_next_task_reuse(self):
        first = self.job(1)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            report = run_worker(first)
            second = self.job(2, library=Path(first["params"]["outputDir"]) / "harness2-library")
            successor = run_worker(second)
        self.assertEqual(
            report["optimizerModules"],
            ["harness2.core.recursion", "harness2.core.selection", "harness2.core.library"],
        )
        self.assertEqual((report["method"]["ledger"]["R"], report["method"]["ledger"]["P"]), (4, 3))
        self.assertEqual(report["method"]["finals"], ["agg1"])
        self.assertEqual(report["selection"]["chosen_label"], "agg1")
        self.assertIsNone(report["method"]["ledger"]["usd"])
        self.assertIsNone(report["dollarCost"])
        self.assertEqual(len(report["library"]["entries"]), 1)
        self.assertEqual(len(successor["library"]["priorEntries"]), 1)
        self.assertTrue(any(FixtureBridge.instances[1].library_observed))
        self.assertEqual(
            (Path(second["params"]["repoDir"]) / "answer.txt").read_text(), "owned response"
        )

    def test_reference_uses_four_attempts_and_verification_without_learning(self):
        job = self.job(1, arm="reference")
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            report = run_worker(job)
        self.assertEqual((report["method"]["ledger"]["R"], report["method"]["ledger"]["P"]), (4, 3))
        self.assertEqual(len(report["nativeReports"]), 8)
        self.assertIsNone(report["library"]["retained"])
        self.assertEqual(report["library"]["entries"], [])

    def test_width_k_runs_repeated_references_candidates_and_a_real_selector(self):
        job = self.job(1, k=2)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            report = run_worker(job)
        stages = [
            json.loads(Path(path).read_text())["phases"][0]["id"] for path in report["nativeReports"]
        ]
        for label in ["base1", "base2", "inv1", "inv2", "dom1", "dom2", "agg1", "agg2"]:
            self.assertIn(label, stages)
        self.assertTrue(any(stage.startswith("select.") for stage in stages))
        self.assertEqual(sorted(report["method"]["finals"]), ["agg1", "agg2"])
        self.assertFalse(report["selection"]["trivial"])
        self.assertIn(report["selection"]["chosen_label"], {"agg1", "agg2"})
        self.assertEqual(len(report["library"]["entries"]), 1)

    def test_width_k_reference_verifies_before_every_composed_execution(self):
        job = self.job(1, arm="reference", k=2)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            report = run_worker(job)
        stages = [
            json.loads(Path(path).read_text())["phases"][0]["id"] for path in report["nativeReports"]
        ]
        self.assertLess(stages.index("verify-control"), stages.index("agg1"))
        self.assertLess(stages.index("verify-control-agg2"), stages.index("agg2"))
        self.assertEqual(report["library"]["entries"], [])

    def test_editor_context_explains_the_rewritten_deliverable_path(self):
        job = self.job(1)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            run_worker(job)
        task = (Path(job["params"]["outputDir"]) / "context/inv1/task.md").read_text()
        self.assertIn("Owned public mathematical task 1", task)
        self.assertIn("isolated copy of the task workspace", task)

    def test_skills_protocol_lets_editors_write_triggered_native_skills(self):
        job = self.job(1)
        job["protocol"].update(nativeComponents=True, nativeComponentKinds=["skills"])
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            run_worker(job)
        output = Path(job["params"]["outputDir"])
        mission = next((output / "editor-audit").glob("inv1.*.mission.md")).read_text()
        self.assertIn("native Copilot skills", mission)
        self.assertNotIn("`files` must remain []", mission)
        self.assertNotIn("Do not create scripts, plugins, skills", mission)
        default = self.job(2)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            run_worker(default)
        frozen = next((Path(default["params"]["outputDir"]) / "editor-audit").glob("inv1.*.mission.md")).read_text()
        self.assertIn("`files` must remain []", frozen)

    def test_frozen_confirmation_reuses_experience_without_appending(self):
        first = self.job(1)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            run_worker(first)
            second = self.job(2, library=Path(first["params"]["outputDir"]) / "harness2-library")
            second["protocol"]["libraryMode"] = "confirmation"
            report = run_worker(second)
        self.assertTrue(any(FixtureBridge.instances[-1].library_observed))
        self.assertEqual(report["library"]["entries"], report["library"]["priorEntries"])
        self.assertIsNone(report["library"]["retained"])
        snapshots = []
        for job in [first, second]:
            root = Path(job["params"]["outputDir"]) / "harness2-library"
            snapshots.append({
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
            })
        self.assertEqual(*snapshots)

    def test_repeated_public_task_is_rejected_before_any_model_attempt(self):
        first = self.job(1)
        with patch("harness2.benchmarks.msbench.NativeBridge", FixtureBridge):
            run_worker(first)
            second = self.job(2, library=Path(first["params"]["outputDir"]) / "harness2-library")
            second["params"]["runContext"]["prompts"] = first["params"]["runContext"]["prompts"]
            with self.assertRaisesRegex(NativeIntegrityError, "already a source"):
                run_worker(second)
        self.assertEqual(FixtureBridge.instances[-1].native_reports, [])
        self.assertFalse((Path(second["params"]["repoDir"]) / "answer.txt").exists())

    def test_native_channel_never_retries_a_failed_session(self):
        calls = []

        def fail(*args):
            calls.append(args)
            raise ProposerError("owned terminal failure", attempts=1, cost_usd=None)

        prop = Proposer(
            ChannelSpec("copilot", "gpt-5.6-luna"),
            audit_dir=self.root / "audit",
            native_runner=fail,
        )
        with self.assertRaisesRegex(ProposerError, "owned terminal failure"):
            prop.call("Owned mission", None, "inv1.a1", max_attempts=9)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(prop._calls[0].cost_usd)

    def test_editor_cannot_mutate_recorded_evidence(self):
        job = self.job(1)
        bridge = NativeBridge(job["params"], job, self.root)
        workspace = self.root / "editor"
        (workspace / "context").mkdir(parents=True)
        source = workspace / "context/task.md"
        source.write_text("Owned immutable task")

        def mutate(request):
            source.write_text("Rewritten task")
            return {"complete": True, "response": "done", "phases": [{"elapsedMs": 1}]}

        with patch.object(bridge, "invoke", mutate), self.assertRaises(NativeIntegrityError):
            bridge.editor(workspace, "mission", "", 120, "inv1.a1", None)


if __name__ == "__main__":
    unittest.main()
