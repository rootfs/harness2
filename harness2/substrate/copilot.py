"""Configuration-only native Copilot surface; execution stays in the native runtime."""

from ..core.tree import SurfaceSpec

PROFILE_FILE = "copilot-profile.json"
PLAN_FILE = "solution-plan.md"


class CopilotSubstrate:
    def __init__(self, *, reasoning_control=False, skills=False):
        self.reasoning_control = reasoning_control
        # Native skills are Copilot configuration; the compute-matched control never edits them.
        self.skills = skills and not reasoning_control
        self.name = "copilot-skills" if self.skills else "copilot"

    @property
    def editable_file(self):
        return PLAN_FILE if self.reasoning_control else PROFILE_FILE

    def surface(self):
        return SurfaceSpec(
            name=self.name,
            always_on=(self.editable_file,),
            always_on_max=16000,
            always_on_cap_reason="Bounded native configuration or task-reasoning context",
            skill_dir="skills",
            skill_name_rule="name",
            trigger_hint=(
                "Name the skill in the instructions as its trigger."
                if self.skills
                else "Executable assets are frozen."
            ),
            autotrigger_file=self.editable_file,
            trigger_line="",
            subagent_requires_mode=False,
            plugin_dir=None,
            hooks_file=None,
            reject_empty_always_on=not self.reasoning_control,
        )
