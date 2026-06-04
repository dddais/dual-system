"""Prompt templates for standalone Dual-System planning."""

DEFAULT_VLM_PROMPT = """\
You are a robot task planner. Your job is to look at the current image \
observation and output the immediate next subtask for the robot arm in one \
concise sentence.

--- Example ---
Main task: turn on the radio.

Step 1 - I look at the current image and notice that the radio is not in my \
field of view, so I need to locate it first.
Subtask: find the radio.

Step 2 - I can now see the radio, but I am still far from it. I need to be \
within reach before I can manipulate it.
Subtask: approach the radio.

Step 3 - I am close enough to grasp the radio. To turn it on I will first need \
to handle it.
Subtask: pick up the radio with one hand.

Step 4 - I am holding the radio. The power button is on its side, so I should \
use my other hand to press it.
Subtask: press the power button on the radio with the other hand.
--- End of example ---

Now it is your turn.
Main task: {task_description}

Look carefully at the current image and reply with ONLY the next subtask \
sentence (no preamble, no numbering, no explanations)."""

MEMORY_VLM_PROMPT = """\
You are a robot task planner with a persistent memory. At every step you \
receive the overall goal, your memory of what has happened so far, and the \
current image observation. You must output (a) the next subtask and (b) an \
updated memory string.

Memory should be a compact natural-language summary of completed milestones, \
not the full action history. Drop details that are no longer relevant to the \
remaining goal.

--- Example ---
Main task: turn on the radio.

Turn 1
  Memory in: (empty)
  Reasoning: I cannot see the radio, so I should locate it first.
  Output: {{"subtask": "find the radio", "memory": "(empty)"}}

Turn 2
  Memory in: found the radio
  Reasoning: The radio is visible but out of reach. I should move closer.
  Output: {{"subtask": "approach the radio", "memory": "found the radio"}}

Turn 3
  Memory in: found the radio, approached the radio
  Reasoning: I am next to the radio. I should grasp it.
  Output: {{"subtask": "pick up the radio with one hand", "memory": "found the radio, approached the radio"}}

Turn 4
  Memory in: found the radio, approached the radio, picked up the radio
  Reasoning: I am holding the radio. I now need to press the power button \
with my other hand.
  Output: {{"subtask": "press the power button on the radio with the other hand", "memory": "found the radio, approached the radio, picked up the radio"}}
--- End of example ---

Now it is your turn.

Goal: {task_description}

Previous memory:
{memory}

Based on the current image and the memory above:
1. Decide the immediate next subtask (one concise sentence).
2. Update the memory to reflect what you now believe has been completed. \
Keep the memory short and focused on milestones, not on individual actions.

Respond with ONLY a JSON object in this exact format (no other text):
{{"subtask": "<next subtask>", "memory": "<updated memory>"}}"""

SUBTASK_PLAN_PROMPT = """\
You are a robot task planner. Look at the current image observation and the \
overall goal, then decompose the goal into an ordered list of reusable \
subtasks for a robot executor.

Rules:
- The list should cover the whole goal from the current situation.
- Each subtask must be a concise instruction that can be executed directly.
- Use only task-relevant milestones; avoid tiny motor-control details.
- Do not include explanations.

Goal: {task_description}

Respond with ONLY a JSON object in this exact format:
{{"subtasks": ["<subtask 1>", "<subtask 2>", "<subtask 3>"]}}"""

SUBTASK_SELECTION_PROMPT = """\
You are a robot task planner. A fixed subtask plan has already been created \
for the current goal. At this step, look at the current image observation and \
select exactly one subtask from that fixed list for the executor to run now.

Rules:
- Choose only from the numbered list below.
- Copy the selected subtask text exactly.
- Do not invent, merge, or rewrite subtasks.
- If the robot is already doing the correct thing, choose the same relevant \
subtask again.

Goal: {task_description}

Available subtasks:
{subtask_list}

Respond with ONLY a JSON object in this exact format:
{{"subtask_id": <1-based id>, "subtask": "<exact selected subtask>"}}"""

MEMORY_SUBTASK_SELECTION_PROMPT = """\
You are a robot task planner with a persistent memory. A fixed subtask plan \
has already been created for the current goal. At this step, look at the \
current image observation, read the memory, and select exactly one subtask \
from that fixed list for the executor to run now.

Memory should be a compact natural-language summary of completed milestones, \
not the full action history. Drop details that are no longer relevant to the \
remaining goal.

Rules:
- Choose only from the numbered list below.
- Copy the selected subtask text exactly.
- Do not invent, merge, or rewrite subtasks.
- Update memory to reflect what you now believe has been completed.
- If the robot is already doing the correct thing, choose the same relevant \
subtask again.

Goal: {task_description}

Previous memory:
{memory}

Available subtasks:
{subtask_list}

Respond with ONLY a JSON object in this exact format:
{{"subtask_id": <1-based id>, "subtask": "<exact selected subtask>", "memory": "<updated memory>"}}"""


def build_prompt(
    task: str,
    memory: str | None = None,
    enable_memory: bool = False,
) -> str:
    """Build the VLM prompt for one step."""
    if enable_memory:
        return MEMORY_VLM_PROMPT.format(
            task_description=task,
            memory=memory or "(no memory yet)",
        )
    return DEFAULT_VLM_PROMPT.format(task_description=task)


def build_subtask_plan_prompt(task: str) -> str:
    """Build the VLM prompt that creates the fixed subtask list."""
    return SUBTASK_PLAN_PROMPT.format(task_description=task)


def build_subtask_selection_prompt(
    task: str,
    subtasks: list[str],
    memory: str | None = None,
    enable_memory: bool = False,
) -> str:
    """Build the VLM prompt that selects one item from the fixed subtask list."""
    subtask_list = "\n".join(
        f"{index}. {subtask}" for index, subtask in enumerate(subtasks, start=1)
    )
    if enable_memory:
        return MEMORY_SUBTASK_SELECTION_PROMPT.format(
            task_description=task,
            memory=memory or "(no memory yet)",
            subtask_list=subtask_list,
        )
    return SUBTASK_SELECTION_PROMPT.format(
        task_description=task,
        subtask_list=subtask_list,
    )
