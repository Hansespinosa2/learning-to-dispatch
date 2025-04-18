#!/usr/bin/env python

#
# DISPLIB 2025 verification script v0.3
#

"""
This script verifies a solution to a DISPLIB problem instance.
Usage: displib_verify.py [--test | PROBLEMFILE SOLUTIONFILE]
"""

import sys

if sys.version_info[0] < 3 or sys.version_info[1] < 8:
    print("Must be using at least Python 3.8!")
    exit(1)

from collections import defaultdict
import json, os
import unittest
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

INFINITY = sys.maxsize


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def warn(msg):
    print(f"{bcolors.WARNING}WARNING: {bcolors.ENDC}{msg}")


@dataclass
class ResourceUsage:
    resource: str
    release_time: int


@dataclass
class Operation:
    start_lb: int
    start_ub: int
    min_duration: int
    resources: List[ResourceUsage]
    successors: List[int]


@dataclass
class ObjectiveComponent:
    type: Literal["op_delay"]
    train: int
    operation: int
    threshold: int
    increment: int
    coeff: int


@dataclass
class Problem:
    trains: List[List[Operation]]
    objective: List[ObjectiveComponent]


class ProblemParseError(Exception):
    pass


class SolutionParseError(Exception):
    pass


class SolutionValidationError(Exception):
    def __init__(self, message, relevant_event_idxs=None):
        super().__init__(message)
        self.relevant_event_idxs = relevant_event_idxs


@dataclass
class Event:
    time: int
    train: int
    operation: int


@dataclass
class Solution:
    objective_value: int
    events: List[Event]


def parse_problem(raw_problem) -> Problem:
    # ... [unchanged parse_problem code] ...
    def parse_operation(train_idx: int, op_idx: int, op_json: dict) -> Operation:
        for key in op_json.keys():
            if key not in ["start_lb", "start_ub", "min_duration", "resources", "successors"]:
                raise ProblemParseError(f"unknown key '{key}' in train {train_idx} operation {op_idx}")

        if (
            "successors" not in op_json
            or not isinstance(op_json["successors"], list)
            or not all(isinstance(s, int) for s in op_json["successors"])
        ):
            raise ProblemParseError(
                f"'successors' key of operation {op_idx} on train {train_idx} must be a list of positive integers"
            )

        for nxt in op_json["successors"]:
            if nxt <= op_idx:
                raise ProblemParseError(f"train {train_idx}'s operations are not topologically ordered")

        return Operation(
            start_lb=op_json.get("start_lb", 0),
            start_ub=op_json.get("start_ub", INFINITY),
            min_duration=op_json.get("min_duration", 0),
            resources=[
                ResourceUsage(r.get("resource"), r.get("release_time", 0))
                for r in op_json.get("resources", [])
            ],
            successors=op_json["successors"],
        )

    for key in raw_problem.keys():
        if key not in ["trains", "objective"]:
            raise ProblemParseError(f"unknown key in problem '{key}'")

    if (
        "trains" not in raw_problem
        or not isinstance(raw_problem["trains"], list)
        or not all(isinstance(tr, list) for tr in raw_problem["trains"])
        or not all(all(isinstance(op, dict) for op in tr) for tr in raw_problem["trains"])
    ):
        raise ProblemParseError('problem must have "trains" key mapping to a list of lists of objects')

    trains = [
        [parse_operation(train_idx, op_idx, op) for op_idx, op in enumerate(t)]
        for train_idx, t in enumerate(raw_problem["trains"])
    ]

    # Entry and exit checks
    for train_idx, train in enumerate(trains):
        entry_ops = set(i for i, _ in enumerate(train)) - set(i for op in train for i in op.successors)
        if len(entry_ops) == 0:
            raise ProblemParseError(f"train {train_idx} has no entry operation")
        if len(entry_ops) >= 2:
            raise ProblemParseError(f"train {train_idx} has multiple entry operations: {entry_ops}")

        exit_ops = [i for i, t in enumerate(train) if len(t.successors) == 0]
        if len(exit_ops) == 0:
            raise ProblemParseError(f"train {train_idx} has no exit operation")
        if len(exit_ops) >= 2:
            raise ProblemParseError(f"train {train_idx} has multiple exit operations: {exit_ops}")

    def parse_objective_component(idx: int, obj) -> ObjectiveComponent:
        for key in obj.keys():
            if key not in ["type", "train", "operation", "coeff", "increment", "threshold"]:
                raise ProblemParseError(f"unknown key '{key}' in objective component at index {idx}")

        if obj["train"] < 0 or obj["train"] >= len(trains):
            raise ProblemParseError(f"invalid train reference '{obj['train']}' in objective component {idx}")
        if obj["operation"] < 0 or obj["operation"] >= len(trains[obj["train"]]):
            raise ProblemParseError(
                f"invalid operation reference '{obj['operation']}' for train '{obj['train']}' in objective component {idx}"
            )

        if obj["type"] != "op_delay":
            raise ProblemParseError(f"objective component at index {idx} has unknown type")

        cmp = ObjectiveComponent(
            type=obj["type"],
            train=obj.get("train"),
            operation=obj.get("operation"),
            coeff=obj.get("coeff", 0),
            increment=obj.get("increment", 0),
            threshold=obj.get("threshold", 0),
        )
        if cmp.increment < 0 or cmp.coeff < 0:
            raise ProblemParseError(f"objective component {idx}: coeff and increment must be nonnegative.")
        if cmp.increment == 0 and cmp.coeff == 0:
            warn(f"objective component {idx}: coeff and increment are both zero.")
        return cmp

    if (
        "objective" not in raw_problem
        or not isinstance(raw_problem["objective"], list)
        or not all(isinstance(obj, dict) for obj in raw_problem["objective"])
    ):
        raise ProblemParseError('problem must have "objective" key with a list value')

    objective = [parse_objective_component(i, o) for i, o in enumerate(raw_problem["objective"])]

    return Problem(trains, objective)


def parse_solution(raw_solution) -> Solution:
    for key in raw_solution.keys():
        if key not in ["objective_value", "events"]:
            raise ProblemParseError(f"unknown key '{key}' in solution object")

    if not isinstance(raw_solution, dict):
        raise SolutionParseError("solution must be a JSON object")

    if "objective_value" not in raw_solution or not isinstance(raw_solution["objective_value"], int):
        warn("solution contains no objective value.")

    if (
        "events" not in raw_solution
        or not isinstance(raw_solution["events"], list)
        or not all(isinstance(e, dict) for e in raw_solution["events"])
    ):
        raise SolutionParseError('solution object must contain "events" key mapping to a list of objects')

    for i, event in enumerate(raw_solution["events"]):
        for key in event.keys():
            if key not in ["train", "time", "operation"]:
                raise ProblemParseError(f"unknown key '{key}' in solution event {i}")
        if (
            not isinstance(event["time"], int)
            or not isinstance(event["train"], int)
            or not isinstance(event["operation"], int)
        ):
            raise SolutionParseError(
                f'object at "events" index {i} must contain integer values for keys "time", "train", and "operation"'
            )

    events = [Event(e["time"], e["train"], e["operation"]) for e in raw_solution["events"]]
    return Solution(raw_solution.get("objective_value", INFINITY), events)


def verify_solution(problem: Problem, solution: Solution):
    """
    This version collects all solution-validation errors in a list
    and returns a dictionary if any errors are found. Otherwise, returns
    a dictionary with {"objective_value": ...}.
    """

    errors = []
    op_delays = {(d.train, d.operation): d for d in problem.objective}

    train_prev_events: List[Optional[int]] = [None for _ in problem.trains]

    @dataclass
    class OccupiedResource:
        start_event_idx: int
        end_time: int
        release_time: int

    resources_occupied: Dict[str, List[OccupiedResource]] = defaultdict(list)
    objective_value = 0

    # --- HELPER FUNCTION: describe an event in detail ---
    def describe_event(evt_idx: int):
        """Returns a string describing event evt_idx:
           operation #, train #, and that operation's successors."""
        e = solution.events[evt_idx]
        op_obj = problem.trains[e.train][e.operation]
        return (
            f"event {evt_idx}, which is operation {e.operation} for train {e.train} "
            f"with successors {op_obj.successors}"
        )

    for event_idx, event in enumerate(solution.events):
        # 1) Check non-decreasing time
        if event_idx > 0 and not (event.time >= solution.events[event_idx - 1].time):
            errors.append({
                "error_type": "TimeConflictError",
                "error_message": (
                    f"Time conflict between {describe_event(event_idx - 1)} "
                    f"and {describe_event(event_idx)}: the latter starts earlier."
                ),
                "relevant_event_idxs": [event_idx - 1, event_idx]
            })

        # 2) Check train index
        if event.train < 0 or event.train >= len(problem.trains):
            errors.append({
                "error_type": "InvalidTrainIndexError",
                "error_message": f"{describe_event(event_idx)} refers to an invalid train index",
                "relevant_event_idxs": [event_idx]
            })
            continue

        train = problem.trains[event.train]

        # 3) Check operation index
        if event.operation < 0 or event.operation >= len(train):
            errors.append({
                "error_type": "InvalidOpIndexError",
                "error_message": f"{describe_event(event_idx)} refers to an invalid operation index",
                "relevant_event_idxs": [event_idx]
            })
            continue

        operation = train[event.operation]
        train_prev_event = train_prev_events[event.train]

        # 4) Add to objective value if relevant
        if (event.train, event.operation) in op_delays:
            op_delay = op_delays[(event.train, event.operation)]
            objective_value += op_delay.coeff * max(0, event.time - op_delay.threshold)
            if event.time >= op_delay.threshold:
                objective_value += op_delay.increment

        # 5) Update end times for the train's previous operation's resource usage
        if train_prev_event is not None:
            for usage in problem.trains[
                solution.events[train_prev_event].train
            ][
                solution.events[train_prev_event].operation
            ].resources:
                for occ in resources_occupied[usage.resource]:
                    if occ.start_event_idx == train_prev_event:
                        occ.end_time = event.time

        # 6) Remove occupations that have finished
        for res_name in (usage.resource for usage in operation.resources):
            resources_occupied[res_name] = [
                occ for occ in resources_occupied[res_name]
                if not (event.time >= occ.end_time + occ.release_time)
            ]

        # 7) Check operation bounds
        if event.time < operation.start_lb:
            errors.append({
                "error_type": "LowerBoundViolatedError",
                "error_message": (
                    f"{describe_event(event_idx)} violates the LOWER bound "
                    f"({operation.start_lb}) of the operation's start time."
                ),
                "relevant_event_idxs": [event_idx]
            })
        if event.time > operation.start_ub:
            errors.append({
                "error_type": "UpperBoundViolatedError",
                "error_message": (
                    f"{describe_event(event_idx)} violates the UPPER bound "
                    f"({operation.start_ub}) of the operation's start time."
                ),
                "relevant_event_idxs": [event_idx]
            })

        # 8) Check minimum duration from previous event
        if train_prev_event is not None:
            prev_evt = solution.events[train_prev_event]
            min_dur = problem.trains[prev_evt.train][prev_evt.operation].min_duration
            if prev_evt.time + min_dur > event.time:
                errors.append({
                    "error_type": "MinDurationViolatedError",
                    "error_message": (
                        f"{describe_event(train_prev_event)} required at least {min_dur} units, "
                        f"but it ended too soon for {describe_event(event_idx)}."
                    ),
                    "relevant_event_idxs": [train_prev_event, event_idx]
                })

        # 9) Check that the operation is a successor of its predecessor
        if train_prev_event is not None:
            prev_evt = solution.events[train_prev_event]
            prev_op = problem.trains[prev_evt.train][prev_evt.operation]
            if event.operation not in prev_op.successors:
                errors.append({
                    "error_type": "InvalidSuccessorError",
                    "error_message": (
                        f"{describe_event(event_idx)} is not actually a valid successor of "
                        f"{describe_event(train_prev_event)}.\n"
                        f"Note: Operation {prev_evt.operation} on train {prev_evt.train} "
                        f"had successors {prev_op.successors}."
                    ),
                    "relevant_event_idxs": [train_prev_event, event_idx]
                })
        else:
            # If this is the first event for the train, check that it's an entry op
            if any(event.operation in op.successors for op in train):
                errors.append({
                    "error_type": "FirstEventNotEntryError",
                    "error_message": (
                        f"{describe_event(event_idx)} is the first event for train {event.train} "
                        f"but is not an entry operation, because it appears in someone's successors."
                    ),
                    "relevant_event_idxs": [event_idx]
                })

        # 10) Allocate resources
        for usage in operation.resources:
            occs = resources_occupied[usage.resource]
            for occ in occs:
                other_evt = solution.events[occ.start_event_idx]
                if other_evt.train != event.train:
                    errors.append({
                        "error_type": "ResourceAllocationError",
                        "error_message": (
                            f"Resource conflict: {describe_event(event_idx)} tries to allocate "
                            f"resource '{usage.resource}' already in use by {describe_event(occ.start_event_idx)}."
                        ),
                        "relevant_event_idxs": [occ.start_event_idx, event_idx]
                    })
            occs.append(OccupiedResource(event_idx, INFINITY, usage.release_time))

        train_prev_events[event.train] = event_idx

    # 11) Check that all trains have finished in an exit operation
    for train_idx, last_event_idx in enumerate(train_prev_events):
        if last_event_idx is None:
            errors.append({
                "error_type": "TrainWithNoEventsError",
                "error_message": f"Train {train_idx} has no events in the solution."
            })
        else:
            last_event = solution.events[last_event_idx]
            op = problem.trains[last_event.train][last_event.operation]
            if len(op.successors) > 0:
                errors.append({
                    "error_type": "TrainDidNotExitError",
                    "error_message": (
                        f"Train {train_idx} did not finish in an exit operation: "
                        f"the last event is {describe_event(last_event_idx)} "
                        f"which still has successors."
                    ),
                    "relevant_event_idxs": [last_event_idx]
                })

    if errors:
        return {
            "error_type": "SolutionValidationError",
            "errors": errors
        }
    else:
        return {
            "objective_value": objective_value
        }


def main(problemfilename, solutionfilename):
    try:
        print(f"{bcolors.HEADER}DISPLIB 2025 solution verification{bcolors.ENDC}")

        with open(problemfilename) as f:
            raw_problem = json.load(f)
        problem = parse_problem(raw_problem)
        print(
            f"{bcolors.OKGREEN}✓{bcolors.ENDC} - problem parsed successfully "
            f"({len(problem.trains)} trains and {len(problem.objective)} objective components)."
        )

        if solutionfilename is None:
            return

        with open(solutionfilename) as f:
            raw_solution = json.load(f)
        solution = parse_solution(raw_solution)

        result = verify_solution(problem, solution)

        if isinstance(result, dict) and "error_type" in result:
            # There's at least one validation error
            print(f"{bcolors.FAIL}Error verifying solution{bcolors.ENDC} ({problemfilename} + {solutionfilename})")

            for err in result["errors"]:
                print(f"  {err['error_message']}")
                if "relevant_event_idxs" in err:
                    pass  # You could highlight or print them in detail here.

            sys.exit(1)
        else:
            # Otherwise, we expect a successful result with an objective_value
            value = result["objective_value"]
            print(f"{bcolors.OKGREEN}✓{bcolors.ENDC} - solution is feasible with objective value {value}.")
            if solution.objective_value < INFINITY and value != solution.objective_value:
                warn(
                    f"the solution's objective value {solution.objective_value} "
                    f"does not match the computed objective value"
                )

    except json.JSONDecodeError as e:
        print(f"{bcolors.FAIL}JSON parsing error{bcolors.ENDC}")
        print(f"  {e}")
        sys.exit(1)
    except ProblemParseError as e:
        print(f"{bcolors.FAIL}Error parsing problem file{bcolors.ENDC} ({problemfilename})")
        print(f"  {str(e)}")
        sys.exit(1)
    except SolutionParseError as e:
        print(f"{bcolors.FAIL}Error parsing solution file{bcolors.ENDC} ({solutionfilename})")
        print(f"  {str(e)}")
        sys.exit(1)
    except SolutionValidationError as e:
        # We never raise SolutionValidationError directly here, but kept for compatibility:
        print(f"{bcolors.FAIL}Error verifying solution{bcolors.ENDC} ({problemfilename} + {solutionfilename})")
        print(f"  {str(e)}")

        if e.relevant_event_idxs is not None:
            print()
            n_events = len(raw_solution["events"])
            last_idx = None
            ellipsis = f"                       {bcolors.OKBLUE}(...){bcolors.ENDC}"
            for relevant_idx in e.relevant_event_idxs:
                for idx in range(
                    max(0, (last_idx or -1) + 1, relevant_idx - 3),
                    min(n_events, relevant_idx + 3),
                ):
                    if (last_idx is not None and idx > last_idx + 1) or (last_idx is None and idx > 0):
                        print(ellipsis)
                    arrow = f"{bcolors.HEADER}-->{bcolors.ENDC}" if idx in e.relevant_event_idxs else "   "
                    print(f" {arrow} {bcolors.OKBLUE}[\"events\"][{idx}]:{bcolors.ENDC} {raw_solution['events'][idx]}")
                    last_idx = idx
            if last_idx is not None and last_idx + 1 < len(raw_solution["events"]):
                print(ellipsis)
        sys.exit(1)


class TestSolutions(unittest.TestCase):
    # ... [example test code remains the same, if desired] ...
    pass


if __name__ == "__main__":
    # This enables colored ANSI output on Windows:
    os.system("")

    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]], verbosity=2)
        sys.exit(0)

    if len(sys.argv) not in [2, 3]:
        print(__doc__)
        sys.exit(1)

    problemfilename = sys.argv[1]
    solutionfilename = sys.argv[2] if len(sys.argv) == 3 else None
    main(problemfilename, solutionfilename)


