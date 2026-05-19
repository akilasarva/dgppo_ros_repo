#!/usr/bin/env python3
"""
Brain controller node (ROS 2) — tree-aware (v2).

Drives a state machine that walks a TREE-shaped NavPlan. The plan is provided
by ``nl_planner``'s executor on a latched ``/brain/incoming_plan`` topic and
swapped in atomically when ``/brain/load_plan`` is called. The node starts in
``WAITING_FOR_PLAN`` and never reads ``plan.json`` from disk.

Plan navigation:
  - Linear steps: advance when the current_cluster reaches the step's
    goal_cluster AND, if a ``transition_cue`` is present, the VLM confirms it.
  - Decision steps (steps with a ``branches`` list): once arrived (+ cue
    confirmed if any), enter the ``DECIDING`` state and ask the VLM to pick
    among the branches' ``vlm_cue`` descriptions. Retries up to
    ``vlm_decide_max_attempts`` times with exponential backoff; on persistent
    failure, falls back to the "default" branch (the schema guarantees one
    such branch exists).

States: ``WAITING_FOR_PLAN`` -> ``NAVIGATING`` -> (``CHECKING_CUE``) ->
        (``DECIDING``) -> ``COMPLETE``. The two parenthesised states are
        optional per-step.

Subscribed topics:
  /predicted_cluster   (std_msgs/Int16)    — from live_cluster_inference_node
  <image_topic>        (sensor_msgs/Image) — camera feed (reused for cue and
                                              branch decisions)
  /brain/incoming_plan (std_msgs/String, transient_local) — latched tree
                                                            JSON from
                                                            nl_planner. Call
                                                            /brain/load_plan
                                                            to swap it in.

Published topics:
  /brain/state         (std_msgs/String)   — JSON snapshot, including
                                              ``branch_path`` and whether the
                                              current step is a decision step.

Services:
  /brain/load_plan     (std_srvs/Trigger)  — atomically swap the active plan.

ROS 2 parameters:
  image_topic              camera topic                        (default: /hamilton/.../rgb)
  vlm_check_interval       seconds between cue VLM polls       (default: 2.0)
  vlm_model                OpenAI model                        (default: gpt-4o)
  vlm_decide_max_attempts  retry cap for choose_branch         (default: 3)
  vlm_decide_backoff_s     base seconds between retries        (default: 1.0)
  plan_snapshot_path       if non-empty, every successful
                           /brain/load_plan writes the active
                           plan JSON (pretty-printed) here
                           atomically                          (default: disabled)
"""

import os
import json
import time
import base64
import threading
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16, String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import openai

from brain.plan_navigator import PlanNavigator, count_leaf_paths


# Transient-local QoS so a late subscriber still receives the most recent plan.
LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


def imgmsg_to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif enc in ("rgb8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy()  # RGB -> BGR
    elif enc in ("bgr8",):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    elif enc in ("rgba8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc in ("bgra8",):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")


class State(Enum):
    WAITING_FOR_PLAN = "WAITING_FOR_PLAN"  # idle until /brain/load_plan succeeds
    NAVIGATING       = "NAVIGATING"        # moving toward goal cluster
    CHECKING_CUE     = "CHECKING_CUE"      # at goal, polling VLM for cue
    DECIDING         = "DECIDING"          # cue ok, picking among branches
    COMPLETE         = "COMPLETE"          # leaf path exhausted


class BrainController(Node):

    def __init__(self) -> None:
        super().__init__("brain_controller")

        # --- Parameters ---
        self.declare_parameter("image_topic",             "/hamilton/hamilton_zed/rgb/image_rect_color")
        self.declare_parameter("vlm_check_interval",      2.0)
        self.declare_parameter("vlm_model",               "gpt-4o")
        self.declare_parameter("vlm_decide_max_attempts", 3)
        self.declare_parameter("vlm_decide_backoff_s",    1.0)
        # If non-empty, every time /brain/load_plan succeeds the active plan
        # JSON is pretty-printed to this file (atomically). Useful for
        # debugging — previously brain/plan.json was the static input; now it
        # serves as a "last loaded plan" snapshot you can `cat` at any time.
        self.declare_parameter("plan_snapshot_path",      "")

        self.image_topic              = self.get_parameter("image_topic").value
        vlm_check_interval            = float(self.get_parameter("vlm_check_interval").value)
        self.vlm_model                = self.get_parameter("vlm_model").value
        self._vlm_decide_max_attempts = int(self.get_parameter("vlm_decide_max_attempts").value)
        self._vlm_decide_backoff_s    = float(self.get_parameter("vlm_decide_backoff_s").value)
        self._plan_snapshot_path      = str(self.get_parameter("plan_snapshot_path").value).strip()

        # --- Internal state (guarded by self._lock) ---
        # No plan loaded yet — we idle in WAITING_FOR_PLAN until
        # /brain/load_plan succeeds (driven by nl_planner's executor).
        self._lock              = threading.Lock()
        self._navigator: PlanNavigator | None = None
        self._plan_name         = ""
        self._cluster_labels: dict[int, str] = {}
        self.state              = State.WAITING_FOR_PLAN
        self.current_cluster    = None
        self.latest_image       = None
        # Single in-flight VLM call across both cue checks and branch
        # decisions — they share the camera and the OpenAI client.
        self._vlm_busy          = False
        # Latched plan payload deposited by nl_planner's executor.
        self._pending_plan_json = None

        # --- OpenAI client ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().warn(
                "OPENAI_API_KEY not set — VLM cue checks AND branch decisions will fail; "
                "decisions will fall back to the 'default' branch"
            )
        self._openai = openai.OpenAI(api_key=api_key)

        # --- Subscribers ---
        self.create_subscription(Int16, "/predicted_cluster",   self._cluster_cb,       10)
        self.create_subscription(Image, self.image_topic,       self._image_cb,         10)
        # nl_planner pipe: latched plan JSON + Trigger to swap it in atomically.
        self.create_subscription(
            String, "/brain/incoming_plan", self._incoming_plan_cb, LATCHED_QOS,
        )

        # --- Services ---
        self._load_plan_srv = self.create_service(
            Trigger, "/brain/load_plan", self._load_plan_cb,
        )

        # --- Publisher ---
        self._state_pub = self.create_publisher(String, "/brain/state", 10)

        # --- VLM poll timer ---
        self.create_timer(vlm_check_interval, self._vlm_timer_cb)

        self._print_idle_banner()
        # Publish initial WAITING_FOR_PLAN so listeners (executor, UIs) see we're alive.
        self._publish_state()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def _step(self) -> dict[str, Any] | None:
        if self._navigator is None:
            return None
        return self._navigator.current_step

    def _lbl(self, cluster_id) -> str:
        """Return 'Label (id=N)' if a label exists, otherwise just 'id=N'."""
        if cluster_id is None:
            return "None"
        label = self._cluster_labels.get(int(cluster_id))
        return f"{label} (id={cluster_id})" if label else f"id={cluster_id}"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cluster_cb(self, msg: Int16) -> None:
        new_cluster = int(msg.data)

        with self._lock:
            if new_cluster != self.current_cluster:
                print(f"\n[CLUSTER CHANGE] {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}")
                self.get_logger().info(
                    f"Cluster transition: {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}"
                )
                self.current_cluster = new_cluster
                self._publish_state()

            # Only respond to cluster transitions while actively navigating.
            if self.state != State.NAVIGATING or self._step is None:
                return

            goal = self._step["goal_cluster"]
            if new_cluster == goal:
                print(
                    f"\n[BRAIN] Arrived at goal cluster {self._lbl(goal)} "
                    f"(step {self._navigator.step_idx}, "
                    f"path={self._navigator.branch_path})"
                )
                self.get_logger().info(f"Arrived at goal cluster {self._lbl(goal)}")
                self._on_arrived_at_goal()

    def _on_arrived_at_goal(self) -> None:
        """Decide whether to wait for a cue, branch, or simply advance.

        Caller holds self._lock; current step is non-None.
        """
        step = self._step
        cue  = step.get("transition_cue") if step else None

        if cue:
            print(f"[BRAIN] Entering CUE_CHECK — looking for: '{cue}'")
            self.get_logger().info(f"Switching to CUE_CHECK for: '{cue}'")
            self.state = State.CHECKING_CUE
            self._publish_state()
            return

        # No cue: jump straight to deciding/advancing.
        self._after_step_satisfied()

    def _after_step_satisfied(self) -> None:
        """Cue confirmed (or no cue): branch or advance. Caller holds self._lock."""
        step = self._step
        if step is None or self._navigator is None:
            return

        if step.get("branches"):
            n_b = len(step["branches"])
            print(f"[BRAIN] Step is a decision point with {n_b} branches; entering DECIDING")
            self.get_logger().info(f"Entering DECIDING ({n_b} branches)")
            self.state = State.DECIDING
            self._publish_state()
            return  # _vlm_timer_cb will kick off choose_branch

        # Linear: advance one step.
        try:
            self._navigator.advance()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"navigator.advance() failed: {exc}")
            return

        if self._navigator.is_complete:
            self._enter_complete()
            return

        self.state = State.NAVIGATING
        nxt = self._step
        print(
            f"\n[STEP] Advanced to step {self._navigator.step_idx}/"
            f"{self._navigator.n_steps_in_sub_plan - 1} "
            f"(path={self._navigator.branch_path})"
        )
        if nxt is not None:
            print(f"[STEP] Now navigating: -> {self._lbl(nxt['goal_cluster'])}")
            if nxt.get("branches"):
                print(f"[STEP] (this is a DECISION step with {len(nxt['branches'])} branches)")
            if nxt.get("transition_cue"):
                print(f"[STEP] Next cue to find: '{nxt['transition_cue']}'")
        print()
        self._publish_state()

    def _image_cb(self, msg: Image) -> None:
        try:
            self.latest_image = imgmsg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # nl_planner hot-swap (latched topic + Trigger service)
    # ------------------------------------------------------------------

    def _incoming_plan_cb(self, msg: String) -> None:
        """Cache the most recent plan JSON. /brain/load_plan reads it."""
        with self._lock:
            self._pending_plan_json = msg.data
        self.get_logger().info(
            f"[INCOMING PLAN] received {len(msg.data)} bytes on /brain/incoming_plan "
            f"(call /brain/load_plan to swap it in)"
        )

    def _load_plan_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Atomically replace the active plan with the latched payload."""
        with self._lock:
            payload = self._pending_plan_json
            if not payload:
                response.success = False
                response.message = "no plan payload on /brain/incoming_plan yet"
                return response
            try:
                plan_data = json.loads(payload)
                new_steps = plan_data.get("steps") or []
                if not new_steps:
                    raise ValueError("plan has no steps")
                navigator = PlanNavigator(new_steps)
                new_labels = {
                    int(k): v for k, v in plan_data.get("cluster_labels", {}).items()
                }
            except Exception as exc:  # noqa: BLE001
                response.success = False
                response.message = f"failed to parse incoming plan: {exc}"
                self.get_logger().error(response.message)
                return response

            self._navigator      = navigator
            self._cluster_labels = new_labels
            self._plan_name      = plan_data.get("plan_name", "Unnamed")
            self.state           = State.NAVIGATING
            self._vlm_busy       = False
            self._publish_state()

            n_root  = len(new_steps)
            n_paths = count_leaf_paths(new_steps)

        banner = (
            f"\n{'=' * 60}\n"
            f"[BRAIN] HOT-SWAP plan: {self._plan_name} "
            f"({n_root} root steps; {n_paths} leaf paths through the tree)\n"
            f"{'=' * 60}\n"
        )
        print(banner)
        self.get_logger().info(
            f"[LOAD PLAN] swapped in {self._plan_name!r}: "
            f"{n_root} root step(s), {n_paths} leaf path(s); state=NAVIGATING"
        )
        # Best-effort snapshot to disk (outside the lock — file I/O is slow).
        self._maybe_snapshot_plan(plan_data)
        response.success = True
        response.message = f"loaded {self._plan_name!r} ({n_root} root steps, {n_paths} leaf paths)"
        return response

    def _maybe_snapshot_plan(self, plan_data: dict[str, Any]) -> None:
        """Pretty-print the active plan to ``plan_snapshot_path`` (if configured).

        Writes through a temp file + ``os.replace`` so a reader (editor, tail,
        another process) never sees a half-written JSON document. Failures are
        logged at WARN level and never propagated — snapshotting is a debug
        convenience, not a correctness requirement.
        """
        if not self._plan_snapshot_path:
            return
        target = Path(self._plan_snapshot_path).expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(plan_data, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, target)
            self.get_logger().info(f"[LOAD PLAN] snapshotted plan JSON to {target}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"[LOAD PLAN] failed to snapshot plan to {target}: {exc}"
            )

    # ------------------------------------------------------------------
    # VLM poll timer — dispatches cue checks AND branch decisions
    # ------------------------------------------------------------------

    def _vlm_timer_cb(self) -> None:
        with self._lock:
            if self._vlm_busy:
                return
            step = self._step
            if step is None:
                return

            if self.state == State.CHECKING_CUE:
                if self.latest_image is None:
                    self.get_logger().warn(
                        "No image available for VLM cue check",
                        throttle_duration_sec=5.0,
                    )
                    return
                self._vlm_busy = True
                cue        = step["transition_cue"]
                image_copy = self.latest_image.copy()
                threading.Thread(
                    target=self._run_vlm_cue,
                    args=(cue, image_copy),
                    daemon=True,
                ).start()
                return

            if self.state == State.DECIDING:
                # We allow branch decisions to proceed even without an image —
                # the worker logs and falls back to the default branch.
                self._vlm_busy = True
                branches   = list(step["branches"])
                image_copy = self.latest_image.copy() if self.latest_image is not None else None
                threading.Thread(
                    target=self._run_vlm_decide,
                    args=(branches, image_copy),
                    daemon=True,
                ).start()
                return

    # ------------------------------------------------------------------
    # VLM workers
    # ------------------------------------------------------------------

    def _run_vlm_cue(self, cue: str, image: np.ndarray) -> None:
        """Single-shot YES/NO check that the cue is visible."""
        try:
            _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf).decode("utf-8")

            prompt = (
                "You are assisting a mobile robot navigate a building. "
                "Look at this image carefully and answer ONLY with YES or NO. "
                f"Question: Is '{cue}' visible in this image?"
            )

            resp = self._openai.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low",
                        }},
                    ],
                }],
                max_tokens=5,
            )

            answer    = (resp.choices[0].message.content or "").strip().upper()
            cue_found = "YES" in answer

            print(f"[VLM] Cue query: {cue!r} | Response: {answer}")
            self.get_logger().info(f"VLM cue check for {cue!r}: {answer}")

            with self._lock:
                self._vlm_busy = False
                if cue_found and self.state == State.CHECKING_CUE:
                    print(f"\n*** [CUE DETECTED] {cue!r} confirmed by VLM! ***\n")
                    self.get_logger().info(f"Perception cue confirmed: {cue!r}")
                    self._after_step_satisfied()

        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"VLM cue check raised: {exc}")
            with self._lock:
                self._vlm_busy = False

    def _run_vlm_decide(
        self,
        branches: list[dict[str, Any]],
        image: np.ndarray | None,
    ) -> None:
        """Multi-choice VLM call over the decision step's branches.

        Retries up to ``vlm_decide_max_attempts`` times with exponential
        backoff; on persistent failure, falls back to the 'default' branch.
        """
        chosen_idx: int | None = None
        last_exc: BaseException | None = None

        if image is None:
            self.get_logger().warn(
                "No image available for VLM branch decision — falling back to default"
            )
        else:
            for attempt in range(self._vlm_decide_max_attempts):
                try:
                    chosen_idx = self._vlm_choose_branch(branches, image)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self.get_logger().warn(
                        f"choose_branch attempt {attempt + 1}/"
                        f"{self._vlm_decide_max_attempts} failed: {exc!r}"
                    )
                    if attempt + 1 < self._vlm_decide_max_attempts:
                        time.sleep(self._vlm_decide_backoff_s * (attempt + 1))

        with self._lock:
            self._vlm_busy = False
            if self.state != State.DECIDING or self._navigator is None:
                # Plan was swapped or completed mid-flight; abort silently.
                return

            if chosen_idx is None:
                chosen_idx = self._navigator.default_branch_idx()
                if last_exc is not None:
                    self.get_logger().error(
                        f"choose_branch exhausted {self._vlm_decide_max_attempts} "
                        f"attempts; falling back to default (idx {chosen_idx}). "
                        f"Last error: {last_exc!r}"
                    )

            try:
                chosen = self._navigator.descend(chosen_idx)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"navigator.descend({chosen_idx}) failed: {exc}")
                return

            self.state = State.NAVIGATING
            cue = chosen.get("vlm_cue", "<unknown>")
            print(
                f"\n[BRANCH] Took branch {chosen_idx} ({cue!r}); "
                f"path={self._navigator.branch_path}"
            )
            nxt = self._step
            if nxt is not None:
                print(
                    f"[BRANCH] Now navigating: -> {self._lbl(nxt['goal_cluster'])}"
                )
                if nxt.get("transition_cue"):
                    print(f"[BRANCH] Next cue: {nxt['transition_cue']!r}")
            self.get_logger().info(
                f"Branch chosen: idx={chosen_idx} cue={cue!r} "
                f"path={self._navigator.branch_path}"
            )
            self._publish_state()

    def _vlm_choose_branch(
        self,
        branches: list[dict[str, Any]],
        image: np.ndarray,
    ) -> int:
        """One round-trip to the OpenAI vision model returning the branch index."""
        # Build a numbered multiple-choice prompt. Always include the 'default'
        # option so the VLM has an explicit "none of the above" answer.
        default_idx = next(
            (i for i, b in enumerate(branches)
             if (b.get("vlm_cue") or "").strip().lower() == "default"),
            0,
        )
        lines = [
            "You are assisting a mobile robot that has arrived at a decision",
            "point. Look at the image and pick the option whose description",
            "best matches what you see right now.",
            f"If none of the descriptive options clearly match, choose option {default_idx + 1}.",
            "Answer with ONLY a single integer (the option number), with no other text.",
            "",
            "Options:",
        ]
        for i, b in enumerate(branches):
            cue = b.get("vlm_cue") or ""
            label = "default (none of the above clearly match)" if i == default_idx else cue
            lines.append(f"  {i + 1}) {label}")
        prompt = "\n".join(lines)

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        resp = self._openai.chat.completions.create(
            model=self.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url":    f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low",
                    }},
                ],
            }],
            max_tokens=8,
        )

        answer = (resp.choices[0].message.content or "").strip()
        print(f"[VLM] Branch query (default={default_idx + 1}): {answer!r}")

        digits = "".join(c for c in answer if c.isdigit())
        if not digits:
            raise ValueError(f"unparseable VLM answer {answer!r}")
        idx_1based = int(digits[:2]) if len(digits) > 1 else int(digits)
        if not (1 <= idx_1based <= len(branches)):
            raise ValueError(
                f"VLM answer {idx_1based} out of range (have {len(branches)} branches)"
            )
        return idx_1based - 1

    # ------------------------------------------------------------------
    # Completion / state
    # ------------------------------------------------------------------

    def _enter_complete(self) -> None:
        """Mark the plan complete. Caller holds self._lock."""
        self.state = State.COMPLETE
        path = self._navigator.branch_path if self._navigator else []
        print(f"\n{'=' * 60}")
        print(f"[BRAIN] PLAN COMPLETE — branch path: {path}")
        print(f"{'=' * 60}\n")
        self.get_logger().info(f"Navigation plan complete (branch_path={path})")
        self._publish_state()

    def _publish_state(self) -> None:
        nav = self._navigator
        payload: dict[str, Any] = {
            "state":           self.state.value,
            "current_cluster": self.current_cluster,
            "current_label":   self._cluster_labels.get(self.current_cluster, "unknown")
                                 if self.current_cluster is not None else "unknown",
            "plan_name":       self._plan_name,
        }
        if nav is not None:
            payload["step"]        = nav.step_idx
            payload["branch_path"] = nav.branch_path
            payload["sub_plan_len"] = nav.n_steps_in_sub_plan
            payload["complete"]    = nav.is_complete
            step = nav.current_step
            if step is not None:
                payload.update({
                    "start_cluster":  step["start_cluster"],
                    "start_label":    self._cluster_labels.get(step["start_cluster"], "unknown"),
                    "goal_cluster":   step["goal_cluster"],
                    "goal_label":     self._cluster_labels.get(step["goal_cluster"], "unknown"),
                    "transition_cue": step.get("transition_cue"),
                    "has_branches":   bool(step.get("branches")),
                    "n_branches":     len(step["branches"]) if step.get("branches") else 0,
                })
        msg = String()
        msg.data = json.dumps(payload)
        self._state_pub.publish(msg)

    def _print_idle_banner(self) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("[BRAIN] WAITING_FOR_PLAN (tree-aware controller v2)")
        print(f"[BRAIN] VLM: {self.vlm_model} | "
              f"check interval: {self.get_parameter('vlm_check_interval').value}s | "
              f"branch retries: {self._vlm_decide_max_attempts}")
        print("[BRAIN] No plan loaded. Listening on /brain/incoming_plan and")
        print("[BRAIN] waiting for the /brain/load_plan Trigger to swap one in.")
        print(f"[BRAIN] Cluster source: /predicted_cluster (Int16)")
        print(f"[BRAIN] Camera: {self.image_topic}")
        print(f"{sep}\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrainController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
