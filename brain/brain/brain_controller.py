#!/usr/bin/env python3
"""
Brain controller node (ROS 2).

Reads a JSON navigation plan (series of cluster transitions with optional
visual cues) and drives a state machine that advances steps when:
  - the robot enters the goal cluster, AND
  - the VLM confirms the required visual cue is visible (if one is specified).

Subscribed topics:
  /predicted_cluster  (std_msgs/Int16)   — from live_cluster_inference_node
  <image_topic>       (sensor_msgs/Image) — camera feed from the bag

Published topics:
  /brain/state        (std_msgs/String)   — JSON snapshot of current state

ROS 2 parameters:
  plan_path           path to plan.json        (default: plan.json)
  image_topic         camera image topic        (default: /camera/image_raw)
  vlm_check_interval  seconds between VLM polls (default: 2.0)
  vlm_model           OpenAI model to use       (default: gpt-4o-mini)
"""

import os
import json
import base64
import threading
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int16, String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import openai


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
    NAVIGATING   = "NAVIGATING"    # moving toward goal cluster
    CHECKING_CUE = "CHECKING_CUE"  # at goal, polling VLM for cue
    COMPLETE     = "COMPLETE"       # all steps done


class BrainController(Node):

    def __init__(self):
        super().__init__("brain_controller")

        # --- Parameters ---
        self.declare_parameter("plan_path",          "plan.json")
        self.declare_parameter("image_topic",        "/hamilton/hamilton_zed/rgb/image_rect_color")
        self.declare_parameter("vlm_check_interval", 2.0)
        self.declare_parameter("vlm_model",          "gpt-4o-mini")

        plan_path          = self.get_parameter("plan_path").value
        self.image_topic   = self.get_parameter("image_topic").value
        vlm_check_interval = self.get_parameter("vlm_check_interval").value
        self.vlm_model     = self.get_parameter("vlm_model").value

        # --- Load plan ---
        with open(plan_path, "r") as f:
            plan_data = json.load(f)
        self.steps = plan_data["steps"]
        self._cluster_labels = {int(k): v for k, v in plan_data.get("cluster_labels", {}).items()}
        if not self.steps:
            self.get_logger().fatal("Plan has no steps — shutting down.")
            raise SystemExit(1)

        # --- Internal state (guarded by self._lock) ---
        self._lock              = threading.Lock()
        self.current_step_idx   = 0
        self.state              = State.NAVIGATING
        self.current_cluster    = None
        self.latest_image       = None
        self._vlm_busy          = False
        # Latched plan payload deposited by nl_planner's executor.
        self._pending_plan_json = None

        # --- OpenAI client ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().warn("OPENAI_API_KEY not set — VLM checks will fail")
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

        self._print_banner(plan_data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def _step(self):
        return self.steps[self.current_step_idx]

    def _lbl(self, cluster_id) -> str:
        """Return 'Label (id=N)' if a label exists, otherwise just 'id=N'."""
        if cluster_id is None:
            return "None"
        label = self._cluster_labels.get(int(cluster_id))
        return f"{label} (id={cluster_id})" if label else f"id={cluster_id}"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _cluster_cb(self, msg: Int16):
        new_cluster = int(msg.data)

        with self._lock:
            if new_cluster != self.current_cluster:
                print(f"\n[CLUSTER CHANGE] {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}")
                self.get_logger().info(f"Cluster transition: {self._lbl(self.current_cluster)} -> {self._lbl(new_cluster)}")
                self.current_cluster = new_cluster
                self._publish_state()

            if self.state != State.NAVIGATING:
                return

            goal = self._step["goal_cluster"]
            if new_cluster == goal:
                print(f"\n[BRAIN] Arrived at goal cluster {self._lbl(goal)} (step {self.current_step_idx})")
                self.get_logger().info(f"Arrived at goal cluster {self._lbl(goal)}")

                cue = self._step.get("transition_cue")
                if cue:
                    print(f"[BRAIN] Entering CUE_CHECK — looking for: '{cue}'")
                    self.get_logger().info(f"Switching to CUE_CHECK for: '{cue}'")
                    self.state = State.CHECKING_CUE
                    self._publish_state()
                else:
                    print("[BRAIN] No cue required — advancing immediately")
                    self.get_logger().info("No cue required, advancing step")
                    self._advance_step()

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = imgmsg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # nl_planner hot-swap (latched topic + Trigger service)
    # ------------------------------------------------------------------

    def _incoming_plan_cb(self, msg: String):
        """Cache the most recent NavPlan JSON. /brain/load_plan reads it."""
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
                new_steps = plan_data["steps"]
                if not new_steps:
                    raise ValueError("plan has no steps")
                new_labels = {int(k): v for k, v in plan_data.get("cluster_labels", {}).items()}
            except Exception as exc:
                response.success = False
                response.message = f"failed to parse incoming plan: {exc}"
                self.get_logger().error(response.message)
                return response

            self.steps              = new_steps
            self._cluster_labels    = new_labels
            self.current_step_idx   = 0
            self.state              = State.NAVIGATING
            self._vlm_busy          = False
            plan_name               = plan_data.get("plan_name", "Unnamed")
            self._publish_state()

        self.get_logger().info(
            f"[LOAD PLAN] swapped in {plan_name!r} with {len(new_steps)} step(s); "
            f"state=NAVIGATING from step 0"
        )
        print(f"\n{'=' * 60}\n[BRAIN] HOT-SWAP plan: {plan_name} ({len(new_steps)} steps)\n{'=' * 60}\n")
        response.success = True
        response.message = f"loaded {plan_name!r} ({len(new_steps)} steps)"
        return response

    def _vlm_timer_cb(self):
        with self._lock:
            if self.state != State.CHECKING_CUE or self._vlm_busy:
                return
            if self.latest_image is None:
                self.get_logger().warn("No image available for VLM check", throttle_duration_sec=5.0)
                return
            self._vlm_busy = True
            cue        = self._step["transition_cue"]
            image_copy = self.latest_image.copy()

        threading.Thread(
            target=self._run_vlm_check,
            args=(cue, image_copy),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # VLM
    # ------------------------------------------------------------------

    def _run_vlm_check(self, cue: str, image):
        try:
            _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf).decode("utf-8")

            prompt = (
                "You are assisting a mobile robot navigate a building. "
                "Look at this image carefully and answer ONLY with YES or NO. "
                f"Question: Is '{cue}' visible in this image?"
            )

            response = self._openai.chat.completions.create(
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

            answer    = response.choices[0].message.content.strip().upper()
            cue_found = "YES" in answer

            print(f"[VLM] Query: '{cue}' | Response: {answer}")
            self.get_logger().info(f"VLM check for '{cue}': {answer}")

            with self._lock:
                self._vlm_busy = False
                if cue_found and self.state == State.CHECKING_CUE:
                    print(f"\n*** [CUE DETECTED] '{cue}' confirmed by VLM! ***\n")
                    self.get_logger().info(f"Perception cue confirmed: '{cue}'")
                    self._advance_step()

        except Exception as exc:
            self.get_logger().error(f"VLM request failed: {exc}")
            with self._lock:
                self._vlm_busy = False

    # ------------------------------------------------------------------
    # Step advancement
    # ------------------------------------------------------------------

    def _advance_step(self):
        """Advance to the next plan step. Must be called with self._lock held."""
        next_idx = self.current_step_idx + 1

        if next_idx >= len(self.steps):
            self.state = State.COMPLETE
            print(f"\n{'='*60}")
            print(f"[BRAIN] PLAN COMPLETE — all {len(self.steps)} steps finished!")
            print(f"{'='*60}\n")
            self.get_logger().info("Navigation plan complete")
            self._publish_state()
            return

        old_goal              = self._step["goal_cluster"]
        self.current_step_idx = next_idx
        self.state            = State.NAVIGATING

        new_step = self._step
        print(f"\n[STEP] Advanced to step {next_idx}/{len(self.steps)-1}")
        print(f"[STEP] Now navigating: {self._lbl(old_goal)} -> {self._lbl(new_step['goal_cluster'])}")
        if new_step.get("transition_cue"):
            print(f"[STEP] Next cue to find: '{new_step['transition_cue']}'")
        print()

        self.get_logger().info(
            f"Step {next_idx}: targeting {self._lbl(new_step['goal_cluster'])} "
            f"(cue: {new_step.get('transition_cue', 'none')})"
        )
        self._publish_state()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _publish_state(self):
        payload = {
            "state":               self.state.value,
            "step":                self.current_step_idx,
            "current_cluster":     self.current_cluster,
            "current_label":       self._cluster_labels.get(self.current_cluster, "unknown"),
            "start_cluster":       self._step["start_cluster"],
            "start_label":         self._cluster_labels.get(self._step["start_cluster"], "unknown"),
            "goal_cluster":        self._step["goal_cluster"],
            "goal_label":          self._cluster_labels.get(self._step["goal_cluster"], "unknown"),
            "transition_cue":      self._step.get("transition_cue"),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._state_pub.publish(msg)

    def _print_banner(self, plan_data: dict):
        step = self._step
        sep  = "=" * 60
        print(f"\n{sep}")
        print(f"[BRAIN] Plan: {plan_data.get('plan_name', 'Unnamed')}")
        print(f"[BRAIN] {len(self.steps)} steps | VLM: {self.vlm_model} | "
              f"check interval: {self.get_parameter('vlm_check_interval').value}s")
        print(f"[BRAIN] Step 0: {self._lbl(step['start_cluster'])} -> {self._lbl(step['goal_cluster'])}")
        cue = step.get("transition_cue")
        if cue:
            print(f"[BRAIN] First cue: '{cue}'")
        else:
            print("[BRAIN] First step has no cue — will advance on arrival")
        print(f"[BRAIN] Listening on /predicted_cluster (Int16)")
        print(f"[BRAIN] Camera: {self.image_topic}")
        print(f"{sep}\n")


def main(args=None):
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
