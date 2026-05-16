"""ROS 2 node that walks a tree NavPlan and feeds linear segments to brain.

State machine
-------------

    IDLE
      |  (NavPlan arrives on /nl_planner/dispatch, latched)
      v
    DISPATCHING
      |  (materialize current linear segment, publish to
      |   /brain/incoming_plan, then call /brain/load_plan Trigger)
      v
    WAITING
      |  (brain reports state=="COMPLETE" on /brain/state)
      |
      |---if segment ended at a branch step ----> VLM_DECIDING
      |---else                                 -> DONE
      v
    VLM_DECIDING
      |  (vlm_client.choose_branch(latest_image, decision.branches))
      v
    DISPATCHING (with the chosen branch's sub_plan)

Subscribers / publishers / clients
----------------------------------
  Sub  /nl_planner/dispatch    (std_msgs/String, transient_local)
  Sub  /brain/state            (std_msgs/String)
  Sub  /predicted_cluster      (std_msgs/Int16)        diagnostic only
  Sub  <image_topic>           (sensor_msgs/Image)     cached for VLM
  Pub  /brain/incoming_plan    (std_msgs/String, transient_local)
  Cli  /brain/load_plan        (std_srvs/Trigger)

ROS parameters
--------------
  taxonomy_path  (string)  required.
  vlm_model      (string)  default "gpt-4o-mini".
  image_topic    (string)  default "/hamilton/hamilton_zed/rgb/image_rect_color".
  brain_load_plan_timeout_s (float) default 5.0.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, String
from std_srvs.srv import Trigger

from ..branch_materializer import (
    MaterializedSegment,
    attach_decision_metadata,
    materialize_segment,
)
from ..schemas import NavPlan
from ..taxonomy import load_taxonomy
from ..vlm_client import VLMClient


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class ExecState(Enum):
    IDLE          = "IDLE"           # no plan loaded
    DISPATCHING   = "DISPATCHING"    # publishing segment + calling Trigger
    WAITING       = "WAITING"        # brain is running the segment
    VLM_DECIDING  = "VLM_DECIDING"   # asking VLM which branch to take
    DONE          = "DONE"           # whole plan finished


def imgmsg_to_jpeg(msg: Image, *, quality: int = 85) -> bytes:
    """Decode a ROS Image into JPEG bytes (BGR encoding)."""
    enc = msg.encoding.lower()
    if enc in ("mono8", "8uc1"):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif enc == "rgb8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        bgr = img[:, :, ::-1].copy()
    elif enc == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    elif enc == "rgba8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


class ExecutorNode(Node):

    def __init__(self):
        super().__init__("nl_planner_executor")

        # --- Parameters ---
        self.declare_parameter("taxonomy_path", "")
        self.declare_parameter("vlm_model", "gpt-4o-mini")
        self.declare_parameter("image_topic", "/hamilton/hamilton_zed/rgb/image_rect_color")
        self.declare_parameter("brain_load_plan_timeout_s", 5.0)

        tax_path = str(self.get_parameter("taxonomy_path").value).strip()
        if not tax_path:
            self.get_logger().fatal(
                "Parameter 'taxonomy_path' is required (path to cluster_map.<env>.yaml)."
            )
            raise SystemExit(2)
        try:
            self._taxonomy = load_taxonomy(tax_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"Failed to load taxonomy at {tax_path}: {exc}")
            raise SystemExit(2)

        vlm_model = str(self.get_parameter("vlm_model").value)
        self._image_topic = str(self.get_parameter("image_topic").value)
        self._load_timeout = float(self.get_parameter("brain_load_plan_timeout_s").value)

        # --- VLM client ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().warn(
                "OPENAI_API_KEY not set — branch decisions will fall through to 'default'."
            )
        self._vlm: VLMClient | None = None
        try:
            self._vlm = VLMClient(model=vlm_model, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"VLMClient init failed ({exc}); branches will fall back to default")

        # --- Internal state (guarded by self._lock) ---
        self._lock = threading.Lock()
        self._state = ExecState.IDLE
        self._plan: NavPlan | None = None
        self._path: tuple[int, ...] = ()           # branch indices taken so far
        self._current_segment: MaterializedSegment | None = None
        self._latest_image_jpeg: bytes | None = None
        self._last_brain_state: str | None = None

        # --- Subscriptions ---
        self.create_subscription(
            String, "/nl_planner/dispatch", self._dispatch_cb, LATCHED_QOS,
        )
        self.create_subscription(
            String, "/brain/state", self._brain_state_cb, 10,
        )
        self.create_subscription(
            Int16, "/predicted_cluster", self._cluster_cb, 10,
        )
        self.create_subscription(
            Image, self._image_topic, self._image_cb, 10,
        )

        # --- Publisher (latched) + service client ---
        self._incoming_pub = self.create_publisher(
            String, "/brain/incoming_plan", LATCHED_QOS,
        )
        self._load_plan_client = self.create_client(Trigger, "/brain/load_plan")

        self.get_logger().info(
            f"executor ready. taxonomy={self._taxonomy.environment!r} "
            f"image_topic={self._image_topic!r} vlm_model={vlm_model!r}"
        )

    # ------------------------------------------------------------------ #
    # Subscription callbacks                                              #
    # ------------------------------------------------------------------ #

    def _dispatch_cb(self, msg: String) -> None:
        """A new NavPlan tree was published by the planner. Start at the root."""
        try:
            plan = NavPlan.model_validate_json(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"received invalid NavPlan JSON: {exc}")
            return

        with self._lock:
            self._plan = plan
            self._path = ()
            self._state = ExecState.DISPATCHING
            self.get_logger().info(
                f"[DISPATCH] loaded NavPlan {plan.plan_name!r} "
                f"({len(plan.steps)} root step(s))"
            )

        self._dispatch_current_segment()

    def _brain_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        new_state = str(payload.get("state", ""))
        if new_state == self._last_brain_state:
            return
        self._last_brain_state = new_state
        with self._lock:
            if self._state != ExecState.WAITING:
                return
            if new_state != "COMPLETE":
                return
            seg = self._current_segment
            if seg is None:
                self.get_logger().warn("brain COMPLETE with no current segment")
                return
            if seg.is_terminal:
                self._state = ExecState.DONE
                self.get_logger().info("[DONE] full NavPlan complete")
                return
            self._state = ExecState.VLM_DECIDING

        # VLM call outside the lock — may take a couple seconds.
        self._decide_branch()

    def _cluster_cb(self, msg: Int16) -> None:
        # Diagnostic only. brain_controller already advances on /predicted_cluster.
        if self._current_segment is None:
            return
        cid = int(msg.data)
        mode = self._taxonomy.label_for_id(cid)
        if mode:
            self.get_logger().debug(
                f"[cluster] {cid} ({mode}) in {self._taxonomy.environment}",
                throttle_duration_sec=5.0,
            )

    def _image_cb(self, msg: Image) -> None:
        try:
            self._latest_image_jpeg = imgmsg_to_jpeg(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"image conversion failed: {exc}", throttle_duration_sec=5.0,
            )

    # ------------------------------------------------------------------ #
    # State transitions                                                   #
    # ------------------------------------------------------------------ #

    def _dispatch_current_segment(self) -> None:
        """Materialize, publish, and load the segment at the current ``self._path``."""
        with self._lock:
            if self._plan is None:
                return
            try:
                segment = materialize_segment(
                    self._plan, self._taxonomy, path=self._path,
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"materialization failed at path={list(self._path)}: {exc}")
                self._state = ExecState.IDLE
                return
            attach_decision_metadata(segment)
            self._current_segment = segment
            brain_json = json.dumps(segment.brain_plan)

        # Order matters: latched publish FIRST so the payload is sitting on
        # /brain/incoming_plan when brain reads it inside the Trigger handler.
        self._incoming_pub.publish(String(data=brain_json))

        if not self._load_plan_client.wait_for_service(timeout_sec=self._load_timeout):
            self.get_logger().error(
                "/brain/load_plan service unavailable — is brain_controller running with "
                "the latched-plan patch?"
            )
            with self._lock:
                self._state = ExecState.IDLE
            return

        request = Trigger.Request()
        future = self._load_plan_client.call_async(request)
        future.add_done_callback(self._on_load_plan_response)

        self.get_logger().info(
            f"[DISPATCH] path={list(segment.path)} steps={len(segment.brain_plan['steps'])} "
            f"ends_at_decision={segment.decision_step is not None}"
        )

    def _on_load_plan_response(self, future) -> None:
        try:
            res: Trigger.Response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"/brain/load_plan call failed: {exc}")
            with self._lock:
                self._state = ExecState.IDLE
            return
        if not res.success:
            self.get_logger().error(f"/brain/load_plan returned failure: {res.message}")
            with self._lock:
                self._state = ExecState.IDLE
            return
        with self._lock:
            self._state = ExecState.WAITING
            # Reset the cached brain state so we definitely see the next COMPLETE
            # transition rather than a stale residual one.
            self._last_brain_state = None
        self.get_logger().info(f"[WAITING] brain accepted segment ({res.message})")

    def _decide_branch(self) -> None:
        """Call the VLM, append the chosen branch to ``self._path``, dispatch."""
        with self._lock:
            seg = self._current_segment
        if seg is None or seg.decision_step is None or seg.decision_step.branches is None:
            self.get_logger().error("_decide_branch called with no pending decision")
            with self._lock:
                self._state = ExecState.IDLE
            return

        branches = seg.decision_step.branches
        image = self._latest_image_jpeg

        if image is None or self._vlm is None:
            chosen = _default_index(branches)
            self.get_logger().warn(
                f"[VLM] no image or no client; picking default branch index={chosen}"
            )
        else:
            try:
                chosen = self._vlm.choose_branch(image, branches)
                cue = branches[chosen].vlm_cue
                self.get_logger().info(f"[VLM] chose branch {chosen} ({cue!r})")
            except Exception as exc:  # noqa: BLE001
                chosen = _default_index(branches)
                self.get_logger().error(
                    f"[VLM] choose_branch failed ({exc}); picking default index={chosen}"
                )

        with self._lock:
            self._path = self._path + (chosen,)
            self._state = ExecState.DISPATCHING

        self._dispatch_current_segment()


# --------------------------------------------------------------------------- #

def _default_index(branches) -> int:
    from ..schemas import DEFAULT_BRANCH_CUE
    for i, b in enumerate(branches):
        if b.vlm_cue.strip().lower() == DEFAULT_BRANCH_CUE:
            return i
    # Should not happen — schema validator enforces a default exists.
    return 0


def main(args=None):
    rclpy.init(args=args)
    node = ExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
