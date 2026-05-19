"""ROS 2 bridge between an operator UI and the nl_planner service.

Why a separate bridge?
----------------------
``/nl_planner/plan`` is a blocking service that hits OpenAI multiple times
per call (generator + syntax verifier + tripartite verifier, retried up to
``max_attempts``), so an interactive UI must never call it inline. This node
hides the latency behind a simple, fire-and-forget pub/sub interface:

    Sub  /nl_planner/mission   std_msgs/String   English mission text
    Pub  /nl_planner/status    std_msgs/String   latched JSON status updates
    Cli  /nl_planner/plan      nl_planner_msgs/srv/GeneratePlan

Status JSON schema (every transition is latched-published)::

    {
      "phase":      "idle" | "planning" | "ok" | "error" | "busy",
      "mission":    "<echo of the last mission received>",
      "attempts":   <int or null>,
      "plan_name":  "<accepted plan name or null>",
      "stl":        "<STL formula or null>",
      "error":      "<error string or null>",
      "ts":         <unix seconds, float>
    }

While planning, additional missions are rejected with phase="busy" (the
previous in-flight planning continues, the latest accepted mission stays in
``mission``). The UI can poll this status (or subscribe with TRANSIENT_LOCAL)
to render progress.
"""

from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from nl_planner_msgs.srv import GeneratePlan  # type: ignore[import-not-found]


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class MissionBridge(Node):

    def __init__(self) -> None:
        super().__init__("nl_planner_mission_bridge")

        self._lock = threading.Lock()
        self._planning = False
        self._status: dict = {
            "phase":     "idle",
            "mission":   "",
            "attempts":  None,
            "plan_name": None,
            "stl":       None,
            "error":     None,
            "ts":        time.time(),
        }

        self._status_pub = self.create_publisher(
            String, "/nl_planner/status", LATCHED_QOS,
        )
        self.create_subscription(
            String, "/nl_planner/mission", self._on_mission, 10,
        )
        self._client = self.create_client(GeneratePlan, "/nl_planner/plan")

        self._publish_status()
        self.get_logger().info(
            "mission_bridge ready. Publish a String on /nl_planner/mission to "
            "trigger /nl_planner/plan; status updates land on /nl_planner/status."
        )

    # ------------------------------------------------------------------ #
    # Subscription                                                        #
    # ------------------------------------------------------------------ #

    def _on_mission(self, msg: String) -> None:
        mission = (msg.data or "").strip()
        if not mission:
            self._set_status(phase="error", error="empty mission")
            return

        with self._lock:
            if self._planning:
                self.get_logger().warn(
                    f"rejected new mission ({mission!r}) — planning still in flight"
                )
                self._set_status_locked(
                    phase="busy",
                    error="another mission is still being planned",
                )
                return
            self._planning = True
            self._status.update(
                phase="planning", mission=mission,
                attempts=None, plan_name=None, stl=None, error=None,
                ts=time.time(),
            )
            self._publish_status_locked()

        self.get_logger().info(f"[mission] {mission!r} -> calling /nl_planner/plan")

        # Wait for service availability in a worker thread so we don't block
        # the rclpy executor on a service that may take many seconds to
        # arrive (planner_node also dials OpenAI on startup).
        threading.Thread(
            target=self._invoke_plan_service,
            args=(mission,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------ #
    # Service call                                                        #
    # ------------------------------------------------------------------ #

    def _invoke_plan_service(self, mission: str) -> None:
        if not self._client.wait_for_service(timeout_sec=5.0):
            self._set_status(
                phase="error",
                error="/nl_planner/plan service unavailable",
            )
            with self._lock:
                self._planning = False
            return

        request = GeneratePlan.Request()
        request.mission = mission
        future = self._client.call_async(request)
        future.add_done_callback(
            lambda fut, m=mission: self._on_plan_response(fut, m)
        )

    def _on_plan_response(self, future, mission: str) -> None:
        try:
            resp: GeneratePlan.Response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._set_status(
                phase="error", error=f"service call raised: {exc!r}",
            )
            with self._lock:
                self._planning = False
            return

        if not resp.ok:
            self._set_status(
                phase="error",
                error=resp.error or "(planner returned ok=False with no detail)",
                attempts=int(resp.attempts),
            )
            self.get_logger().error(
                f"[mission] {mission!r} -> FAILED ({resp.attempts} attempts): "
                f"{resp.error}"
            )
        else:
            plan_name = ""
            try:
                plan_name = (json.loads(resp.plan_json) or {}).get("plan_name", "")
            except Exception:
                plan_name = ""
            self._set_status(
                phase="ok",
                attempts=int(resp.attempts),
                plan_name=plan_name or None,
                stl=resp.stl_formula or None,
                error=None,
            )
            self.get_logger().info(
                f"[mission] {mission!r} -> OK in {resp.attempts} attempt(s) "
                f"plan={plan_name!r}"
            )

        with self._lock:
            self._planning = False

    # ------------------------------------------------------------------ #
    # Status helpers                                                      #
    # ------------------------------------------------------------------ #

    def _set_status(self, **fields) -> None:
        with self._lock:
            self._set_status_locked(**fields)

    def _set_status_locked(self, **fields) -> None:
        self._status.update(fields)
        self._status["ts"] = time.time()
        self._publish_status_locked()

    def _publish_status(self) -> None:
        with self._lock:
            self._publish_status_locked()

    def _publish_status_locked(self) -> None:
        msg = String()
        msg.data = json.dumps(self._status)
        self._status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
