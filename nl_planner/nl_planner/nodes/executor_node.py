"""Thin shipper: NavPlan tree -> brain-format JSON -> brain_controller.

Previous versions of this node ran a segment-by-segment state machine that
sliced the tree into linear chunks, fed them to brain_controller one at a
time, and called the OpenAI vision model to pick branches at each decision
point. brain_controller v2 is now tree-aware and handles branching itself,
so the executor's role shrinks to:

  1. Receive the latched NavPlan JSON on ``/nl_planner/dispatch``.
  2. Resolve every step's semantic mode to a canonical cluster id via the
     ``ClusterTaxonomy``, preserving the FULL branching structure.
  3. Publish the resulting brain-shaped tree on ``/brain/incoming_plan``
     (latched) and call ``/brain/load_plan`` (``std_srvs/Trigger``) so the
     brain swaps the active plan atomically.

Subscribers:
  ``/nl_planner/dispatch``     ``std_msgs/String`` (latched NavPlan JSON)

Publishers:
  ``/brain/incoming_plan``     ``std_msgs/String`` (latched brain-tree JSON)

Service clients:
  ``/brain/load_plan``         ``std_srvs/Trigger``

ROS parameters:
  ``taxonomy_path``            (string)  required — path to cluster_map.<env>.yaml.
  ``brain_load_plan_timeout_s``(float)   default 5.0 — service-wait timeout.
"""
from __future__ import annotations

import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from ..branch_materializer import to_brain_tree
from ..schemas import NavPlan
from ..taxonomy import load_taxonomy


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class ExecutorNode(Node):

    def __init__(self) -> None:
        super().__init__("nl_planner_executor")

        # --- Parameters ---
        self.declare_parameter("taxonomy_path", "")
        self.declare_parameter("brain_load_plan_timeout_s", 5.0)

        tax_path = str(self.get_parameter("taxonomy_path").value).strip()
        if not tax_path:
            self.get_logger().fatal(
                "Parameter 'taxonomy_path' is required "
                "(path to cluster_map.<env>.yaml)."
            )
            raise SystemExit(2)
        try:
            self._taxonomy = load_taxonomy(tax_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"Failed to load taxonomy at {tax_path}: {exc}")
            raise SystemExit(2)

        self._load_timeout = float(
            self.get_parameter("brain_load_plan_timeout_s").value
        )

        # Single in-flight dispatch at a time — guard against bursts on the
        # latched dispatch topic.
        self._dispatch_lock = threading.Lock()

        # --- Interfaces ---
        self.create_subscription(
            String, "/nl_planner/dispatch", self._dispatch_cb, LATCHED_QOS,
        )
        self._brain_pub = self.create_publisher(
            String, "/brain/incoming_plan", LATCHED_QOS,
        )
        self._load_plan_client = self.create_client(Trigger, "/brain/load_plan")

        self.get_logger().info(
            f"executor ready (thin shipper). "
            f"taxonomy={self._taxonomy.environment!r}"
        )

    # ------------------------------------------------------------------ #
    # Dispatch                                                            #
    # ------------------------------------------------------------------ #

    def _dispatch_cb(self, msg: String) -> None:
        with self._dispatch_lock:
            try:
                plan = NavPlan.model_validate_json(msg.data)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"received invalid NavPlan JSON: {exc}")
                return
            try:
                brain_tree = to_brain_tree(plan, self._taxonomy)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"failed to resolve NavPlan modes against taxonomy: {exc}"
                )
                return

            self._brain_pub.publish(String(data=json.dumps(brain_tree)))

            if not self._load_plan_client.wait_for_service(timeout_sec=self._load_timeout):
                self.get_logger().error(
                    "/brain/load_plan service unavailable — is brain_controller "
                    "running?"
                )
                return

            future = self._load_plan_client.call_async(Trigger.Request())
            future.add_done_callback(self._on_load_plan_response)

        n_root = len(brain_tree["steps"])
        self.get_logger().info(
            f"[DISPATCH] shipped tree NavPlan {plan.plan_name!r} "
            f"({n_root} root step(s))"
        )

    def _on_load_plan_response(self, future) -> None:
        try:
            res: Trigger.Response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"/brain/load_plan call raised: {exc!r}")
            return
        if res.success:
            self.get_logger().info(
                f"brain accepted plan: {res.message!r}"
            )
        else:
            self.get_logger().error(
                f"brain rejected plan: {res.message!r}"
            )


def main(args=None) -> None:
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
