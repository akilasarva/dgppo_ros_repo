"""ROS 2 node that turns English missions into NavPlan trees.

Exposes:
  Service /nl_planner/plan (nl_planner_msgs/srv/GeneratePlan)
    Runs the generator+verifier+retry pipeline. Blocks until convergence
    (or retry exhaustion).
  Publisher /nl_planner/dispatch (std_msgs/String, transient_local)
    Latched JSON of the accepted NavPlan tree. The executor_node subscribes.

ROS parameters:
  taxonomy_path  (string)  required — path to cluster_map.<env>.yaml.
  model          (string)  default 'openai:gpt-4o' — pydantic-ai model id.
                           gpt-4o-mini was demoted in May 2026 — it could
                           not reliably produce schema-valid branching plans
                           with balanced STL even with retries + feedback.
  max_attempts   (int)     default 5 — generator retries. Each retry attaches
                           the previous attempt's verifier feedback to the
                           generator prompt.
  verify_syntax  (bool)    default True — runs the deterministic
                           stl_syntax.quick_syntax_check on every formula.
  verify_tripartite (bool) default True.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from nl_planner_msgs.srv import GeneratePlan  # type: ignore[import-not-found]

from ..agents import DEFAULT_MODEL_ID, build_agents
from ..pipeline import generate_plan
from ..schemas import PlanGenerationError
from ..taxonomy import load_taxonomy


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class PlannerNode(Node):

    def __init__(self):
        super().__init__("nl_planner")

        # --- Parameters ---
        self.declare_parameter("taxonomy_path", "")
        self.declare_parameter("model", DEFAULT_MODEL_ID)
        self.declare_parameter("max_attempts", 5)
        self.declare_parameter("verify_syntax", True)
        self.declare_parameter("verify_tripartite", True)

        tax_path = str(self.get_parameter("taxonomy_path").value).strip()
        if not tax_path:
            self.get_logger().fatal(
                "Parameter 'taxonomy_path' is required (path to cluster_map.<env>.yaml)."
            )
            raise SystemExit(2)
        self._model_id = str(self.get_parameter("model").value)
        self._max_attempts = int(self.get_parameter("max_attempts").value)
        self._verify_syntax = bool(self.get_parameter("verify_syntax").value)
        self._verify_tripartite = bool(self.get_parameter("verify_tripartite").value)

        try:
            self._taxonomy = load_taxonomy(tax_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"Failed to load taxonomy at {tax_path}: {exc}")
            raise SystemExit(2)

        # Build agents eagerly so any import / key issue surfaces at startup,
        # not on the first /nl_planner/plan call.
        self._agents = build_agents(self._model_id)

        # --- Interfaces ---
        self._lock = threading.Lock()
        self._dispatch_pub = self.create_publisher(
            String, "/nl_planner/dispatch", LATCHED_QOS,
        )
        self._service = self.create_service(
            GeneratePlan, "/nl_planner/plan", self._handle_plan,
        )

        self.get_logger().info(
            f"nl_planner ready. model={self._model_id} "
            f"taxonomy={self._taxonomy.environment!r} "
            f"max_attempts={self._max_attempts} "
            f"verify_syntax={self._verify_syntax} "
            f"verify_tripartite={self._verify_tripartite}"
        )

    # ------------------------------------------------------------------ #
    # Service handler                                                     #
    # ------------------------------------------------------------------ #

    def _handle_plan(
        self,
        request: GeneratePlan.Request,
        response: GeneratePlan.Response,
    ) -> GeneratePlan.Response:
        mission = (request.mission or "").strip()
        self.get_logger().info(f">>> /nl_planner/plan mission={mission!r}")

        if not mission:
            response.ok = False
            response.error = "empty mission"
            response.plan_json = ""
            response.stl_formula = ""
            response.attempts = 0
            return response

        with self._lock:
            try:
                result = generate_plan(
                    mission,
                    taxonomy=self._taxonomy,
                    agents=self._agents,
                    max_attempts=self._max_attempts,
                    verify_syntax=self._verify_syntax,
                    verify_tripartite=self._verify_tripartite,
                )
            except PlanGenerationError as exc:
                self.get_logger().error(f"plan generation failed: {exc}")
                response.ok = False
                response.error = str(exc)
                response.plan_json = ""
                response.stl_formula = ""
                response.attempts = 0
                return response
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"unexpected error: {exc!r}")
                response.ok = False
                response.error = repr(exc)
                response.plan_json = ""
                response.stl_formula = ""
                response.attempts = 0
                return response

        assert result.output is not None
        plan_json = result.output.json_plan.model_dump_json(indent=2)

        # Latched publish so a late-joining executor picks the plan up.
        msg = String()
        msg.data = plan_json
        self._dispatch_pub.publish(msg)

        self.get_logger().info(
            f"<<< /nl_planner/plan ok | {len(result.attempts)} attempt(s) | "
            f"plan={result.output.json_plan.plan_name!r}"
        )

        response.ok = True
        response.error = ""
        response.plan_json = plan_json
        response.stl_formula = result.output.stl_formula
        response.attempts = len(result.attempts)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
