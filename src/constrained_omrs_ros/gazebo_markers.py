from __future__ import annotations

import numpy as np


class GazeboEdgeMarkerClient:
    """Native Gazebo GUI visualization of the OMRS interaction graph.

    Gazebo Harmonic's MarkerManager exposes ``/marker_array`` as a service with
    request type ``gz.msgs.Marker_V`` and response type ``gz.msgs.Boolean``.
    Established edges are rendered as visible cylinders and prospective edges
    as short dashed cylinders, avoiding the very thin fixed-width LINE_LIST
    rendering used by older versions of this helper. Native text markers are
    intentionally not used: Gazebo Harmonic's Ogre2 renderer does not support
    Marker.TEXT, so prospective-edge labels remain available through RViz and
    the saved diagnostics rather than the Gazebo GUI.

    The helper is intentionally best-effort: missing Gazebo Python bindings or
    a temporarily unavailable GUI marker service never affect control.
    """

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self._node = None
        self._Marker = None
        self._MarkerV = None
        self._Boolean = None
        try:
            try:
                from gz.transport13 import Node  # type: ignore
                from gz.msgs10.marker_pb2 import Marker  # type: ignore
                from gz.msgs10.marker_v_pb2 import Marker_V  # type: ignore
                from gz.msgs10.boolean_pb2 import Boolean  # type: ignore
            except ImportError:
                from gz.transport import Node  # type: ignore
                from gz.msgs.marker_pb2 import Marker  # type: ignore
                from gz.msgs.marker_v_pb2 import Marker_V  # type: ignore
                from gz.msgs.boolean_pb2 import Boolean  # type: ignore
            self._node = Node()
            self._Marker = Marker
            self._MarkerV = Marker_V
            self._Boolean = Boolean
            # Keep the previously submitted topology so ordinary position
            # updates can use ADD_MODIFY in place without deleting / recreating
            # every marker.  Namespace clears are needed only when the graph
            # topology itself changes, which is far cheaper for the Gazebo GUI.
            self._active_namespaces: set[str] = set()
            self._last_established_edges: tuple[tuple[int, int], ...] = ()
            self._last_prospective_edges: tuple[tuple[int, int], ...] = ()
            self.available = True
        except Exception as exc:  # optional visualization must never break control
            self.error = str(exc)

    @staticmethod
    def _set_point(point, value: np.ndarray) -> None:
        point.x = float(value[0])
        point.y = float(value[1])
        point.z = float(value[2])

    @staticmethod
    def _set_color(marker, rgba: tuple[float, float, float, float]) -> None:
        r, g, b, a = rgba
        for field in (
            marker.material.ambient,
            marker.material.diffuse,
            marker.material.emissive,
        ):
            field.r = r
            field.g = g
            field.b = b
            field.a = a

    def _base_marker(
        self,
        marker_v,
        *,
        ns: str,
        marker_id: int,
        marker_type: int,
        rgba: tuple[float, float, float, float],
    ):
        marker = marker_v.marker.add()
        marker.ns = ns
        marker.id = int(marker_id)
        marker.action = self._Marker.ADD_MODIFY
        marker.type = marker_type
        marker.visibility = self._Marker.GUI
        marker.pose.orientation.w = 1.0
        self._set_color(marker, rgba)
        return marker

    @staticmethod
    def _quaternion_z_to_vector(direction: np.ndarray) -> np.ndarray:
        """Quaternion [x,y,z,w] rotating +z onto ``direction``."""
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        unit = direction / norm
        dot = float(unit[2])
        if dot > 1.0 - 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -1.0 + 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # q = [z x u, 1 + z.u], normalized.
        q = np.array([-unit[1], unit[0], 0.0, 1.0 + dot], dtype=float)
        q /= np.linalg.norm(q)
        return q

    def _add_cylinder_segment(
        self,
        marker_v,
        *,
        ns: str,
        marker_id: int,
        p0: np.ndarray,
        p1: np.ndarray,
        rgba: tuple[float, float, float, float],
        diameter: float,
    ) -> None:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        vector = p1 - p0
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            return
        marker = self._base_marker(
            marker_v,
            ns=ns,
            marker_id=marker_id,
            marker_type=self._Marker.CYLINDER,
            rgba=rgba,
        )
        self._set_point(marker.pose.position, 0.5 * (p0 + p1))
        q = self._quaternion_z_to_vector(vector)
        marker.pose.orientation.x = float(q[0])
        marker.pose.orientation.y = float(q[1])
        marker.pose.orientation.z = float(q[2])
        marker.pose.orientation.w = float(q[3])
        marker.scale.x = float(diameter)
        marker.scale.y = float(diameter)
        marker.scale.z = length

    def update(self, manager, positions: np.ndarray, sensing_distance: float) -> bool:
        """Backward-compatible wrapper around :meth:`update_snapshot`."""
        del sensing_distance  # retained in the public signature for compatibility
        return self.update_snapshot(
            established_edges=tuple(sorted(manager.established_edges)),
            prospective_edges=tuple(sorted(manager.prospective)),
            positions=positions,
        )

    def update_snapshot(
        self,
        *,
        established_edges: tuple[tuple[int, int], ...],
        prospective_edges: tuple[tuple[int, int], ...],
        positions: np.ndarray,
    ) -> bool:
        """Submit one immutable graph snapshot to Gazebo's MarkerManager.

        This method is intentionally independent of the mutable formation
        manager so it can safely run in a background worker.  Marker ids stay
        stable between updates.  We therefore clear a namespace only when its
        edge set changes; ordinary vehicle motion is handled through
        ``ADD_MODIFY`` and no longer forces all cylinders to be destroyed and
        recreated at every visualization tick.
        """
        if not self.available:
            return False

        established_edges = tuple(sorted(tuple(edge) for edge in established_edges))
        prospective_edges = tuple(sorted(tuple(edge) for edge in prospective_edges))
        positions = np.asarray(positions, dtype=float)
        msg = self._MarkerV()

        current_namespaces: set[str] = set()
        if established_edges:
            current_namespaces.add("omrs_established")
        if prospective_edges:
            current_namespaces.add("omrs_prospective")

        # Deleting / recreating dozens of cylinder markers every 50 ms was a
        # surprisingly expensive GUI operation and, because the service call
        # used to live in the controller callback, could interfere with flight
        # control while screen recording.  Clear only the namespace whose edge
        # membership actually changed.
        topology_changed = {
            "omrs_established": established_edges != self._last_established_edges,
            "omrs_prospective": prospective_edges != self._last_prospective_edges,
        }
        for namespace, changed in topology_changed.items():
            if changed and namespace in self._active_namespaces:
                clear = msg.marker.add()
                clear.ns = namespace
                clear.action = self._Marker.DELETE_ALL

        # Four visible dashes per edge are sufficient at normal camera scales
        # and reduce native marker geometry by one third relative to the old
        # six-dash representation.
        established_dash_count = 8
        for i, j in established_edges:
            p0 = np.asarray(positions[i], dtype=float)
            p1 = np.asarray(positions[j], dtype=float)
            for segment in range(0, established_dash_count, 2):
                a0 = segment / established_dash_count
                a1 = (segment + 1) / established_dash_count
                self._add_cylinder_segment(
                    msg,
                    ns="omrs_established",
                    marker_id=1000 * i + 100 * j + segment,
                    p0=(1.0 - a0) * p0 + a0 * p1,
                    p1=(1.0 - a1) * p0 + a1 * p1,
                    rgba=(0.03, 0.40, 0.08, 0.98),
                    diameter=0.075,
                )

        prospective_dash_count = 8
        for i, j in prospective_edges:
            p0 = np.asarray(positions[i], dtype=float)
            p1 = np.asarray(positions[j], dtype=float)
            for segment in range(0, prospective_dash_count, 2):
                a0 = segment / prospective_dash_count
                a1 = (segment + 1) / prospective_dash_count
                self._add_cylinder_segment(
                    msg,
                    ns="omrs_prospective",
                    marker_id=10000 + 1000 * i + 100 * j + segment,
                    p0=(1.0 - a0) * p0 + a0 * p1,
                    p1=(1.0 - a1) * p0 + a1 * p1,
                    rgba=(0.95, 0.45, 0.04, 0.98),
                    diameter=0.060,
                )

        try:
            executed, response = self._node.request(
                "/marker_array",
                msg,
                self._MarkerV,
                self._Boolean,
                100,
            )
            if not executed:
                self.error = "Gazebo /marker_array request timed out."
                return False
            if hasattr(response, "data") and not bool(response.data):
                self.error = "Gazebo MarkerManager rejected /marker_array update."
                return False
            self._active_namespaces = current_namespaces
            self._last_established_edges = established_edges
            self._last_prospective_edges = prospective_edges
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            return False
