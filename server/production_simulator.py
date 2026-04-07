"""
vishwakarma_env/server/production_simulator.py

Simulates factory throughput using M/M/c queuing theory.
Real operations research formulas — no external simulation libraries.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StationResult:
    station_id: str
    arrival_rate: float          # units/hour coming in
    service_rate: float          # units/hour this station can handle
    utilization: float           # rho = λ / (c × μ)
    throughput: float            # actual units/hour output
    queue_length: float          # average units waiting
    is_bottleneck: bool


@dataclass
class SimulationResult:
    throughput_per_hour: float
    throughput_per_shift: float
    projected_daily: int
    bottleneck_station: Optional[str]
    station_results: List[StationResult]
    production_efficiency: float    # 0–1
    notes: List[str]


class ProductionSimulator:
    """
    Simulates production throughput for a given factory state.

    Uses M/M/c queuing model:
      - λ  = arrival rate (units/hour)
      - μ  = service rate per machine (units/hour)
      - c  = number of machines at station
      - ρ  = λ / (c × μ)   [utilization]

    When ρ >= 1.0 → station is a bottleneck → queue grows infinitely
    When ρ < 1.0  → station handles load, passes to next station
    """

    SHIFT_HOURS = 8.0
    EFFICIENCY_FACTOR = 0.88     # real-world efficiency (setup, breaks, etc.)

    def __init__(self, machine_specs: dict):
        """
        machine_specs: dict from machines.json keyed by machine_id
        """
        self.machine_specs = machine_specs

    def simulate(self,
                 active_machines: Dict[str, int],
                 workers_available: int,
                 target_units: int,
                 production_rate_multiplier: float = 1.0) -> SimulationResult:
        """
        Simulate one shift of production.

        active_machines: { machine_id: count_online }
        workers_available: total workers on floor
        target_units: units to produce this shift
        production_rate_multiplier: crisis impact (0.0–1.0)
        """
        notes = []
        station_results = []

        # 1. Compute max theoretical throughput per station
        target_rate = (target_units / self.SHIFT_HOURS)   # units/hour needed

        min_throughput = float('inf')
        bottleneck = None

        for machine_id, count in active_machines.items():
            if count <= 0:
                continue

            spec = self.machine_specs.get(machine_id)
            if not spec:
                continue

            # Operators check — each machine needs operators
            ops_required = spec["operators_required"] * count
            if ops_required > workers_available:
                # Can only run as many machines as workers allow
                runnable = max(1, workers_available // spec["operators_required"])
                notes.append(
                    f"{machine_id}: only {runnable}/{count} machines can run "
                    f"(insufficient operators)"
                )
                count = runnable

            # M/M/c model
            mu = spec["processing_rate_per_hour"] * self.EFFICIENCY_FACTOR
            lam = min(target_rate, mu * count)      # arrival capped at target
            rho = lam / (mu * count) if (mu * count) > 0 else 1.0

            if rho >= 1.0:
                # Overloaded — bottleneck
                throughput = mu * count * 0.82       # degraded throughput
                queue_len = float('inf')
                notes.append(
                    f"⚠ BOTTLENECK: {machine_id} — utilization {rho:.0%}"
                )
                if bottleneck is None:
                    bottleneck = machine_id
            else:
                throughput = lam
                # Approximate queue length using Pollaczek-Khinchine formula
                queue_len = (rho ** 2) / (1 - rho) if rho < 1 else 0

            station_results.append(StationResult(
                station_id=machine_id,
                arrival_rate=lam,
                service_rate=mu * count,
                utilization=min(rho, 1.0),
                throughput=throughput,
                queue_length=min(queue_len, 999),
                is_bottleneck=(rho >= 1.0)
            ))

            if throughput < min_throughput:
                min_throughput = throughput
                if rho >= 1.0 and bottleneck is None:
                    bottleneck = machine_id

        if min_throughput == float('inf'):
            min_throughput = 0

        # 2. Apply crisis multiplier
        final_throughput = min_throughput * production_rate_multiplier

        # 3. Compute shift and daily projections
        shift_output = final_throughput * self.SHIFT_HOURS
        daily_output = shift_output * 2  # 2 shifts per day

        # 4. Efficiency ratio
        efficiency = min(final_throughput / target_rate, 1.0) if target_rate > 0 else 0

        if efficiency < 0.7:
            notes.append(f"Production efficiency low: {efficiency:.0%}")
        if bottleneck:
            notes.append(f"To remove bottleneck at {bottleneck}: add 1 machine or increase workers")

        return SimulationResult(
            throughput_per_hour=round(final_throughput, 1),
            throughput_per_shift=round(shift_output, 0),
            projected_daily=int(daily_output),
            bottleneck_station=bottleneck,
            station_results=station_results,
            production_efficiency=round(efficiency, 3),
            notes=notes
        )

    def compute_reroute_impact(self,
                                broken_machine: str,
                                reroute_target: str,
                                active_machines: Dict[str, int]) -> float:
        """
        Returns the production rate multiplier after rerouting jobs.
        Rerouting helps but has overhead — typically 85–95% of original rate.
        """
        if reroute_target not in active_machines:
            return 0.70   # bad reroute — no benefit

        target_spec = self.machine_specs.get(reroute_target)
        source_spec = self.machine_specs.get(broken_machine)

        if not target_spec or not source_spec:
            return 0.80

        # Compatibility check — same type = better reroute
        if target_spec["type"] == source_spec["type"]:
            return 0.92   # same machine type — good reroute
        else:
            return 0.78   # different type — partial reroute

    def safety_check(self,
                     active_machines: Dict[str, int],
                     workers_present: int) -> dict:
        """
        Check BIS IS-3696 safety compliance.
        Returns safety report with violations list.
        """
        violations = []
        warnings = []

        total_hazard_load = 0
        for machine_id, count in active_machines.items():
            spec = self.machine_specs.get(machine_id, {})
            hazard = spec.get("hazard_level", 1)
            ops_required = spec.get("operators_required", 1)

            # Worker-to-hazard ratio check (IS 3696 guideline)
            workers_at_station = ops_required * count
            safety_ratio = workers_at_station / (hazard * count)

            if hazard >= 3 and safety_ratio < 1.0:
                violations.append(
                    f"{machine_id}: hazard level {hazard} station "
                    f"understaffed (safety ratio {safety_ratio:.2f} < 1.0)"
                )

            total_hazard_load += hazard * count

        # Overall floor safety ratio
        floor_safety = workers_present / max(total_hazard_load, 1)
        if floor_safety < 0.8:
            warnings.append(
                f"Floor-wide safety ratio low: {floor_safety:.2f} "
                f"(recommend ≥ 0.8 per IS 3696)"
            )

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "floor_safety_ratio": round(floor_safety, 2)
        }
