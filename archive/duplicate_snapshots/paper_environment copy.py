
import numpy as np


class CircularSupplyChainDT:
    """
    Simulation-based circular supply-chain digital twin.

    Five action groups:
    1. Planning
    2. Inventory
    3. Logistics
    4. Expediting
    5. Recycling

    This is a paper-inspired implementation with synthetic data.
    Some simulation parameters and reward normalizers are provisional.
    """

    def __init__(self, episode_length=60, seed=None):
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # Capacities
        self.production_capacity = 520.0
        self.transport_capacity = 600.0
        self.recycling_capacity = 80.0

        # Initial conditions
        self.initial_inventory = 500.0
        self.nominal_inventory_target = 600.0
        self.service_window = 2

        # Cost parameters
        self.production_cost = 10.0
        self.raw_material_cost = 2.0
        self.holding_cost_rate = 0.5
        self.backlog_cost_rate = 2.0

        self.distance_km = 120.0
        self.transport_cost_per_unit_km = 0.06
        self.transport_fixed_cost = 40.0
        self.expedite_surcharge = 3.0
        self.recycling_cost_rate = 0.2

        # Energy and emissions
        self.energy_per_unit = 1.1
        self.base_energy = 200.0
        self.energy_price = 1.0
        self.process_emission_factor = 0.45
        self.transport_emission_factor = 0.06
        self.recycling_emission_saving = 0.3

        self.reset(seed=seed)

    def reset(self, seed=None, scenario=None):
        """Reset the simulation and optionally apply stress scenarios."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.day = 0
        self.inventory = float(self.initial_inventory)
        self.recovered_material = 0.0

        self.orders = []
        self.shipment_queue = []
        self.next_order_id = 1

        self.total_demand = 0.0
        self.total_delivered = 0.0
        self.total_on_time_units = 0.0
        self.total_orders = 0
        self.total_completed_orders = 0
        self.total_otif_orders = 0

        self.total_cost = 0.0
        self.total_emissions = 0.0
        self.total_waste = 0.0
        self.total_energy = 0.0

        self.demand_uplift = 0.0
        self.transport_delay_multiplier = 1.0
        self.energy_cost_multiplier = 1.0

        if scenario:
            self.demand_uplift = float(
                scenario.get("demand", 0.0)
            )
            self.transport_delay_multiplier = float(
                scenario.get("transport", 1.0)
            )
            self.energy_cost_multiplier = float(
                scenario.get("energy", 1.0)
            )

            if self.demand_uplift < 0:
                raise ValueError("Demand uplift cannot be negative.")
            if self.transport_delay_multiplier < 1:
                raise ValueError(
                    "Transport multiplier must be at least 1."
                )
            if self.energy_cost_multiplier < 1:
                raise ValueError(
                    "Energy multiplier must be at least 1."
                )

        return self.get_state()

    def get_state(self):
        """Return the observable simulation state."""
        backlog = sum(
            order["remaining"] for order in self.orders
        )

        return {
            "day": self.day,
            "inventory": round(self.inventory, 2),
            "backlog": round(backlog, 2),
            "recovered_material": round(
                self.recovered_material, 2
            ),
            "pending_shipments": len(self.shipment_queue),
            "production_capacity": self.production_capacity,
            "transport_capacity": self.transport_capacity,
            "recycling_capacity": self.recycling_capacity,
            "demand_uplift": self.demand_uplift,
            "transport_delay_multiplier":
                self.transport_delay_multiplier,
            "energy_cost_multiplier": self.energy_cost_multiplier,
        }

    def _deliver_arriving_shipments(self):
        """Deliver shipments whose transit time has elapsed."""
        arriving_units = 0.0
        on_time_units = 0.0
        remaining_shipments = []

        for shipment in self.shipment_queue:
            shipment["days_remaining"] -= 1

            if shipment["days_remaining"] > 0:
                remaining_shipments.append(shipment)
                continue

            for allocation in shipment["allocations"]:
                order = allocation["order"]
                quantity = allocation["quantity"]

                order["remaining"] = max(
                    0.0, order["remaining"] - quantity
                )
                order["delivered"] += quantity
                arriving_units += quantity

                if self.day <= order["due_day"]:
                    on_time_units += quantity
                    order["on_time_delivered"] += quantity

                if (
                    order["remaining"] <= 1e-8
                    and not order["counted_complete"]
                ):
                    order["counted_complete"] = True
                    self.total_completed_orders += 1

                    if self.day <= order["due_day"]:
                        self.total_otif_orders += 1

        self.shipment_queue = remaining_shipments
        return arriving_units, on_time_units

    def _create_order(self, demand):
        """Create a customer order with a defined due day."""
        order = {
            "id": self.next_order_id,
            "quantity": float(demand),
            "remaining": float(demand),
            "unshipped": float(demand),
            "delivered": 0.0,
            "on_time_delivered": 0.0,
            "due_day": self.day + self.service_window,
            "counted_complete": False,
        }

        self.orders.append(order)
        self.next_order_id += 1
        self.total_orders += 1

    def _dispatch_orders(self, dispatch_capacity, expedite_level):
        """Ship available stock to outstanding orders in FIFO order."""
        dispatch_capacity = min(
            float(dispatch_capacity),
            self.transport_capacity,
            self.inventory,
        )

        remaining_capacity = dispatch_capacity
        dispatched = 0.0
        allocations = []

        for order in self.orders:
            if remaining_capacity <= 1e-8:
                break

            if order["unshipped"] <= 1e-8:
                continue

            quantity = min(
                order["unshipped"], remaining_capacity
            )

            if quantity <= 0:
                continue

            allocations.append({
                "order": order,
                "quantity": quantity,
            })

            order["unshipped"] -= quantity
            remaining_capacity -= quantity
            dispatched += quantity

        if dispatched <= 0:
            return 0.0, 0.0, 0.0

        self.inventory = max(
            0.0, self.inventory - dispatched
        )

        # Illustrative baseline lognormal transit parameters.
        baseline_transit = self.rng.lognormal(
            mean=np.log(2.0),
            sigma=0.35,
        )

        shocked_transit = (
            baseline_transit
            * self.transport_delay_multiplier
        )

        expedite_factors = [1.0, 1.3, 1.5]
        expedite_factor = expedite_factors[expedite_level]

        transit_days = max(
            1,
            int(np.ceil(shocked_transit / expedite_factor)),
        )

        self.shipment_queue.append({
            "days_remaining": transit_days,
            "units": dispatched,
            "allocations": allocations,
        })

        expedite_cost = (
            dispatched * self.expedite_surcharge
            if expedite_level > 0
            else 0.0
        )

        return dispatched, float(transit_days), expedite_cost

    def _calculate_reward(self, metrics):
        """
        Calculate a provisional shared reward.

        Weights follow those reported in the paper. Normalization
        scales are initial estimates, not calibrated baseline values.
        """
        production = max(metrics["production_input"], 1.0)

        cost_per_unit = metrics["operating_cost"] / production
        energy_per_unit = metrics["energy_used"] / production

        normalized = {
            "co2": min(metrics["emissions"] / 5000.0, 2.0),
            "cost": min(metrics["operating_cost"] / 15000.0, 2.0),
            "inventory": min(metrics["inventory"] / 600.0, 2.0),
            "lead_time": min(metrics["transit_days"] / 3.0, 2.0),
            "waste": min(metrics["net_waste"] / 20.0, 2.0),
            "backlog": min(metrics["backlog"] / 1500.0, 2.0),
            "cost_per_unit": min(cost_per_unit / 30.0, 2.0),
            "energy_per_unit": min(energy_per_unit / 2.0, 2.0),
        }

        reward = -(
            1.1 * normalized["co2"]
            + 1.2 * normalized["cost"]
            + 0.5 * normalized["inventory"]
            + 0.7 * normalized["lead_time"]
            + 1.2 * normalized["waste"]
            + 0.6 * normalized["backlog"]
            + 1.0 * normalized["cost_per_unit"]
            + 0.6 * normalized["energy_per_unit"]
        )

        # Service-level and OTIF bonuses.
        if metrics["service_level"] >= 0.95:
            reward += 0.5

        if metrics["otif"] >= 0.95:
            reward += 0.5

        # Provisional recycling bonus.
        gross_scrap = metrics["gross_scrap"]
        if gross_scrap > 0:
            recovery_rate = metrics["recovered"] / gross_scrap
            if recovery_rate >= 0.4:
                reward += 0.25

        return float(reward)

    def step(self, actions):
        """Advance the simulation by one day."""
        required = {
            "planning",
            "inventory",
            "logistics",
            "expediting",
            "recycling",
        }

        if not isinstance(actions, dict) or set(actions) != required:
            raise ValueError(
                "Actions must contain exactly: planning, inventory, "
                "logistics, expediting, recycling."
            )

        for name, value in actions.items():
            if value not in (0, 1, 2):
                raise ValueError(
                    f"Action '{name}' must be 0, 1, or 2."
                )

        if self.day >= self.episode_length:
            raise RuntimeError(
                "Episode finished. Call reset() before stepping."
            )

        self.day += 1

        # 1. Deliver previously dispatched customer orders.
        arrivals, on_time_units = (
            self._deliver_arriving_shipments()
        )
        self.total_delivered += arrivals
        self.total_on_time_units += on_time_units

        # 2. Generate demand using synthetic overdispersed demand.
        demand = float(
            self.rng.poisson(
                500.0 * (1.0 + self.demand_uplift)
            )
        )
        demand *= float(
            self.rng.lognormal(mean=0.0, sigma=0.10)
        )
        self.total_demand += demand

        if demand > 0:
            self._create_order(demand)

        # 3. Production and use of recovered material.
        production_multipliers = [0.9, 1.0, 1.1]
        production_input = (
            self.production_capacity
            * production_multipliers[actions["planning"]]
        )

        recovered_used = min(
            self.recovered_material, production_input
        )
        self.recovered_material -= recovered_used
        virgin_material_used = production_input - recovered_used

        yield_rate = float(self.rng.uniform(0.92, 0.98))
        usable_output = production_input * yield_rate
        gross_scrap = production_input - usable_output

        self.inventory += usable_output

        # 4. Inventory target action.
        target_adjustments = [-100.0, 0.0, 100.0]
        inventory_target = (
            self.nominal_inventory_target
            + target_adjustments[actions["inventory"]]
        )

        # 5. Recycling and recovery of scrap.
        recovery_fractions = [0.2, 0.4, 0.6]
        requested_recovery = (
            gross_scrap
            * recovery_fractions[actions["recycling"]]
        )
        recovered = min(
            requested_recovery, self.recycling_capacity
        )

        net_waste = max(0.0, gross_scrap - recovered)
        self.recovered_material += recovered
        self.total_waste += net_waste

        # 6. Dispatch products to customer orders.
        dispatch_fractions = [0.75, 0.90, 1.0]
        dispatch_capacity = (
            self.transport_capacity
            * dispatch_fractions[actions["logistics"]]
        )

        dispatched, transit_days, expedite_cost = (
            self._dispatch_orders(
                dispatch_capacity,
                actions["expediting"],
            )
        )

        # 7. Operating costs.
        production_cost = production_input * self.production_cost
        material_cost = (
            virgin_material_used * self.raw_material_cost
        )
        holding_cost = self.inventory * self.holding_cost_rate

        backlog = sum(
            order["remaining"] for order in self.orders
        )
        backlog_cost = backlog * self.backlog_cost_rate

        transport_cost = 0.0
        transport_emissions = 0.0

        if dispatched > 0:
            transport_cost = (
                self.transport_cost_per_unit_km
                * self.distance_km
                * dispatched
                + self.transport_fixed_cost
                + expedite_cost
            )

            transport_emissions = (
                self.transport_emission_factor
                * self.distance_km
                * dispatched
            )

        energy_used = (
            self.energy_per_unit * production_input
            + self.base_energy
        )

        energy_cost = (
            energy_used
            * self.energy_price
            * self.energy_cost_multiplier
        )

        process_emissions = (
            energy_used * self.process_emission_factor
        )

        recycling_cost = recovered * self.recycling_cost_rate
        emission_savings = (
            recovered * self.recycling_emission_saving
        )

        emissions = max(
            0.0,
            process_emissions
            + transport_emissions
            - emission_savings,
        )

        operating_cost = (
            production_cost
            + material_cost
            + holding_cost
            + backlog_cost
            + transport_cost
            + energy_cost
            + recycling_cost
        )

        self.total_cost += operating_cost
        self.total_emissions += emissions
        self.total_energy += energy_used

        # Cumulative on-time unit fill rate and order-level OTIF.
        service_level = (
            self.total_on_time_units / self.total_demand
            if self.total_demand > 0
            else 0.0
        )

        otif = (
            self.total_otif_orders / self.total_orders
            if self.total_orders > 0
            else 0.0
        )

        # 8. Return metrics with compatibility aliases for the test.
        metrics = {
            "day": self.day,
            "demand": round(demand, 2),
            "production_input": round(production_input, 2),
            "production": round(usable_output, 2),
            "yield_rate": round(yield_rate, 4),
            "usable_output": round(usable_output, 2),
            "gross_scrap": round(gross_scrap, 2),
            "virgin_material_used": round(
                virgin_material_used, 2
            ),
            "recovered_material_used": round(
                recovered_used, 2
            ),
            "recovered": round(recovered, 2),
            "recovered_material": round(
                self.recovered_material, 2
            ),
            "dispatched": round(dispatched, 2),
            "arrivals": round(arrivals, 2),
            "delivered": round(arrivals, 2),
            "pending_shipments": len(self.shipment_queue),
            "transit_days": transit_days,
            "inventory": round(self.inventory, 2),
            "backlog": round(backlog, 2),
            "inventory_target": inventory_target,
            "service_level": round(service_level, 4),
            "otif": round(otif, 4),
            "completed_orders": self.total_completed_orders,
            "otif_orders": self.total_otif_orders,
            "net_waste": round(net_waste, 2),
            "energy_used": round(energy_used, 2),
            "operating_cost": round(operating_cost, 2),
            "cumulative_cost": round(self.total_cost, 2),
            "emissions": round(emissions, 2),
            "co2_emissions": round(emissions, 2),
            "cumulative_emissions": round(
                self.total_emissions, 2
            ),
            "demand_uplift": self.demand_uplift,
            "transport_delay_multiplier":
                self.transport_delay_multiplier,
            "energy_cost_multiplier": self.energy_cost_multiplier,
        }

        reward = self._calculate_reward(metrics)
        metrics["reward"] = round(reward, 4)

        done = self.day >= self.episode_length
        return self.get_state(), metrics, done
