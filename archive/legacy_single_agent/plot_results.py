
import pandas as pd
import matplotlib.pyplot as plt

# Read results from all 30 evaluation episodes.
df = pd.read_csv("evaluation_all_runs.csv")

# Calculate average metrics for each strategy.
summary = df.groupby("strategy", sort=False).agg(
    service_level=("service_level", "mean"),
    operating_cost=("operating_cost", "mean"),
    co2_emissions=("co2_emissions", "mean"),
    ending_inventory=("ending_inventory", "mean"),
    total_reward=("total_reward", "mean")
).reset_index()

# Graph 1: Average operating cost.
plt.figure()
plt.bar(summary["strategy"], summary["operating_cost"])
plt.title("Average Operating Cost Across 30 Episodes")
plt.xlabel("Production strategy")
plt.ylabel("Total operating cost per episode")
plt.tight_layout()
plt.savefig("operating_cost_comparison.png", dpi=300)
plt.show()

# Graph 2: Average CO2 emissions.
plt.figure()
plt.bar(summary["strategy"], summary["co2_emissions"])
plt.title("Average CO2 Emissions Across 30 Episodes")
plt.xlabel("Production strategy")
plt.ylabel("CO2 emissions per episode")
plt.tight_layout()
plt.savefig("emissions_comparison.png", dpi=300)
plt.show()

# Graph 3: Average service level.
plt.figure()
plt.bar(summary["strategy"], summary["service_level"] * 100)
plt.title("Average Service Level Across 30 Episodes")
plt.xlabel("Production strategy")
plt.ylabel("Service level (%)")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig("service_level_comparison.png", dpi=300)
plt.show()

# Graph 4: Average ending inventory.
plt.figure()
plt.bar(summary["strategy"], summary["ending_inventory"])
plt.title("Average Ending Inventory Across 30 Episodes")
plt.xlabel("Production strategy")
plt.ylabel("Ending inventory (units)")
plt.tight_layout()
plt.savefig("inventory_comparison.png", dpi=300)
plt.show()

print("All four graphs saved successfully.")
