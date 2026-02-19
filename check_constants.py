"""Check SUMO TraCI constants"""
import sys

try:
    import traci
    print("Using traci")
except ImportError as e:
    print(f"traci not available: {e}")
    sys.exit(1)

# Find LAST_STEP constants
print("\n=== LAST_STEP constants ===")
for attr in dir(traci.constants):
    if 'LAST_STEP' in attr:
        print(f"  traci.constants.{attr}")

# Find VAR_ constants
print("\n=== VAR_ constants ===")
var_consts = [attr for attr in dir(traci.constants) if attr.startswith('VAR_')]
for attr in sorted(var_consts):
    print(f"  traci.constants.{attr}")

print(f"\nTotal VAR_ constants: {len(var_consts)}")
