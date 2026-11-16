import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_memory import Store, add_extracted, plan_brief

EPISODES = [
    {"start_place": "Hall", "start_have": [],
     "links": [{"from": "Hall", "action": "north", "to": "Cellar", "have": [], "do": [], "steps": [3]},
               {"from": "Cellar", "action": "south", "to": "Hall", "have": [], "do": [], "steps": [5]}],
     "things": [{"thing": "lamp", "place": "Hall", "in": "", "got_by": "take lamp", "have": [], "do": [],
                 "steps": [1]},
                {"thing": "iron key", "place": "Cellar", "in": "", "got_by": "take iron key", "have": [], "do": [],
                 "steps": [4]}],
     "acts": [{"action": "light lamp", "place": "Hall", "have": ["lamp"], "steps": [2]}],
     "enter": [{"place": "Cellar", "have": ["lamp"], "do": ["light lamp"], "steps": [3]}]},
    {"start_place": "Hall", "start_have": [],
     "links": [{"from": "Hall", "action": "east", "to": "Study", "have": [], "do": ["open door"], "steps": [2]}],
     "things": [{"thing": "gem", "place": "Study", "in": "chest", "got_by": "take gem", "have": [],
                 "do": ["unlock chest with iron key", "open chest"], "steps": [5]}],
     "acts": [{"action": "open door", "place": "Hall", "have": [], "steps": [1]},
              {"action": "unlock chest with iron key", "place": "Study", "have": ["iron key"], "steps": [3]}],
     "enter": []},
]


def main():
    store = Store()
    for i, facts in enumerate(EPISODES):
        add_extracted(store, facts, f"episode {i + 1}")
    print("Goal: get the gem\n")
    print("\n".join(plan_brief(store, "Get the gem.")))


if __name__ == "__main__":
    main()
