from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import sys
import random

# ============================
# 1. Model class: mdp
# ============================

class mdp:
    """
    Data structure for MC / MDP models:
    - states: list of states
    - actions: list of actions (can be empty for MC)
    - transitions: dict (state, action) -> [(next_state, weight or prob), ...]
      Convention: for MC (no actions), use action=None
    - use_actions / use_no_actions: used to detect mixed MC/MDP models
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.transitions = {}      # { (state, action) : [(next_state, weight), ...] }
        self.use_actions = False
        self.use_no_actions = False

    # --------- Interfaces to fill the model ---------

    def add_states(self, states):
        self.states = states

    def add_actions(self, actions):
        self.actions = actions

    def add_trans_with_action(self, dep, act, targets, weights):
        """
        Transition with action:
        dep [act] -> w1: t1 + w2: t2 + ...
        """
        self.use_actions = True
        key = (dep, act)
        if key not in self.transitions:
            self.transitions[key] = []
        for t, w in zip(targets, weights):
            self.transitions[key].append((t, w))

    def add_trans_without_action(self, dep, targets, weights):
        """
        Transition without action (MC):
        dep -> w1: t1 + w2: t2 + ...
        Convention: action = None
        """
        self.use_no_actions = True
        key = (dep, None)
        if key not in self.transitions:
            self.transitions[key] = []
        for t, w in zip(targets, weights):
            self.transitions[key].append((t, w))

    # --------- Tools: normalization + model checking ---------

    def normalize(self):
        """
        Convert weights into real probabilities:
        weight / sum(weight)
        """
        new_trans = {}
        for key, lst in self.transitions.items():
            total = sum(w for (_, w) in lst)
            if total <= 0:
                continue
            new_trans[key] = [(s, w / total) for (s, w) in lst]
        self.transitions = new_trans

    def check(self):
        """
        Basic model validation:
        - All states / targets must be declared
        - Mixed MC/MDP transitions are not allowed (warning)
        """
        ok = True

        # 1) Check if all states are declared
        for (s, a), lst in self.transitions.items():
            if s not in self.states:
                print("Error: state", s, "not declared in States")
                ok = False
            for (t, _) in lst:
                if t not in self.states:
                    print("Error: target state", t, "not declared in States")
                    ok = False

        # 2) Mixed transitions warning
        if self.use_actions and self.use_no_actions:
            print("Error: model mixes transitions with and without actions")
            ok = False

        return ok

    def get_absorbing_states(self):
        """
        Automatically detect absorbing states:
        A state is absorbing if its only transition is to itself with prob = 1
        (requires calling normalize() first)
        """
        absorbing = set()
        for (s, a), lst in self.transitions.items():
            if len(lst) == 1:
                t, p = lst[0]
                if t == s and abs(p - 1.0) < 1e-12:
                    absorbing.add(s)
        return absorbing

    # --------- Simple MC / MDP simulation ---------

    def simulate_mc(self, init_state, horizon=10):
        """
        MC simulation: assumes action=None transitions
        Returns a list of states visited
        """
        path = [init_state]
        current = init_state

        for _ in range(horizon):
            key = (current, None)
            if key not in self.transitions:
                break
            targets, probs = zip(*self.transitions[key])
            next_state = random.choices(targets, weights=probs, k=1)[0]
            path.append(next_state)
            current = next_state

        return path

    def simulate_mdp(self, init_state, policy=None, horizon=10):
        """
        MDP simulation with optional policy(state) -> action
        If policy=None, choose a random available action
        """
        current = init_state
        path = [init_state]

        for _ in range(horizon):
            available = [(s, a) for (s, a) in self.transitions.keys()
                         if s == current and a is not None]
            if not available:
                break

            if policy is None:
                _, action = random.choice(available)
            else:
                action = policy(current)

            key = (current, action)
            targets, probs = zip(*self.transitions[key])
            next_state = random.choices(targets, weights=probs, k=1)[0]

            path.append((action, next_state))
            current = next_state

        return path

    # --------- General simulation (Scheme C + automatic absorbing states) ---------

    def simulate(self, init_state, policy=None, horizon=10, terminal_states=None):
        """
        General simulation (Scheme C):
        - If transitions with actions exist in current state → treat as MDP
        - Else if (state, None) exists → treat as MC
        - Else stop

        Extra:
        - If terminal_states=None → automatically use absorbing states
        - Otherwise use user-provided terminal state set

        Returns: (path, path_probability)
        """

        if terminal_states is None:
            terminal_states = self.get_absorbing_states()
        else:
            terminal_states = set(terminal_states)

        current = init_state
        path = [init_state]
        path_prob = 1.0   # Initial probability

        for _ in range(horizon):
            if current in terminal_states:
                break

            available_actions = [a for (s, a) in self.transitions.keys()
                                 if s == current and a is not None]

            if available_actions:
                if policy is None:
                    action = random.choice(available_actions)
                else:
                    action = policy(current)
                    if (current, action) not in self.transitions:
                        action = random.choice(available_actions)

                key = (current, action)

            elif (current, None) in self.transitions:
                key = (current, None)

            else:
                break

            targets, probs = zip(*self.transitions[key])
            next_state = random.choices(targets, weights=probs, k=1)[0]

            prob = probs[targets.index(next_state)]
            path_prob *= prob

            path.append(next_state)
            current = next_state

        return path, path_prob


# ============================
# 2. Listener that builds the model
# ============================

class MDPBuilderListener(gramListener):
    """
    Listener that fills the mdp object instead of printing
    """
    def __init__(self):
        self.model = mdp()

    def enterDefstates(self, ctx):
        states = [str(x) for x in ctx.ID()]
        self.model.add_states(states)

    def enterDefactions(self, ctx):
        actions = [str(x) for x in ctx.ID()]
        self.model.add_actions(actions)

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        targets = ids
        weights = [int(str(x)) for x in ctx.INT()]
        self.model.add_trans_with_action(dep, act, targets, weights)

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        targets = ids
        weights = [int(str(x)) for x in ctx.INT()]
        self.model.add_trans_without_action(dep, targets, weights)


# ============================
# 3. Original print listener (debug)
# ============================

class gramPrintListener(gramListener):

    def __init__(self):
        pass
        
    def enterDefstates(self, ctx):
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))


# ============================
# 4. main: parse + build + test
# ============================

def main():
    lexer = gramLexer(StdinStream())
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()

    builder = MDPBuilderListener()
    walker = ParseTreeWalker()
    walker.walk(builder, tree)

    model = builder.model

    model.normalize()
    ok = model.check()
    if not ok:
        print("Model has errors, please check above messages.")

    print("=== States ===")
    print(model.states)
    print("=== Actions ===")
    print(model.actions)
    print("=== Transitions (after normalization) ===")
    for key, lst in model.transitions.items():
        print(key, "->", lst)

    if model.states:
        init = model.states[0]
        print("\nGeneral simulation from", init)
        path, prob = model.simulate(init_state=init, horizon=10)
        print("Path:", path)
        print("Path Probability:", prob)

if __name__ == '__main__':
    main()
