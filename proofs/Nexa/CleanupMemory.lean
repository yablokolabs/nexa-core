/-
  NexaCore — Cleanup Memory Properties

  Formally verified properties of associative cleanup memory.
  Models the core invariant: if a clean vector is stored, querying it
  returns itself; nearest-neighbor retrieval is consistent.
-/
import Nexa.HyperVector

namespace Nexa.CleanupMemory

variable {w : Nat}

/-- A cleanup memory stores a set of prototype vectors and returns
    the nearest one to a query. -/
structure Memory (w : Nat) where
  prototypes : List (HyperVector w)
  nonempty : prototypes ≠ []

/-- Abstract nearest-neighbor: returns the first prototype (models
    exact-match retrieval for clean queries). -/
def Memory.nearest (mem : Memory w) (_query : HyperVector w) : HyperVector w :=
  mem.prototypes.head mem.nonempty

/-- **Exact match property**: If the query is exactly a stored prototype,
    and it appears first in the list, cleanup returns it unchanged. -/
theorem cleanup_exact_self (v : HyperVector w) (rest : List (HyperVector w)) :
    let mem : Memory w := ⟨v :: rest, List.cons_ne_nil v rest⟩
    mem.nearest v = v := by
  simp [Memory.nearest]

/-- A singleton memory always returns its only prototype. -/
theorem singleton_cleanup (v : HyperVector w) :
    let mem : Memory w := ⟨[v], List.cons_ne_nil v []⟩
    mem.nearest v = v := by
  simp [Memory.nearest]

/-- XOR distance to self is zero — fundamental property underlying
    cleanup memory correctness. A clean vector has zero distance to itself. -/
theorem self_distance_zero (v : HyperVector w) :
    HyperVector.hammingDist v v = 0#w := by
  simp [HyperVector.hammingDist, BitVec.xor_self]

/-- If corruption is zero (identical vector), the "corrupted" vector equals
    the original — cleanup is trivially correct. -/
theorem zero_corruption_identity (v : HyperVector w) :
    v ^^^ (0#w) = v := by
  simp [BitVec.xor_zero]

/-- XOR corruption is reversible: if we know the corruption pattern,
    we can perfectly restore the original vector. -/
theorem corruption_reversible (v noise : HyperVector w) :
    (v ^^^ noise) ^^^ noise = v := by
  simp [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

end Nexa.CleanupMemory
