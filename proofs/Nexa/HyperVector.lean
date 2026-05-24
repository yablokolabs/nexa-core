/-
  NexaCore — HyperVector Type Definitions

  Core type modeling for the formal verification layer.
  Binary hypervectors are modeled as BitVec n, where n is the dimensionality.
-/

namespace Nexa

/-- A binary hypervector of width `w` bits. -/
abbrev HyperVector (w : Nat) := BitVec w

/-- The zero hypervector (all bits clear). -/
def HyperVector.zero (w : Nat) : HyperVector w := 0#w

/-- XOR binding of two hypervectors. -/
def HyperVector.bind {w : Nat} (a b : HyperVector w) : HyperVector w := a ^^^ b

/-- Unbinding via XOR (same operation as bind — XOR is self-inverse). -/
def HyperVector.unbind {w : Nat} (bound key : HyperVector w) : HyperVector w := bound ^^^ key

/-- Bundling (majority vote) of a list of hypervectors, approximated via bitwise OR for two vectors.
    Full majority-vote bundling is a runtime operation; here we model the algebraic identity cases. -/
def HyperVector.bundle2 {w : Nat} (a b : HyperVector w) : HyperVector w := a ||| b

/-- Hamming weight (population count) of a hypervector. -/
def HyperVector.weight {w : Nat} (v : HyperVector w) : BitVec w := BitVec.cpop v

/-- Hamming distance between two hypervectors = popcount of their XOR. -/
def HyperVector.hammingDist {w : Nat} (a b : HyperVector w) : BitVec w :=
  BitVec.cpop (a ^^^ b)

/-- A permutation on hypervector indices, represented as a bijective function on Fin w. -/
structure HVPerm (w : Nat) where
  fwd : Fin w → Fin w
  rev : Fin w → Fin w
  fwd_rev : ∀ i, fwd (rev i) = i
  rev_fwd : ∀ i, rev (fwd i) = i

/-- A deterministic encoder maps values of type α to hypervectors. -/
structure Encoder (α : Type) (w : Nat) where
  encode : α → HyperVector w
  decode : HyperVector w → α
  roundtrip : ∀ x, decode (encode x) = x

end Nexa
