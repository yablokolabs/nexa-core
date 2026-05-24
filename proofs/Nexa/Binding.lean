/-
  NexaCore — XOR Binding Algebra

  Formally verified properties of the XOR binding operation used in
  hyperdimensional computing. XOR is the fundamental binding operator
  for binary hypervectors.
-/
import Nexa.HyperVector

namespace Nexa.Binding

variable {w : Nat}

/-- XOR binding is commutative: bind(A, B) = bind(B, A). -/
theorem bind_comm (a b : HyperVector w) : HyperVector.bind a b = HyperVector.bind b a := by
  simp [HyperVector.bind, BitVec.xor_comm]

/-- XOR binding is associative: bind(bind(A, B), C) = bind(A, bind(B, C)). -/
theorem bind_assoc (a b c : HyperVector w) :
    HyperVector.bind (HyperVector.bind a b) c = HyperVector.bind a (HyperVector.bind b c) := by
  simp [HyperVector.bind, BitVec.xor_assoc]

/-- XOR binding with zero is identity: bind(A, 0) = A. -/
theorem bind_zero (a : HyperVector w) : HyperVector.bind a (HyperVector.zero w) = a := by
  simp [HyperVector.bind, HyperVector.zero, BitVec.xor_zero]

/-- XOR binding with self cancels: bind(A, A) = 0. -/
theorem bind_self_cancel (a : HyperVector w) : HyperVector.bind a a = HyperVector.zero w := by
  simp [HyperVector.bind, HyperVector.zero, BitVec.xor_self]

/-- **Core HDC property**: XOR binding is reversible.
    unbind(bind(A, B), A) = B.
    This is the foundation of hyperdimensional symbolic retrieval. -/
theorem bind_unbind_reverse (a b : HyperVector w) :
    HyperVector.unbind (HyperVector.bind a b) a = b := by
  simp only [HyperVector.bind, HyperVector.unbind]
  rw [BitVec.xor_assoc, BitVec.xor_comm b a, ← BitVec.xor_assoc,
      BitVec.xor_self, BitVec.zero_xor]

/-- Symmetric unbinding: unbind(bind(A, B), B) = A. -/
theorem bind_unbind_reverse_sym (a b : HyperVector w) :
    HyperVector.unbind (HyperVector.bind a b) b = a := by
  simp only [HyperVector.bind, HyperVector.unbind, BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

/-- Double binding cancels: bind(bind(A, B), B) = A. -/
theorem double_bind_cancel (a b : HyperVector w) :
    HyperVector.bind (HyperVector.bind a b) b = a := by
  simp only [HyperVector.bind, BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

/-- Binding is an involution with respect to a fixed key:
    bind(bind(A, K), K) = A for any key K. -/
theorem bind_involution (a k : HyperVector w) :
    HyperVector.bind (HyperVector.bind a k) k = a := by
  exact double_bind_cancel a k

/-- **Role-filler binding**: In HDC, role-filler pairs are bound as R ⊕ F.
    Given a bound pair and either component, the other can be recovered.
    This theorem shows: if bound = bind(role, filler), then unbind(bound, role) = filler. -/
theorem role_filler_recovery (role filler : HyperVector w) :
    HyperVector.unbind (HyperVector.bind role filler) role = filler :=
  bind_unbind_reverse role filler

end Nexa.Binding
