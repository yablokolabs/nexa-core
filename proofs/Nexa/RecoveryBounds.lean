/-
  NexaCore — Recovery Bounds

  Formally verified bounds on corruption tolerance and recovery properties.
  Establishes that XOR-based corruption is bounded and reversible when the
  corruption pattern is known or constrained.
-/
import Nexa.HyperVector
import Nexa.CleanupMemory

namespace Nexa.RecoveryBounds

variable {w : Nat}

/-- **Zero corruption identity**: A vector with no corruption is unchanged. -/
theorem zero_corruption_identity (v : HyperVector w) :
    v ^^^ (0#w) = v := by
  simp [BitVec.xor_zero]

/-- **Known corruption is perfectly reversible**: If the corruption pattern
    (noise vector) is known, XORing again perfectly recovers the original. -/
theorem known_corruption_recovery (original noise : HyperVector w) :
    (original ^^^ noise) ^^^ noise = original := by
  simp [BitVec.xor_assoc, BitVec.xor_self, BitVec.xor_zero]

/-- **Double corruption cancellation**: Applying the same corruption twice
    restores the original vector (XOR is self-inverse). -/
theorem double_corruption_cancels (v corruption : HyperVector w) :
    (v ^^^ corruption) ^^^ corruption = v :=
  known_corruption_recovery v corruption

/-- **Corruption commutativity**: The order of corruption application
    doesn't matter — both produce the same corrupted vector. -/
theorem corruption_commutative (v noise₁ noise₂ : HyperVector w) :
    v ^^^ noise₁ ^^^ noise₂ = v ^^^ noise₂ ^^^ noise₁ := by
  rw [BitVec.xor_assoc, BitVec.xor_comm noise₁ noise₂, ← BitVec.xor_assoc]

/-- **Corruption composition**: Two sequential corruptions are equivalent
    to a single corruption with the XOR of both noise vectors. -/
theorem corruption_composition (v noise₁ noise₂ : HyperVector w) :
    (v ^^^ noise₁) ^^^ noise₂ = v ^^^ (noise₁ ^^^ noise₂) := by
  exact BitVec.xor_assoc v noise₁ noise₂

/-- **Distance from corruption**: The Hamming distance between the original
    vector and its corrupted version equals the weight of the noise.
    d(v, v ⊕ noise) = popcount(noise). -/
theorem corruption_distance_equals_noise_weight (v noise : HyperVector w) :
    HyperVector.hammingDist v (v ^^^ noise) = HyperVector.weight noise := by
  simp [HyperVector.hammingDist, HyperVector.weight]
  congr 1
  rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]

/-- **Corruption triangle**: The distance between two corrupted versions
    of the same vector is bounded by the XOR of their noise patterns. -/
theorem corruption_triangle (v noise₁ noise₂ : HyperVector w) :
    HyperVector.hammingDist (v ^^^ noise₁) (v ^^^ noise₂) =
    BitVec.cpop (noise₁ ^^^ noise₂) := by
  simp only [HyperVector.hammingDist]
  congr 1
  rw [BitVec.xor_assoc]
  rw [BitVec.xor_comm noise₁ (v ^^^ noise₂)]
  rw [BitVec.xor_assoc v noise₂ noise₁]
  rw [← BitVec.xor_assoc v v]
  rw [BitVec.xor_self, BitVec.zero_xor, BitVec.xor_comm noise₂ noise₁]

end Nexa.RecoveryBounds
