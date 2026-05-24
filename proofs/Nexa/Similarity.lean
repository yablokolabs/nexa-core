/-
  NexaCore — Similarity Metric Properties

  Formally verified properties of the Hamming distance metric
  used for hypervector similarity measurement.
-/
import Nexa.HyperVector

namespace Nexa.Similarity

variable {w : Nat}

/-- Hamming distance of a vector with itself is zero: d(x, x) = 0. -/
theorem hamming_self_zero (v : HyperVector w) :
    HyperVector.hammingDist v v = 0#w := by
  simp [HyperVector.hammingDist, BitVec.xor_self]

/-- Hamming distance is symmetric: d(x, y) = d(y, x). -/
theorem hamming_symmetric (a b : HyperVector w) :
    HyperVector.hammingDist a b = HyperVector.hammingDist b a := by
  simp [HyperVector.hammingDist, BitVec.xor_comm]

/-- XOR with zero has popcount equal to the vector's own popcount.
    d(x, 0) = popcount(x). -/
theorem hamming_from_zero (v : HyperVector w) :
    HyperVector.hammingDist v (HyperVector.zero w) = HyperVector.weight v := by
  simp [HyperVector.hammingDist, HyperVector.zero, HyperVector.weight, BitVec.xor_zero]

/-- Distance between a vector and its complement is maximal.
    Every bit differs, so XOR produces all-ones. -/
theorem hamming_complement_xor (v : HyperVector w) :
    v ^^^ ~~~v = BitVec.allOnes w := by
  ext i
  simp [BitVec.getElem_xor, BitVec.getElem_not, BitVec.getElem_allOnes]

/-- **Binding preserves distance structure**: Hamming distance is invariant under
    XOR with a fixed key. d(a ⊕ k, b ⊕ k) = d(a, b).
    This is critical for HDC — binding doesn't distort similarity relationships. -/
theorem hamming_bind_invariant (a b k : HyperVector w) :
    HyperVector.hammingDist (HyperVector.bind a k) (HyperVector.bind b k) =
    HyperVector.hammingDist a b := by
  simp only [HyperVector.hammingDist, HyperVector.bind]
  congr 1
  rw [BitVec.xor_assoc, BitVec.xor_comm k (b ^^^ k),
      BitVec.xor_assoc b k k, BitVec.xor_self, BitVec.xor_zero]

end Nexa.Similarity
