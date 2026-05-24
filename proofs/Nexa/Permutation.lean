/-
  NexaCore — Permutation Invariants

  Formally verified properties of permutation operations on hypervectors.
  Permutations are used in HDC for sequence encoding and positional binding.
-/
import Nexa.HyperVector

namespace Nexa.Permutation

variable {w : Nat}

/-- Apply a permutation to an array of bits by reindexing. -/
def applyPerm (σ : HVPerm w) (v : Array α) (hv : v.size = w) : Array α :=
  Array.ofFn fun (i : Fin w) =>
    let j := σ.fwd i
    v[j.val]'(by omega)

/-- Apply the inverse permutation. -/
def applyInvPerm (σ : HVPerm w) (v : Array α) (hv : v.size = w) : Array α :=
  Array.ofFn fun (i : Fin w) =>
    let j := σ.rev i
    v[j.val]'(by omega)

/-- **Permutation inverse roundtrip**: applying a permutation then its inverse
    recovers the original vector. P⁻¹(P(v)) = v. -/
theorem perm_inverse_roundtrip (σ : HVPerm w) (v : Array α) (hv : v.size = w) :
    applyInvPerm σ (applyPerm σ v hv) (by simp [applyPerm, Array.size_ofFn]) =
    Array.ofFn fun (i : Fin w) => v[i.val]'(by omega) := by
  simp only [applyPerm, applyInvPerm]
  congr 1
  ext i
  simp only [Array.getElem_ofFn]
  congr 1
  exact congrArg Fin.val (σ.fwd_rev i)

/-- Forward permutation preserves array size. -/
theorem perm_preserves_size (σ : HVPerm w) (v : Array α) (hv : v.size = w) :
    (applyPerm σ v hv).size = w := by
  simp [applyPerm, Array.size_ofFn]

/-- Inverse permutation preserves array size. -/
theorem inv_perm_preserves_size (σ : HVPerm w) (v : Array α) (hv : v.size = w) :
    (applyInvPerm σ v hv).size = w := by
  simp [applyInvPerm, Array.size_ofFn]

/-- Composition of two permutations yields a valid permutation. -/
def composePerm (σ₁ σ₂ : HVPerm w) : HVPerm w where
  fwd := σ₂.fwd ∘ σ₁.fwd
  rev := σ₁.rev ∘ σ₂.rev
  fwd_rev i := by simp [Function.comp, σ₁.fwd_rev, σ₂.fwd_rev]
  rev_fwd i := by simp [Function.comp, σ₂.rev_fwd, σ₁.rev_fwd]

/-- The identity permutation. -/
def identityPerm (w : Nat) : HVPerm w where
  fwd := id
  rev := id
  fwd_rev _ := rfl
  rev_fwd _ := rfl

/-- Composing a permutation with its inverse yields identity behavior. -/
theorem compose_inverse_is_identity (σ : HVPerm w) :
    ∀ i : Fin w, (composePerm σ ⟨σ.rev, σ.fwd, σ.rev_fwd, σ.fwd_rev⟩).fwd i = i := by
  intro i
  simp [composePerm, Function.comp, σ.rev_fwd]

end Nexa.Permutation
