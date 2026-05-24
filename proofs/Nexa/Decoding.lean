/-
  NexaCore — Decoding Correctness

  Formally verified properties of the decoder subsystem.
  Proves exact decoding, symbolic unbinding, and structural properties.
-/
import Nexa.HyperVector
import Nexa.Encoding
import Nexa.Binding

namespace Nexa.Decoding

variable {α : Type} {w : Nat}

/-- **Exact decoder is a left-inverse of the encoder.**
    For any well-formed Encoder, the decode function recovers the original input. -/
theorem exact_decode_inverse (enc : Encoder α w) (x : α) :
    enc.decode (enc.encode x) = x :=
  enc.roundtrip x

/-- **Symbolic unbinding recovers bound components.**
    If we bind two symbols and unbind with the first, we recover the second. -/
theorem symbolic_unbind_recovers (a b : HyperVector w) :
    HyperVector.unbind (HyperVector.bind a b) a = b :=
  Binding.bind_unbind_reverse a b

/-- Symmetric unbinding: unbinding with the second component recovers the first. -/
theorem symbolic_unbind_recovers_sym (a b : HyperVector w) :
    HyperVector.unbind (HyperVector.bind a b) b = a :=
  Binding.bind_unbind_reverse_sym a b

/-- **Nested binding decoding**: For a structure A ⊕ (B ⊕ C), we can recover
    components by sequential unbinding. -/
theorem nested_unbind (a b c : HyperVector w) :
    HyperVector.unbind (HyperVector.unbind (HyperVector.bind a (HyperVector.bind b c)) a) b = c := by
  rw [Binding.bind_unbind_reverse]
  exact Binding.bind_unbind_reverse b c

/-- **Decode after re-encode is identity**: encoding, decoding, then re-encoding
    yields the same vector. -/
theorem reencode_identity (enc : Encoder α w) (x : α) :
    enc.encode (enc.decode (enc.encode x)) = enc.encode x := by
  rw [enc.roundtrip]

/-- The exact decoder applied to an uncorrupted vector produces a value that
    re-encodes to the same vector (no information loss). -/
theorem decode_encode_fixpoint (enc : Encoder α w) (x : α) :
    enc.encode (enc.decode (enc.encode x)) = enc.encode x := by
  rw [enc.roundtrip x]

end Nexa.Decoding
