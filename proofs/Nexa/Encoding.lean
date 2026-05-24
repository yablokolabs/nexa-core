/-
  NexaCore — Encoding Correctness

  Formally verified properties of deterministic encoding/decoding.
  Proves that well-formed encoders satisfy roundtrip correctness.
-/
import Nexa.HyperVector

namespace Nexa.Encoding

variable {α : Type} {w : Nat}

/-- **Roundtrip correctness**: For any deterministic encoder,
    decode(encode(x)) = x. This follows directly from the Encoder structure
    which requires a proof of this property at construction. -/
theorem roundtrip_correct (enc : Encoder α w) (x : α) :
    enc.decode (enc.encode x) = x :=
  enc.roundtrip x

/-- If an encoder has a left-inverse decoder, it must be injective.
    Distinct inputs produce distinct encodings. -/
theorem encode_injective (enc : Encoder α w) :
    Function.Injective enc.encode := by
  intro x y h
  have hx := enc.roundtrip x
  have hy := enc.roundtrip y
  rw [h] at hx
  exact hx.symm.trans hy

/-- Encoding is deterministic: encoding the same value twice yields the same vector. -/
theorem encode_deterministic (enc : Encoder α w) (x : α) :
    enc.encode x = enc.encode x :=
  rfl

/-- **Composition of encoders**: If we have two roundtrip-correct encoding stages,
    their composition is also roundtrip-correct. -/
theorem compose_roundtrip
    (enc₁ : Encoder α w) (enc₂ : Encoder (HyperVector w) w) (x : α) :
    enc₂.decode (enc₂.encode (enc₁.encode x)) = enc₁.encode x :=
  enc₂.roundtrip (enc₁.encode x)

/-- A pair of encoders can be used sequentially: full pipeline roundtrip. -/
theorem pipeline_roundtrip (enc₁ : Encoder α w) (enc₂ : Encoder (HyperVector w) w) (x : α) :
    enc₁.decode (enc₂.decode (enc₂.encode (enc₁.encode x))) = x := by
  rw [enc₂.roundtrip]
  exact enc₁.roundtrip x

end Nexa.Encoding
