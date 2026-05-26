/-
  NexaCore — Encryption Correctness

  Formally verified properties of the seed-based XOR encryption
  scheme (NexaCrypto). Since encryption is XOR with a deterministic
  key stream, correctness follows directly from XOR algebra.
-/
import Nexa.HyperVector
import Nexa.Binding
import Nexa.Similarity

namespace Nexa.Encryption

variable {w : Nat}

/-- XOR encryption with a key stream is modeled as binding with a key vector. -/
def encrypt (plaintext key : HyperVector w) : HyperVector w :=
  HyperVector.bind plaintext key

/-- XOR decryption is the same operation as encryption (XOR is self-inverse). -/
def decrypt (ciphertext key : HyperVector w) : HyperVector w :=
  HyperVector.bind ciphertext key

/-- **Encryption roundtrip**: decrypt(encrypt(v, k), k) = v.
    The core correctness property of NexaCrypto. -/
theorem encrypt_decrypt_roundtrip (v key : HyperVector w) :
    decrypt (encrypt v key) key = v := by
  simp only [encrypt, decrypt]
  exact Binding.bind_involution v key

/-- **Encryption is its own inverse**: encrypt(encrypt(v, k), k) = v.
    XOR encryption and decryption are the same operation. -/
theorem encrypt_self_inverse (v key : HyperVector w) :
    encrypt (encrypt v key) key = v := by
  exact encrypt_decrypt_roundtrip v key

/-- **Key cancellation**: Encrypting with two different keys and then
    decrypting both recovers the original.
    decrypt(k₂, decrypt(k₁, encrypt(k₁, encrypt(k₂, v)))) = v -/
theorem double_encrypt_decrypt (v k₁ k₂ : HyperVector w) :
    decrypt (decrypt (encrypt (encrypt v k₁) k₂) k₂) k₁ = v := by
  simp only [encrypt, decrypt]
  rw [Binding.bind_involution (HyperVector.bind v k₁) k₂]
  exact Binding.bind_involution v k₁

/-- **Ciphertext is binding**: The ciphertext is just the XOR binding of
    plaintext and key, so it preserves the algebraic structure. -/
theorem ciphertext_is_binding (v key : HyperVector w) :
    encrypt v key = HyperVector.bind v key :=
  rfl

/-- **Distance preservation under encryption**: Encrypting two vectors with
    the same key preserves their Hamming distance.
    d(encrypt(a,k), encrypt(b,k)) = d(a,b).
    This means similarity queries work on encrypted data. -/
theorem encrypt_preserves_distance (a b key : HyperVector w) :
    HyperVector.hammingDist (encrypt a key) (encrypt b key) =
    HyperVector.hammingDist a b := by
  simp only [encrypt]
  exact Similarity.hamming_bind_invariant a b key

end Nexa.Encryption
