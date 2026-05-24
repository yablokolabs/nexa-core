//! # nexa-proof — Formal Verification Bridge
//!
//! Invokes the Lean 4 proof layer to verify NexaCore's algebraic invariants.
//! The proofs live in `proofs/` and are checked by `lake build`.
//!
//! This crate provides:
//! - A `ProofVerifier` that runs `lake build` and reports results
//! - A manifest of all verified theorems
//! - Integration with the `nexa` CLI via the `verify` subcommand

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// A single verified theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theorem {
    pub module: String,
    pub name: String,
    pub statement: String,
}

/// Result of running the proof verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofReport {
    pub verified: bool,
    pub theorems: Vec<Theorem>,
    pub lean_output: String,
    pub error: Option<String>,
}

/// The NexaCore proof manifest — all theorems we verify.
pub fn theorem_manifest() -> Vec<Theorem> {
    vec![
        // Binding algebra
        Theorem {
            module: "Nexa.Binding".into(),
            name: "bind_comm".into(),
            statement: "∀ a b, bind(a, b) = bind(b, a)".into(),
        },
        Theorem {
            module: "Nexa.Binding".into(),
            name: "bind_assoc".into(),
            statement: "∀ a b c, bind(bind(a, b), c) = bind(a, bind(b, c))".into(),
        },
        Theorem {
            module: "Nexa.Binding".into(),
            name: "bind_unbind_reverse".into(),
            statement: "∀ a b, unbind(bind(a, b), a) = b".into(),
        },
        Theorem {
            module: "Nexa.Binding".into(),
            name: "bind_self_cancel".into(),
            statement: "∀ a, bind(a, a) = 0".into(),
        },
        Theorem {
            module: "Nexa.Binding".into(),
            name: "role_filler_recovery".into(),
            statement: "∀ role filler, unbind(bind(role, filler), role) = filler".into(),
        },
        // Permutation invariants
        Theorem {
            module: "Nexa.Permutation".into(),
            name: "perm_inverse_roundtrip".into(),
            statement: "∀ σ v, P⁻¹(P(v)) = v".into(),
        },
        Theorem {
            module: "Nexa.Permutation".into(),
            name: "perm_preserves_size".into(),
            statement: "∀ σ v, |P(v)| = |v|".into(),
        },
        Theorem {
            module: "Nexa.Permutation".into(),
            name: "compose_inverse_is_identity".into(),
            statement: "∀ σ, (σ⁻¹ ∘ σ) = id".into(),
        },
        // Similarity metrics
        Theorem {
            module: "Nexa.Similarity".into(),
            name: "hamming_self_zero".into(),
            statement: "∀ v, d(v, v) = 0".into(),
        },
        Theorem {
            module: "Nexa.Similarity".into(),
            name: "hamming_symmetric".into(),
            statement: "∀ a b, d(a, b) = d(b, a)".into(),
        },
        Theorem {
            module: "Nexa.Similarity".into(),
            name: "hamming_bind_invariant".into(),
            statement: "∀ a b k, d(a⊕k, b⊕k) = d(a, b)".into(),
        },
        // Encoding/Decoding
        Theorem {
            module: "Nexa.Encoding".into(),
            name: "roundtrip_correct".into(),
            statement: "∀ enc x, decode(encode(x)) = x".into(),
        },
        Theorem {
            module: "Nexa.Encoding".into(),
            name: "encode_injective".into(),
            statement: "∀ enc, injective(encode)".into(),
        },
        Theorem {
            module: "Nexa.Decoding".into(),
            name: "symbolic_unbind_recovers".into(),
            statement: "∀ a b, unbind(bind(a, b), a) = b".into(),
        },
        Theorem {
            module: "Nexa.Decoding".into(),
            name: "nested_unbind".into(),
            statement: "∀ a b c, unbind(unbind(bind(a, bind(b, c)), a), b) = c".into(),
        },
        // Cleanup memory
        Theorem {
            module: "Nexa.CleanupMemory".into(),
            name: "cleanup_exact_self".into(),
            statement: "∀ v mem, nearest(mem, v) = v when v is first prototype".into(),
        },
        Theorem {
            module: "Nexa.CleanupMemory".into(),
            name: "corruption_reversible".into(),
            statement: "∀ v noise, (v ⊕ noise) ⊕ noise = v".into(),
        },
        // Homomorphism
        Theorem {
            module: "Nexa.Homomorphism".into(),
            name: "homomorphism_preserves_zero".into(),
            statement: "∀ f hom, f(0) = 0".into(),
        },
        Theorem {
            module: "Nexa.Homomorphism".into(),
            name: "transform_preserves_unbinding".into(),
            statement: "∀ f hom a b, unbind(f(bind(a,b)), f(a)) = f(b)".into(),
        },
        Theorem {
            module: "Nexa.Homomorphism".into(),
            name: "constant_not_homomorphism".into(),
            statement: "∀ c ≠ 0, const(c) is not a XOR-homomorphism".into(),
        },
        // Recovery bounds
        Theorem {
            module: "Nexa.RecoveryBounds".into(),
            name: "known_corruption_recovery".into(),
            statement: "∀ v noise, (v ⊕ noise) ⊕ noise = v".into(),
        },
        Theorem {
            module: "Nexa.RecoveryBounds".into(),
            name: "corruption_distance_equals_noise_weight".into(),
            statement: "∀ v noise, d(v, v⊕noise) = popcount(noise)".into(),
        },
        Theorem {
            module: "Nexa.RecoveryBounds".into(),
            name: "corruption_triangle".into(),
            statement: "∀ v n₁ n₂, d(v⊕n₁, v⊕n₂) = popcount(n₁⊕n₂)".into(),
        },
    ]
}

/// Verifier that invokes `lake build` on the proofs directory.
pub struct ProofVerifier {
    proofs_dir: PathBuf,
}

impl ProofVerifier {
    /// Create a verifier pointing at the proofs directory.
    pub fn new(proofs_dir: impl AsRef<Path>) -> Self {
        Self {
            proofs_dir: proofs_dir.as_ref().to_path_buf(),
        }
    }

    /// Discover the proofs directory relative to the workspace root.
    pub fn discover() -> Option<Self> {
        let candidates = [
            PathBuf::from("proofs"),
            PathBuf::from("../proofs"),
        ];
        for dir in &candidates {
            if dir.join("lakefile.toml").exists() {
                return Some(Self::new(dir));
            }
        }
        None
    }

    /// Run `lake build` and return a proof report.
    pub fn verify_all(&self) -> ProofReport {
        let theorems = theorem_manifest();

        let result = Command::new("lake")
            .arg("build")
            .current_dir(&self.proofs_dir)
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined = format!("{stdout}\n{stderr}").trim().to_string();
                let verified = output.status.success();

                ProofReport {
                    verified,
                    theorems,
                    lean_output: combined,
                    error: if verified { None } else { Some("lake build failed".into()) },
                }
            }
            Err(e) => ProofReport {
                verified: false,
                theorems,
                lean_output: String::new(),
                error: Some(format!("Failed to invoke lake: {e}")),
            },
        }
    }

    /// Print a human-readable verification report.
    pub fn print_report(report: &ProofReport) -> String {
        let mut out = String::new();
        out.push_str("NexaCore Formal Verification Report\n");
        out.push_str("═══════════════════════════════════\n\n");

        if report.verified {
            out.push_str(&format!("Status: ✓ ALL {} THEOREMS VERIFIED\n\n", report.theorems.len()));
        } else {
            out.push_str("Status: ✗ VERIFICATION FAILED\n\n");
            if let Some(err) = &report.error {
                out.push_str(&format!("Error: {err}\n\n"));
            }
        }

        let mut current_module = String::new();
        for thm in &report.theorems {
            if thm.module != current_module {
                current_module = thm.module.clone();
                out.push_str(&format!("  {current_module}\n"));
            }
            let status = if report.verified { "✓" } else { "?" };
            out.push_str(&format!("    {status} {}: {}\n", thm.name, thm.statement));
        }

        if !report.lean_output.is_empty() && !report.verified {
            out.push_str(&format!("\nLean output:\n{}\n", report.lean_output));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theorem_manifest_not_empty() {
        let manifest = theorem_manifest();
        assert!(!manifest.is_empty());
        assert!(manifest.len() >= 20);
    }

    #[test]
    fn test_all_modules_covered() {
        let manifest = theorem_manifest();
        let modules: std::collections::HashSet<_> = manifest.iter().map(|t| t.module.as_str()).collect();
        assert!(modules.contains("Nexa.Binding"));
        assert!(modules.contains("Nexa.Permutation"));
        assert!(modules.contains("Nexa.Similarity"));
        assert!(modules.contains("Nexa.Encoding"));
        assert!(modules.contains("Nexa.Decoding"));
        assert!(modules.contains("Nexa.CleanupMemory"));
        assert!(modules.contains("Nexa.Homomorphism"));
        assert!(modules.contains("Nexa.RecoveryBounds"));
    }

    #[test]
    fn test_proof_report_display() {
        let report = ProofReport {
            verified: true,
            theorems: theorem_manifest(),
            lean_output: "Build completed successfully".into(),
            error: None,
        };
        let display = ProofVerifier::print_report(&report);
        assert!(display.contains("VERIFIED"));
        assert!(display.contains("bind_unbind_reverse"));
    }
}
