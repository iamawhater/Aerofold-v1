"""Regenerate versioned demo inputs. Run: python -m scripts.generate_demo."""

from pathlib import Path

from backend.demo import library_csv, make_demo

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "data"
    target.mkdir(exist_ok=True)
    dataset, oracle = make_demo()
    (target / "demo_library.csv").write_text(library_csv(dataset), encoding="utf-8")
    (target / "demo_oracle.csv").write_text(library_csv(dataset, oracle), encoding="utf-8")
    (target / "demo_library.fasta").write_text(
        "\n".join(
            f">{v.variant_id}/{chain}\n{sequence}"
            for v in [dataset.parent, *dataset.variants]
            for chain, sequence in (("H", v.heavy_sequence), ("L", v.light_sequence))
            if sequence
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote 500 variants, parent, 48 seed measurements and separate synthetic oracle.")
