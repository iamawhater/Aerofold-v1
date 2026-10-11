"""Build the evidence-led AeroFold-Ab product assessment PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "AeroFold_Ab_Product_Assessment.pdf"
INK = colors.HexColor("#21352E")
MUTED = colors.HexColor("#60736B")
TEAL = colors.HexColor("#19745F")
PALE = colors.HexColor("#EAF1EC")
GOLD = colors.HexColor("#B58237")
RED = colors.HexColor("#A34B44")
RULE = colors.HexColor("#DDE5DF")
WHITE = colors.white


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Kicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=TEAL, spaceAfter=8, uppercase=True))
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=31, leading=36, textColor=INK, alignment=TA_LEFT, spaceAfter=14))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontName="Helvetica", fontSize=13, leading=19, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=10, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.2, leading=14, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.6, leading=10.5, textColor=MUTED, spaceAfter=4))
styles.add(ParagraphStyle(name="Cellx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.7, leading=10.6, textColor=INK, spaceAfter=0))
styles.add(ParagraphStyle(name="CellHeadx", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.6, leading=10, textColor=WHITE, spaceAfter=0))
styles.add(ParagraphStyle(name="Calloutx", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=15, textColor=INK, spaceAfter=0))
styles.add(ParagraphStyle(name="Metricx", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=18, leading=21, textColor=INK, alignment=TA_CENTER, spaceAfter=2))
styles.add(ParagraphStyle(name="Labelx", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.2, leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=0))
styles.add(ParagraphStyle(name="Sourcex", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.4, leading=10.5, textColor=INK, leftIndent=12, firstLineIndent=-12, spaceAfter=5, splitLongWords=1))


def p(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return p(f'<font color="{TEAL.hexval()}"><b>-</b></font>  {text}')


def callout(text: str, fill=PALE, accent=TEAL, width=7.05 * inch) -> Table:
    box = Table([[p(text, "Calloutx")]], colWidths=[width])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return box


def grid(headers, rows, widths, font=7.7):
    data = [[p(str(value), "CellHeadx") for value in headers]]
    data.extend([[p(str(value), "Cellx") for value in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    rules = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            rules.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F6F8F6")))
    table.setStyle(TableStyle(rules))
    return table


def metric_cards(items):
    cells = []
    for title, value, sub, color in items:
        cells.append(Table([[p(value, "Metricx")], [p(title.upper(), "Labelx")], [p(sub, "Labelx")]], colWidths=[1.67 * inch], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8F6")),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TEXTCOLOR", (0, 0), (-1, 0), color),
        ])))
    outer = Table([cells], colWidths=[1.72 * inch] * len(cells), hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
    return outer


def chrome(canvas, doc):
    canvas.saveState()
    w, h = letter
    if doc.page == 1:
        canvas.setFillColor(INK)
        canvas.rect(0, h - 12, w, 12, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.6)
        canvas.line(0.72 * inch, h - 0.52 * inch, w - 0.72 * inch, h - 0.52 * inch)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(TEAL)
        canvas.drawString(0.72 * inch, h - 0.42 * inch, "AEROFOLD-AB  /  PRODUCT ASSESSMENT")
    canvas.setStrokeColor(RULE)
    canvas.line(0.72 * inch, 0.52 * inch, w - 0.72 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.72 * inch, 0.34 * inch, "INTERNAL PRODUCT AND SCIENTIFIC REVIEW  |  28 SEP 2026")
    canvas.drawRightString(w - 0.72 * inch, 0.34 * inch, f"{doc.page:02d}")
    canvas.restoreState()


class AssessmentDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch, topMargin=0.72 * inch, bottomMargin=0.72 * inch, title="AeroFold-Ab Product and Scientific Assessment", author="OpenAI")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="assessment", frames=frame, onPage=chrome)])


story = []

# Page 1 - the product proposition and executive readout.
story += [Spacer(1, 0.36 * inch), p("PRODUCT PURPOSE  /  SCIENTIFIC REVIEW  /  ROADMAP", "Kicker"),
          p("From sequence library<br/>to smarter experiments.", "CoverTitle"),
          p("A product and scientific assessment of AeroFold-Ab: what the application is designed to do, what the current implementation can support, where evidence is thin, and the work that would make it dependable for serious antibody research.", "Deck"),
          Spacer(1, 0.06 * inch),
          callout("AeroFold-Ab is a local research workbench for ranking antibody variants under binding, stability and diversity constraints. Its most credible role today is helping a scientist decide what to measure next - not declaring a sequence therapeutic-ready."),
          Spacer(1, 0.23 * inch),
          p("EXECUTIVE READOUT", "Kicker"),
          metric_cards([
              ("Useful prototype", "WORKING", "End-to-end local loop", TEAL),
              ("Evidence breadth", "LIMITED", "One landscape, one seed", GOLD),
              ("Therapeutic claim", "NOT READY", "No clinical validation", RED),
          ]),
          Spacer(1, 0.19 * inch),
          Table([[p("SEQUENCE LIBRARY", "Labelx"), p("MODEL + FILTER", "Labelx"), p("DIVERSE BATCH", "Labelx"), p("ASSAY + LEARN", "Labelx")]], colWidths=[1.75 * inch] * 4, style=TableStyle([
              ("BACKGROUND", (0,0), (-1,-1), PALE), ("BOX", (0,0), (-1,-1), 0.5, RULE),
              ("TOPPADDING", (0,0), (-1,-1), 12), ("BOTTOMPADDING", (0,0), (-1,-1), 12),
              ("LINEAFTER", (0,0), (-2,-1), 0.5, WHITE),
          ])),
          Spacer(1, 0.2 * inch),
          p("The product closes a practical loop: ingest parent and candidate sequences; combine sequence features with measured binding and stability labels; fit two Gaussian-process surrogates; apply feasibility rules; rank candidates using Monte Carlo expected hypervolume improvement (MC-EHVI); select a diverse batch with a determinantal point process (DPP); then capture the run for review, export and the next assay cycle."),
          p("The dashboard already supports a separate benchmark demo. It packages a curated 512-variant slice of the 4-4-20 antibody Tite-Seq landscape and runs AeroFold against a random policy. The benchmark records a useful caution: under the default seed, random selection wins on top-binder recall. The demo should make that kind of result visible, not promise a win."),
          Spacer(1, 0.08 * inch)]
story.append(PageBreak())

# Page 2 - product behavior and current architecture.
story += [p("01  /  WHAT THE APPLICATION DOES", "Kicker"), p("A practical loop with clear seams", "H1x"),
          p("The architecture separates the user interface, API boundary, sequence features, surrogate models, acquisition logic, persistence and benchmark evaluator. That separation is a sound base for iterating on the science without entangling the UI with model code."),
          grid(["Stage", "Current behavior", "What it gives the team"], [
              ("Library and assays", "CSV/FASTA parsing; paired or single variable domains; measured binding and stability values; SQLite revisions.", "A traceable starting set and round-by-round evidence."),
              ("Sequence context", "Approximate CDR transfer, parent-relative mutations and optional ESM-2 embeddings with masked pseudo-likelihood.", "Interpretable edits plus a slower pretrained representation option."),
              ("Model", "Two independent PyTorch exact RBF GPs. The model returns posterior means and latent epistemic standard deviations.", "A rankable prediction and an uncertainty estimate for each objective."),
              ("Choose", "Hard feasibility mask, marginal MC-EHVI, an uncertainty blend and quality-weighted greedy DPP for 24 candidates.", "A diverse, constrained research shortlist for review."),
              ("Learn again", "Append assay values, create immutable run snapshots, compare history and export generic plate/order files.", "An operational pathway from recommendations to future measurements."),
              ("Benchmark", "Curated Tite-Seq fragment subset; shared seed; active and random policies; recall, rank correlation and hypervolume.", "An executable retrospective comparison with visible limitations."),
          ], [1.12 * inch, 3.25 * inch, 2.53 * inch]),
          Spacer(1, 0.15 * inch), p("A strong product boundary", "H2x"),
          p("The current distinction between measured outcomes and predictions is important. The ordinary discovery flow fits only uploaded observations; synthetic oracle outcomes are isolated for the local demo. The benchmark runner likewise passes revealed labels into model fitting and keeps hidden labels in retrospective evaluation. Database-backed job state, checksummed benchmark files, bounded submissions, input validation and explicit failed states are useful foundations."),
          callout("Keep the product positioned as decision support. A prediction is a hypothesis about a particular assay and context; a feasibility filter is a modeled screen. Neither is a structural-stability result or an experimental release decision.", fill=colors.HexColor("#FFF4E9"), accent=GOLD),
          PageBreak()]

# Page 3 - scientific limitations and their implications.
story += [p("02  /  SCIENTIFIC LIMITS", "Kicker"), p("What the model cannot yet know", "H1x"),
          p("The main improvement opportunity is evidence quality, not another layer of visual polish. A complete product must show which claims are supported by measurements, which are inferred by models and which remain untested."),
          grid(["Gap", "Current state", "Consequence and improvement"], [
              ("Assay meaning", "Binding is an assay-specific score. Main workflow stability is an adjusted 0-100 assay score; benchmark 'stability' is transformed expression.", "Never mix these as if they were the same physical property. Add assay type, units, conditions, replicate counts and transforms as required metadata."),
              ("Benchmark sequence", "Packaged Tite-Seq variants are two 10-aa CDR1H/CDR3H fragments concatenated into 20 residues.", "These are not complete VH/VL sequences. Do not interpret their fragment projection as full-protein context. Validate a fixed 4-4-20 scaffold and residue mapping before an ESM comparison."),
              ("Benchmark target", "The stability axis is expression normalized against WT and mapped to 0-100; it is explicitly not structural stability.", "Tite-Seq affinity curves were designed to separate affinity from expression/display confounding. Treat expression as its own measured proxy and label the biological context."),
              ("Generalization", "One 512-variant curated slice, one default random seed and one affinity landscape.", "A win, loss or correlation here is not evidence of general advantage. Add multiple tasks, split regimes, random seeds, paired uncertainty intervals and independent antibody families."),
              ("Uncertainty", "Exact GP sigma is a latent model estimate; hyperparameters and data are limited.", "Assess calibration, coverage and sharpness on held-out variants. Do not present sigma as an assay confidence interval without calibration."),
              ("CDR labels", "Current annotations are approximate and positions are ungapped-parent coordinates, not a standard scheme.", "Use validated IMGT numbering / region boundaries, preserve insertion coordinates and report annotation confidence before position-specific design claims."),
              ("Developability", "Charge, hydrophobic patch, APR and naturalness are sequence heuristics.", "They are not substitutes for expression, aggregation, solubility, thermal stability or specificity assays. Calibrate each screen against measured program data."),
          ], [1.15 * inch, 2.55 * inch, 3.2 * inch]),
          Spacer(1, 0.1 * inch),
          p("A subtle benchmark issue", "H2x"),
          p("Top-5% recall includes all ties at the cutoff and all observed hits, even candidates that a model's feasibility gate might exclude. That is transparent, but teams should also report recall among feasible binders and best-so-far binding. Spearman is measured on the common still-unseen intersection; that group changes each round and depends on what either policy selected. Add a predeclared held-out panel evaluated every round for comparable rank-correlation curves."),
          PageBreak()]

# Page 4 - what the current benchmark establishes.
story += [p("03  /  BENCHMARK MODE", "Kicker"), p("A useful diagnostic, not a verdict", "H1x"),
          p("The packaged benchmark is reproducible and immediately usable. Its default run starts both agents from the same 20 labels and adds five batches of 24. Seed 42 produced the following values after 140 observations per agent:"),
          metric_cards([
              ("Top-5% recall", "7.7%", "Random: 26.9%", RED),
              ("Spearman rho", "0.389", "Random: 0.415", GOLD),
              ("Hypervolume", "0.797", "Random: 0.947", RED),
          ]), Spacer(1, 0.17 * inch),
          callout("On this specific seeded run, the random baseline found more top binders and achieved higher cumulative hypervolume. The result is retained as observed. It does not show that random selection is generally superior." , fill=colors.HexColor("#FFF0EF"), accent=RED),
          Spacer(1, 0.13 * inch), p("Interpret the comparison precisely", "H2x"),
          bullet("The random policy samples uniformly only from candidates it predicts feasible under its own GP model. It is a learned-feasibility random baseline, not unconstrained random sampling from the entire library."),
          bullet("Each policy receives the same seed IDs, then separate observed labels. A model cannot learn from measurements selected only by the other policy."),
          bullet("Both batches are selected before either is measured. If either policy cannot fill a feasible batch, the simulation stops without lowering its gate."),
          bullet("Recall, enrichment, correlation and hypervolume answer different questions. Enrichment is relative to the library-wide top-binder frequency; hypervolume uses fixed full-dataset normalization for evaluation."),
          bullet("One random seed hides variance. A credible report needs many seeds, paired trajectory differences, confidence intervals and a runtime budget."),
          PageBreak(), p("03A  /  BENCHMARK PROTOCOL", "Kicker"),
          p("Make the comparison fair and repeatable", "H1x"),
          p("Retrospective benchmarks are useful when every method is evaluated under the same information limits. Fix the data split and assay budget before comparing models, and let the evaluator reveal oracle labels only after a virtual assay is selected."),
          Spacer(1, 0.12 * inch), p("Recommended evaluation matrix", "H2x"),
          grid(["Question", "Minimum useful comparison"], [
              ("Does the representation help?", "Same sequences, labels, GP and acquisition; descriptors vs ESM; repeat seeds; report held-out prediction and active-learning trajectories."),
              ("Does active learning help?", "Random, greedy predicted mean, uncertainty/UCB, MC-EHVI and MC-EHVI + DPP; equal total assays and the same feasibility rule."),
              ("Does the result transfer?", "Additional antibody binding and expression/stability landscapes; random and mutation-distance splits; leave-one-family-out where possible."),
              ("Are uncertainty and constraints useful?", "Calibration/coverage plus measured pass rates, false-feasible rate, assay outcomes and diversity / redundancy at each round."),
          ], [1.62 * inch, 5.12 * inch]),
          Spacer(1, 0.12 * inch),
          p("A minimal study protocol", "H2x"),
          bullet("Define the target and its units first: affinity, expression, aggregation, thermal stability and specificity are separate endpoints."),
          bullet("Hold out a fixed test panel. Exclude its labels from seed sampling, GP fitting, acquisition, threshold tuning and all model selection."),
          bullet("Repeat trajectories across multiple paired seeds. Report each metric by round, mean and uncertainty across seeds, total assays, wall-clock time and hardware."),
          bullet("Ablate one decision at a time: descriptor versus ESM; mean pooling versus likelihood; greedy versus uncertainty; EHVI with versus without DPP."),
          bullet("Predefine a practically meaningful improvement and constraint pass rate with the assay team. Test those claims in a blinded prospective round."),
          Spacer(1, 0.08 * inch),
          callout("For an ESM ablation on 4-4-20, reconstruct each variant on a verified, fixed antibody scaffold using authoritative sequence coordinates. Then compare full-context ESM embeddings and masked likelihoods to the same framework with CDR substitutions, under identical splits, labels, GP, acquisition and assay budgets. Keep language likelihood as its own feature or ablation so its contribution can be measured."),
          Spacer(1, 0.11 * inch),
          p("A recent protein-engineering study found that ESM representations improved some tasks but worsened others, and that uncertainty metrics alone did not reliably predict active-learning gains. That is why this app should test ESM on its own measured landscapes instead of assuming a pretrained model will win.", "Smallx"),
          PageBreak()]

# Page 5 - engineering and product risks.
story += [p("04  /  PRODUCT AND ENGINEERING GAPS", "Kicker"), p("Make trustworthy behavior the default", "H1x"),
          grid(["Area", "Observed limitation", "Top-notch direction"], [
              ("Experiment design", "Model fitting, thresholds and comparisons inherit users' assay noise and protocol drift.", "First-class experiment/target registry, protocol version, replicate-level data, controls and plate metadata."),
              ("Model monitoring", "No prediction-service calibration report, prospective holdout dashboard or drift alarm.", "Record calibration and pass-rate by round; compare predicted vs observed; flag extrapolation and distribution shift."),
              ("GP scaling", "Exact GP is limited to 512 training points; larger datasets use evenly spaced rows by library order.", "Use a principled subset or sparse/variational GP; benchmark latency and validate loss against full models on tractable datasets."),
              ("Batch strategy", "Independent objective GPs ignore cross-objective dependence; marginal EHVI is not joint qEHVI; DPP is a heuristic.", "Compare joint Bayesian acquisition, Thompson sampling, greedy and diversity variants. Keep hard constraints explicit and auditable."),
              ("Annotation", "Approximate CDR transfer and repetitive alignments can misplace mutations, especially with indels or unusual germlines.", "Adopt a pinned antibody-numbering tool; include scheme, alignment, germline, confidence and user review in the sequence heatmap."),
              ("Operations", "The embedded queue and SQLite recovery are single-process; restarting marks active jobs failed.", "For multiuser deployment, separate API and workers, use a durable queue/shared DB, resumable checkpoints, migrations, metrics and structured traces."),
              ("Security", "Optional bearer-token gate is not user or tenant management; SQLite, caches and audit exports contain sequence data.", "Before remote use, add identity/RBAC, encrypted storage/backups, retention controls, secret rotation, TLS, audit access and tenant isolation."),
              ("Lab automation", "Exports are generic manifests and protein order sheets, not instrument-specific protocols.", "Add validated vendor/instrument adapters, source plate maps, volumes, liquid classes, control wells, checksums and dry-run validation."),
              ("User trust", "The benchmark UI is transparent but could be mistaken for evidence of antibody efficacy or stability.", "Label each value as observed/predicted/proxy; show dataset, assay, run ID, uncertainty and domain-of-use at point of decision."),
          ], [1.04 * inch, 2.42 * inch, 3.28 * inch]),
          Spacer(1, 0.14 * inch),
          callout("Keep user sequences local by default. Treat ESM caches, database files, logs and audit exports as sensitive research data; minimize who can retrieve them.", fill=colors.HexColor("#EEF3FB"), accent=colors.HexColor("#5378A6")),
          PageBreak()]

# Page 6 - actionable maturity roadmap and sources.
story += [p("05  /  ROADMAP", "Kicker"), p("Earn confidence in stages", "H1x"),
          grid(["Priority", "Work", "Exit evidence"], [
              ("P0  Scientific truth", "Separate assay labels and units. Correct benchmark UI/metadata so expression is an expression outcome. Preserve source transforms and checksums.", "A reviewer can trace every outcome from source row and preprocessing to displayed label."),
              ("P0  Benchmark credibility", "Add fixed held-out evaluation IDs, multiple random seeds, paired comparisons, greedy and uncertainty baselines, confidence intervals, and feasibility-aware recall.", "Results report uncertainty across seeds and never leak holdout outcomes into acquisition."),
              ("P1  Sequence context", "Use validated full 4-4-20 scFv context only if source mapping is exact; add complete-sequence antibody datasets with pinned numbering and assay metadata.", "Sequence provenance and mapped residues are verified; ESM embeddings/language scores match the full-chain input."),
              ("P1  Prospective validation", "Run a small blinded, prospective assay round. Calibrate binding and measured developability endpoints, quantify repeatability and threshold pass rates.", "Predeclared metric and assay protocol; raw replicates and failed assays retained."),
              ("P1  Production operations", "Durable job workers, shared database, tenant identity, encrypted data handling, deployment observability, retention and recovery.", "Load, restore, access-control and incident-response checks pass before remote customer data."),
              ("P2  Advanced optimization", "Evaluate joint batch acquisitions, calibrated or ensemble uncertainty, explicit assay cost and multi-assay constraints.", "Ablations show reproducible gains over simple greedy and random baselines on independent landscapes."),
          ], [1.0 * inch, 3.25 * inch, 2.49 * inch]),
          Spacer(1, 0.12 * inch), p("A sensible release bar", "H2x"),
          p("Call the system a dependable research decision-support product after it demonstrates reproducible, statistically summarized benefits on multiple relevant datasets and prospective measurements, has validated antibody numbering and assay semantics, and provides secure, recoverable data handling. A therapeutic nomination requires substantially more evidence: target-specific experimental confirmation, orthogonal binding and developability assays, structure/biophysical review and the organization's normal safety and development process."),
          p("Sources and basis", "H2x"),
          p("Code-level statements in this report were checked against the AeroFold-Ab source, README, benchmark manifest and recorded local validation. Scientific context below supports the assessment; proposed improvements are engineering recommendations, not claims made by those publications.", "Smallx"),
          p('[1] Adams RM et al. Measuring the sequence-affinity landscape of antibodies with massively parallel titration curves. eLife 5:e23156 (2016). <link href="https://doi.org/10.7554/eLife.23156" color="#19745F">doi:10.7554/eLife.23156</link>. Tite-Seq measures titration-derived affinity in CDR1H/CDR3H variants and was designed to address expression/display confounding.', "Sourcex"),
          p('[2] IMGT. IMGT unique numbering for immunoglobulin and T-cell receptor variable regions. <link href="https://www.imgt.org/IMGTScientificChart/Numbering/IMGTnumbering.html" color="#19745F">IMGT Scientific Chart</link>. Supports standard numbering and CDR/framework boundary recommendations.', "Sourcex"),
          p('[3] Lin Z et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023). <link href="https://doi.org/10.1126/science.ade2574" color="#19745F">doi:10.1126/science.ade2574</link>. Source for ESM-2; pretrained embeddings should be evaluated for the task rather than assumed superior.', "Sourcex"),
          p('[4] Greenman KP, Amini AP, Yang KK. Benchmarking uncertainty quantification for protein engineering. PLOS Computational Biology 21(1):e1012639 (2025). <link href="https://doi.org/10.1371/journal.pcbi.1012639" color="#19745F">doi:10.1371/journal.pcbi.1012639</link>. Evaluates calibration, prediction and active-learning / optimization performance across landscapes; findings caution that no method wins across every setting.', "Sourcex"),
          p('[5] Packaged source data: <link href="https://github.com/jbkinney/16_titeseq/tree/54130c8fc6081bbfa8301c41e1afcfa6a5d79590" color="#19745F">jbkinney/16_titeseq</link>, pinned commit 54130c8fc6081bbfa8301c41e1afcfa6a5d79590. Curated subset details and replicate hashes are included in data/benchmarks/titeseq_4_4_20.json.', "Sourcex"),
]

doc = AssessmentDoc(OUT)
doc.build(story)
print(OUT)

