Quantum Error-Correction DecoderOps Workbench Research Pack
Executive summary
The strongest first product is Quantum Error-Correction DecoderOps Workbench: a vendor-neutral, on-prem-first benchmarking, packaging, and deployment-readiness system for QEC decoder pipelines, built around Stim + PyMatching + Sinter + NVIDIA Ising-Decoding + ONNX Runtime, with CUDA-Q/CUDA-QX QEC as an integration target and TensorRT as an optional acceleration layer on NVIDIA hardware. That is the narrowest path that is both technically serious and commercially credible today. It attacks a real engineering bottleneck — choosing, validating, exporting, and operationalising decoders under latency and logical-error constraints — without pretending you are shipping a full control stack or a production fault-tolerant runtime on day one. 

The calibration-VLM route is real, but it is weaker as a first paid product. Public evidence shows calibration is painful and that vendors are already turning it into workflow software, state machines, graph orchestration, and autonomous tune-up products; however, a useful commercial calibration product quickly collides with customer-specific hardware interfaces, unpublished experiment libraries, private plot formats, and proprietary decision logic. NVIDIA’s public calibration artefacts are openly interesting, but the Quantum Calibration Agent Blueprint is explicitly a reference blueprint, not a production control platform, and the public Ising Calibration 1 reference case is based on Qwen3.5-35B-A3B, which is a problem for any customer policy that excludes Chinese-origin base models from the production stack. 

The blunt truth is this: a broader “quantum control-plane reliability” product is too wide for v1, and a calibration agent product is too integration-heavy for your first serious commercial build. DecoderOps is the one that can be delivered locally, mostly in Python, provide value before hardware integration, and still extend into customer-specific real-time deployments later. 

Best product opportunity
The recommendation
[Verified recommendation] Build Quantum Error-Correction DecoderOps Workbench first. It should do five things well: ingest detector error models or syndrome datasets, benchmark multiple decoder paths under reproducible protocols, package deployable artefacts for downstream runtimes, produce buyer-grade technical reports, and leave a clean seam for later real-time integration with customer control stacks. That recommendation matches the public NVIDIA Ising release, which is already organised around training frameworks, pre-trained models, ONNX export, TensorRT paths, generated test data, and CUDA-Q QEC downstream integration. 

The reason this is commercially stronger than calibration-first is simple. The public QEC tooling already exposes a clean offline-to-runtime path: generate or ingest circuits and detector error models, compare decoders, export ONNX, optionally quantise, and integrate with CUDA-Q QEC or other customer runtimes. That is a product seam. By contrast, calibration products become useful only when they are wired deeply into lab-specific experiment registries, controller adapters, fit functions, acceptance thresholds, and remediation workflows. That is not impossible, but it is a much nastier first commercialisation problem. 

Candidate evaluation
Candidate	Technical core	Buyer	User	Why they pay	Narrow enough for first build	Mostly Python	Real hardware needed for MVP usefulness	Commercial verdict
Quantum Error-Correction DecoderOps Workbench	Benchmarking, comparison, export, deployment-readiness of PyMatching, correlated matching, Ising pre-decoders, ONNX/TensorRT, CUDA-Q QEC integration	QEC lead, CTO, hardware R&D head, controls/platform lead	QEC engineer, controls engineer, ML engineer	It answers the painful question: which decoder path should we actually deploy for our noise model and latency budget? It also produces artefacts and reports that customers can act on. 
Yes	Yes	No — public Stim/DEM workflows already make it useful offline. 
Build first
Agentic Quantum Calibration Triage and Replay Workbench	Plot analysis, calibration triage, replay, workflow branching, audit trail	Hardware operations lead, lab manager, calibration lead	Calibration engineer, experimental physicist, device engineer	Calibration consumes expert time and existing vendors already sell automation here. 
Only if heavily narrowed	Yes, mostly	Yes, sooner rather than later	Second product or optional module
Quantum Control-Plane Reliability Workbench	Cross-stack orchestration, timing, observability, replay, decoder/control coupling	CTO, platform lead, systems lead	Controls, runtime, and infra engineers	Pain is real, but the scope sprawls into controller integration, observability, and real-time systems. 
No	Partly	Usually yes	Too broad for v1

Where the pain is actually acute
For superconducting-qubit teams, two pains are plainly acute in public material: microsecond-scale or near-microsecond classical processing for QEC, and repetitive calibration/tune-up overhead. Quantum Machines’ own material is unusually explicit that classical latency and inter-decoder coordination can become a decoding bottleneck, while Riverlane’s entire commercial pitch is built around low-latency real-time QEC coupling into the control stack. That makes DecoderOps particularly relevant if the customer is already moving towards surface-code experiments or wants to prove decoder readiness before real-time deployment. 

For neutral-atom teams, the picture is less clean. NVIDIA’s QCalEval benchmark explicitly spans both superconducting qubits and neutral atoms, which supports the idea that calibration interpretation pain is modality-broad. But the public Ising-Decoding release is very specifically framed around surface-code decoding and the published model cards talk about rotated surface-code syndromes. So, if you target neutral-atom companies first, a decoder product is not automatically the strongest wedge unless they are actively on a fault-tolerance/QEC roadmap compatible with that style of decoder evaluation. 

For error-correction teams and quantum cloud providers, DecoderOps is the strongest fit. Those groups need reproducible evaluation, comparisons across decoders, exportability, deployment artefacts, and evidence that a candidate path can meet latency and logical-error requirements. Public CUDA-QX QEC materials explicitly distinguish offline analysis from real-time error correction on hardware, which is exactly the seam your product should exploit. 

The paid-pilot test
[Inferred] The pain is acute enough for paid pilots when the customer already has one of these conditions: a QEC subgroup evaluating decoders, a hardware team with syndrome traces and no robust benchmark harness, a controls team debating PyMatching versus AI pre-decoding under latency constraints, or an executive/technical need to make a deployment-readiness decision without building the entire evaluation stack internally. The public market evidence is not hypothetical: vendors are already selling real-time QEC stacks, calibration workflow platforms, and autocalibration systems. The gap is a neutral, local-first, benchmark-and-packaging product that helps small and mid-sized teams make rigorous decisions without buying a full proprietary stack. 

Verified artefact and licensing matrix
NVIDIA Ising and related public artefacts
Artefact	Exact official URL	What it is	Licence	Commercial-use status	Redistribution / deployment view	Practical conclusion
NVIDIA Ising landing page	https://developer.nvidia.com/ising	Main official page linking Ising Calibration, Ising Decoding, QCalEval, and tutorials	Page only	N/A	N/A	Use as the canonical index for public artefacts. 
NVIDIA/Ising-Decoding	https://github.com/NVIDIA/Ising-Decoding	Open training and inference framework for AI QEC pre-decoders; includes ONNX export, TensorRT paths, generated CUDA-Q QEC test data	Apache-2.0	Commercially safe in principle	Redistribution under Apache-2.0; local deployment straightforward	Core MVP foundation. 
Ising-Decoder-SurfaceCode-1-Fast	https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast	Open-weight fast decoder checkpoint	NVIDIA Open Model License	Model card says commercial/non-commercial use is allowed	Redistribution allowed under NVIDIA Open Model License with notice obligations; local deployment supported on NVIDIA GPUs	Use as an optional benchmarked model artefact, not your contractual dependency. 
Ising-Decoder-SurfaceCode-1-Accurate	https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate	Open-weight higher-accuracy decoder checkpoint	NVIDIA Open Model License	Model card says commercial/non-commercial use is allowed	Same as above	Use as second benchmark profile. 
CUDA-Q QEC realtime predecoder example	https://nvidia.github.io/cudaqx/examples_rst/qec/realtime_predecoder_pymatching.html	Official downstream integration example for realtime predecoder pipeline	Docs	N/A	Integration target for local/on-prem customers	This is the right “runtime target” story for your v1 reports. 
QCalEval dataset	https://huggingface.co/datasets/nvidia/QCalEval	Quantum calibration plot benchmark dataset	CC-BY-4.0	Dataset page states commercial/non-commercial use	Redistributable under CC-BY-4.0 with attribution	Useful for evaluation, not your first product moat. 
NVIDIA/QCalEval	https://github.com/nvidia/QCalEval	Official evaluation scripts for QCalEval; data loaded from Hugging Face	Apache-2.0	Commercially safe in principle	Local use straightforward	Keep as a sidecar research asset, not the v1 core. 
Quantum Calibration Agent Blueprint	https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint	Reference agent blueprint for AI-powered calibration with CLI and web UI	Apache-2.0	Commercially safe as code, subject to any downstream model/API choices	Local deployment supported; requires model/API access	Reference only; do not confuse it with a turnkey product. 
Ising Calibration 1 model	https://huggingface.co/nvidia/Ising-Calibration-1-35B-A3B	Open-weight calibration VLM reference case	NVIDIA Open Model License on the model card	NVIDIA Open Model License is commercially usable	Redistribution requires licence and notice handling	Legally usable, but policy-sensitive because it is published as based on Qwen3.5-35B-A3B. 
NIM API page for Ising Calibration 1	https://build.nvidia.com/nvidia/ising-calibration-1-35b-a3b	Hosted evaluation / API entry point	NVIDIA hosted service terms	[Unverified] exact enterprise redistribution posture from the page I inspected	Cloud use exists; on-prem terms not established from that page alone	Do not make this a mandatory dependency for v1. 

Final recommended dependency stack and commercialisation safety
Dependency	Exact official URL	Licence	Commercial-use view	Red flags
Stim	https://github.com/quantumlib/stim	Apache-2.0	Commercially safe	None material for v1. 
PyMatching	https://github.com/oscarhiggott/PyMatching	Apache-2.0	Commercially safe	None material for v1. 
CUDA-Q	https://github.com/NVIDIA/cuda-quantum and https://nvidia.github.io/cuda-quantum/latest/using/install/install.html	Apache-2.0 for repo code	Commercially safe in principle	CUDA-Q Python wheels rely on NVIDIA/cuQuantum components for acceleration; check packaging in customer environments. 
CUDA-QX / cudaq-qec	https://github.com/NVIDIA/cudaqx and https://nvidia.github.io/cudaqx/components/qec/introduction.html	Repo mostly Apache-2.0, but libcudaq-qec-nv-qldpc-decoder.so is closed source under NVIDIA Software Licence	Mostly safe if you keep closed components optional and customer-installed	Do not build your moat around the closed qLDPC decoder library. 
ONNX	https://github.com/onnx/onnx	Apache-2.0	Commercially safe	None material. 
ONNX Runtime	https://github.com/microsoft/onnxruntime	MIT	Commercially safe	None material. 
TensorRT	https://developer.nvidia.com/tensorrt	NVIDIA SDK licence	Commercially usable with distribution conditions	Review distributable portions and SaaS packaging carefully; not permissive OSS. 
PyTorch	https://github.com/pytorch/pytorch	BSD-style	Commercially safe	None material. 
cuQuantum / cuQuantum Python	https://github.com/NVIDIA/cuQuantum and https://pypi.org/project/cuquantum-python/	Repo primarily BSD-3-Clause with exceptions; SDK components also carry NVIDIA licence terms	Usable, but not as clean as Stim/PyMatching	Treat as optional acceleration/training dependency, not a mandatory licensing anchor. 

Commercially safe boundary
The commercially safest MVP boundary is this: your product ships your orchestration, benchmarking, packaging, reporting, and local API/UI layer, and integrates with Stim, PyMatching, ONNX, ONNX Runtime, CUDA-Q, and the Apache-licensed Ising-Decoding code. TensorRT, cuQuantum, and any closed CUDA-QX decoder libraries should be optional, customer-installed acceleration targets, not mandatory bundled dependencies. That keeps your legal posture cleaner and avoids tying your value proposition to components whose redistribution terms are narrower than Apache/MIT/BSD. 

Product architecture and MVP
Product definition
Recommended final product name: Quantum Error-Correction DecoderOps Workbench.
One-sentence description: a local-first workbench that benchmarks, compares, packages, and evidence-stamps QEC decoder pipelines — from Stim-generated or customer-provided detector data through to ONNX/TensorRT/CUDA-Q-ready deployment artefacts. This choice is directly aligned with the public Ising-Decoding release, which already covers model training, pre-trained checkpoints, ONNX export, TensorRT inference, test-data generation, and CUDA-Q QEC downstream ingestion. 

Best architecture
The best v1 architecture is a seven-layer local-first system.

Ingestion layer. Accept Stim circuits, stim.DetectorErrorModel files, syndrome arrays, customer-produced shot logs, and Ising-Decoding configuration bundles. Stim and PyMatching already interoperate around detector error models, which gives you a stable public substrate instead of inventing a proprietary internal format on day one. 

Experiment and benchmark runner. Run standardised evaluations across decoder variants: baseline PyMatching, correlated PyMatching where relevant, Ising Fast, Ising Accurate, exported ONNX Runtime, and optional TensorRT engines. The Ising-Decoding repo already exposes workflows for train/inference plus ONNX/TensorRT export and a downstream ablation path. 

Model and decoder execution layer. Keep this pluggable. Do not hard-wire your product to one vendor’s runtime. Support Python-native decoders first; then ONNX Runtime GPU; then optional TensorRT; then CUDA-Q QEC real-time consumption as an export target. That preserves vendor neutrality while still giving NVIDIA-aligned customers a fast path. 

Metrics and evaluation layer. Compute logical error rate, residual syndrome density after pre-decoding, latency, throughput, export success, environment reproducibility, and decoder/runtime compatibility. This is the heart of the product. If you reduce it to “run model, print score”, you have built a toy. The public Ising paper and repo both frame the decoder problem around latency plus LER, not raw model accuracy alone. 

Artefact packaging layer. Emit model checkpoints, ONNX exports, TensorRT engine build metadata, CUDA-Q QEC test harness files, config manifests, benchmark CSV/Parquet, and report bundles. The official repo already provides the ONNX and generated .bin data path specifically for downstream CUDA-Q QEC consumption. 

Report generation layer. Produce engineering reports, procurement/compliance notes, compatibility matrices, and pilot sign-off summaries. This is how the product becomes commercial instead of academic: it must end in decisions and artefacts, not charts alone. [Inferred] Existing vendor stacks sell outcomes, not notebooks; your workbench needs to do the same. 

Optional AI-assisted analysis layer. Keep this strictly optional in v1. Use it for narrative summarisation, anomaly annotation, and report drafting — not as the core correctness engine. The core decoding and measurement logic must remain deterministic and auditable. [Inferred] That is the only credible stance for a first product selling into quantum hardware teams. 

Exact MVP boundary
Your MVP should include:

Stim/PyMatching/Sinter ingestion and benchmark orchestration.
Ising-Decoding model benchmarking with the public Fast and Accurate checkpoints.
ONNX export and ONNX Runtime execution.
Optional TensorRT benchmarking on NVIDIA GPUs.
Packaging for CUDA-Q QEC downstream tests.
Deterministic reporting with reproducibility metadata.
Local CLI and local web/API surface. 
Your MVP should deliberately exclude:

Turnkey real-time control-system integration into customer hardware.
Claims of production-ready fault-tolerant deployment.
Claims of broad code-family support beyond what you actually evaluate.
Calibration VLM decision automation as a core product.
Any dependency on hosted inference or vendor cloud APIs. 
What the product should guarantee
The product should guarantee reproducible decoder evaluation, packaging, and evidence-backed deployment recommendations for the code families and runtimes actually in scope. It should explicitly not claim to provide “quantum advantage”, “fault tolerance”, “real-time control integration”, or “production decoder certification” unless the customer independently validates those on their own hardware stack. Public QEC material is full of latency, throughput, and hardware-coupling caveats; if you hand-wave them away, you will destroy trust immediately. 

Programming reality and deployment model
Programming reality
The honest answer is that this product is mostly Python systems engineering with a quantum-specialised evaluation core. You do need to understand detector error models, syndrome decoding, logical observables, code distance, noise assumptions, and runtime constraints. But you do not need to spend v1 building deep gate-model algorithms or a grand Qiskit-like platform. The public stack is already heavily Python-oriented: Stim, PyMatching, Ising-Decoding, CUDA-Q Python, and cudaq-qec all support Python-first workflows. 

The codebase will mainly consist of:

experiment schemas and config handling;
runners for benchmarks and export workflows;
adapters for Stim, PyMatching, Ising-Decoding, ONNX Runtime, TensorRT, and CUDA-Q QEC;
metrics computation;
reporting and artefact packaging;
a local API/UI layer.
That is not glamorous, but it is exactly why it is commercially sensible. It is mostly disciplined software construction around a narrow scientific bottleneck. 
Where C++ is actually required
C++ should be treated as a later-stage integration requirement, not a v1 prerequisite. You only genuinely need it when you move into CUDA-Q Realtime integration, custom real-time handlers, or customer-specific low-latency coupling into FPGA/controller systems. NVIDIA’s own realtime materials make clear that this is the layer where few-microsecond round trips and tightly-coupled host/FPGA/GPU systems matter. That is not where you should start. 

Internal platform question
Your internal local quantum coding/orchestration platform is optional, not necessary, for this product. Do not build it first. The only parts of that internal platform that would genuinely accelerate this product are: code-generation helpers for benchmark configs, experiment sweep orchestration, report drafting, and artefact comparison. Those are acceleration layers, not the product itself. [Inferred] Build them later, behind the product, not in front of it. 

Deployment model
The correct stance is local-first with optional customer-cloud deployment later. It should not be SaaS-first. Public competitor material reinforces the same reality from different directions: Quantum Machines exposes a local calibration web app and graph orchestration model; Q-CTRL uses a local Python client even when cloud acceleration is involved; Riverlane’s Deltaflow is pitched as directly connected to the customer control stack; CUDA-Q Realtime is explicitly about tightly coupling classical compute to the processor control system. In this market, control-stack proximity is not a feature — it is the environment. 

The reasons customers will push you on-prem or into customer-controlled cloud are obvious: unpublished syndrome traces, proprietary noise models, controller timing details, calibration parameters, experiment metadata, and highly sensitive hardware roadmaps. You should assume serious customers will want source-available deployment logic where possible, local data retention, and no avoidable outbound dependencies. 

Exact toolchain and commands
Recommended first stack
Chosen path for MVP

OS: Ubuntu 24.04
Python: 3.11
Core QEC stack: stim, pymatching, sinter
NVIDIA alignment layer: cudaq, cudaq-qec, NVIDIA/Ising-Decoding
Model/export path: torch via Ising-Decoding requirements, onnx, onnxruntime-gpu
Optional acceleration: tensorrt-cu12 or tensorrt-cu13
Optional training acceleration: cuquantum-python
Local product shell: FastAPI + Typer + DuckDB/SQLite [Inferred engineering choice]
CUDA-Q officially supports Ubuntu 24.04 and Python 3.11+; CUDA-QX provides pip packages and a Docker image; Ising-Decoding targets Python 3.11–3.13. 
Copy-paste-ready base environment
bash
Copy
sudo apt-get update
sudo apt-get install -y git git-lfs python3.11 python3.11-venv python3-pip build-essential
git lfs install

mkdir -p ~/decoderops
cd ~/decoderops

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
These commands are the minimal Ubuntu prerequisites wrapped around the officially documented Python-based installation paths for CUDA-Q, CUDA-QX, and Ising-Decoding. CUDA-Q officially supports Ubuntu 24.04 and Python 3.11+, while Ising-Decoding targets Python 3.11–3.13. 

Copy-paste-ready chosen path
bash
Copy
source ~/decoderops/.venv/bin/activate

python -m pip install cudaq cudaq-qec
python -m pip install stim pymatching sinter
python -m pip install onnx onnxruntime-gpu

git clone https://github.com/NVIDIA/Ising-Decoding.git
cd Ising-Decoding
git lfs pull

# Inference-only install
pip install -r code/requirements_public_inference.txt

# Optional: if you want CUDA-enabled PyTorch selection to be explicit
export TORCH_CUDA=cu130

# Verify Python compatibility
bash code/scripts/check_python_compat.sh

# Run inference using the shipped public workflow
WORKFLOW=inference bash code/scripts/local_run.sh
CUDA-QX documents pip install cudaq-qec; ONNX Runtime documents pip install onnxruntime-gpu for CUDA 12.x; Ising-Decoding documents the inference requirements file, the compatibility check, and the local inference workflow. 

Copy-paste-ready export and packaging path
bash
Copy
source ~/decoderops/.venv/bin/activate
cd ~/decoderops/Ising-Decoding

# Export ONNX only
ONNX_WORKFLOW=1 WORKFLOW=inference bash code/scripts/local_run.sh

# Export ONNX and run TensorRT inference
ONNX_WORKFLOW=2 QUANT_FORMAT=int8 WORKFLOW=inference bash code/scripts/local_run.sh

# Generate downstream test data for CUDA-Q QEC realtime ingestion
python3 code/export/generate_test_data.py \
  --distance 13 \
  --n-rounds 104 \
  --num-samples 10000 \
  --basis X \
  --p-error=0.003 \
  --simple-noise
These are direct combinations of the officially documented Ising-Decoding export modes and its generate_test_data.py path for CUDA-Q QEC realtime use. TensorRT workflows in that repo require tensorrt and modelopt. 

Optional TensorRT path
bash
Copy
source ~/decoderops/.venv/bin/activate
python -m pip install --upgrade tensorrt-cu12
TensorRT’s official pip documentation recommends python3 -m pip install --upgrade tensorrt by default and allows explicit CUDA major selection with -cu12 or -cu13. TensorRT 10.x supports Ubuntu 24.04 x86-64 in the current support matrix. 

Optional cuQuantum path
bash
Copy
source ~/decoderops/.venv/bin/activate
python -m pip install -v --no-cache-dir cuquantum-python
NVIDIA documents cuquantum-python as a meta-package for CUDA 12 and 13 wheels; use it only if you actually need the acceleration or training path. For the MVP, it is optional, not mandatory. 

Fallback container paths
bash
Copy
# CUDA-QX all-in-one container
docker pull ghcr.io/nvidia/cudaqx
docker run --gpus all -it ghcr.io/nvidia/cudaqx
bash
Copy
# CUDA-Q official container path
docker run -it --gpus all --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
CUDA-QX officially publishes ghcr.io/nvidia/cudaqx, and CUDA-Q documents the NGC nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest container run command. 

Blackwell caveat
If you later move to Blackwell, CUDA-Q’s current docs state that Blackwell is supported for CUDA 13.x only in the GPU simulation requirements, and the local install guide also notes extra caveats around CUDA 12.8 and Python wheels on Blackwell. ONNX Runtime currently exposes a nightly CUDA 13.x path. Do not assume today’s CUDA 12 stack will transfer cleanly. 

Benchmark, competitor, and cross-check
Benchmark and pilot design
A technically credible demo is not “the model ran”. It is this: reproduce a public baseline, compare multiple decoder pipelines under a declared noise model, export the winning path, prove runtime compatibility, and package a report that a hardware team could use to decide what to test next. The minimum serious benchmark set should include PyMatching baseline, correlated PyMatching if graphlike/correlation assumptions justify it, Ising Fast, Ising Accurate, ONNX Runtime, and optional TensorRT. 

The metrics that matter most are:

logical error rate;
residual syndrome density after pre-decoding;
latency per inference / per round / end-to-end pipeline;
throughput in shots or rounds per second;
exportability to ONNX and downstream runtime load success;
reproducibility with fixed seeds, config hashes, dependency fingerprints, and retained outputs.
That metric shape follows the NVIDIA paper’s emphasis on low latency plus lower LER, and the broader sector emphasis on real-time classical bottlenecks. 
Public datasets and generators are good enough for a serious MVP. Use Stim generated circuits and detector error models, Sinter for fast Monte Carlo studies, the Ising-Decoding repo’s public configs and model files, and the repo’s own generated CUDA-Q QEC test-data path. For calibration experimentation later, you have QCalEval and its official evaluation scripts, but that should remain outside the first commercial boundary. 

[Inferred] A paid pilot should look like this: customer provides syndrome traces or noise assumptions; you benchmark baseline versus AI-assisted decoder paths locally; you export at least one candidate ONNX artefact; you prove one downstream runtime path loads correctly; and you deliver a report with deployment recommendations, compatibility caveats, and a risk register. If you cannot produce a deployment-readiness package, you have not built a workbench; you have built a benchmark harness. 

Competitor and gap analysis
NVIDIA / CUDA-Q / CUDA-QX QEC provide the most relevant public technical substrate today: models, export path, runtime integration, and examples. But NVIDIA is not currently offering the vendor-neutral DecoderOps product you need to build. What it exposes publicly is a toolkit and reference pathway, not a local-first decision and packaging platform for teams comparing decoder choices across their own data and downstream environments. That is your opening. 

Quantum Machines / QUAlibrate are strong evidence that calibration workflow orchestration is commercially real. Their public docs and product pages emphasise reusable calibration nodes, DAG-style graphs, graph traversal, parallel calibration, logging, reruns, and local web execution. That is precisely why a calibration workbench is plausible — and also why it is a crowded and integration-heavy first product area. Their visible gap is not “calibration workflow exists”; it is that they are anchored to OPX / QUA / controller-centric workflows, not a vendor-neutral QEC decoder benchmarking product. 

Q-CTRL Boulder Opal is the clearest opposite number on the calibration side. Public material positions it as autonomous calibration and tune-up software, with state-machine orchestration, controller interpreters, device data management, and a local Python client. Again, this proves the pain is real. It also proves that a first product head-to-head on calibration would mean walking straight into a company already selling a polished control-automation story. That is not where you should start. 

Riverlane Deltaflow is the strongest adjacent product on the decoder/control side. It is explicitly a QEC stack: decoder, interface, orchestration, direct connection to the control stack, and real-time hardware integration. That validates the market. But it is also a proprietary integrated stack, not a neutral workbench for customers who want to benchmark options, package artefacts, and decide what to deploy before committing to a full vendor stack. That is your gap. 

EdenCode is [Unverified as a benchmarkable competitor] on the basis of the public material I inspected. Its public site claims relevance to the NVIDIA Ising launch, but I did not find enough detailed official documentation on a shipped decoder benchmarking/deployment product to treat it as a clearly mapped competitor in the same category. 

Knowledge-base and sector cross-check
After cross-checking the idea against current public engineering evidence, the recommended product is still DecoderOps.

At the level of serious technical understanding, nothing about this recommendation is gimmicky. Fault-tolerant quantum computing requires syndrome extraction, decoding, classical processing, and feed-forward that does not collapse the logical clock. Current technical papers and vendor engineering material say the same thing in different language: low-latency decoding is crucial, backlog is a real risk, and control/decoder coupling matters. That is exactly the domain a DecoderOps workbench speaks to. 

The distinction between algorithmic quantum software and hardware/control-plane engineering also survives the cross-check. The public tools you would actually use here — Stim, PyMatching, CUDA-Q QEC, Ising-Decoding, Riverlane-style interfaces, QUAlibrate-style orchestration — are not primarily about variational algorithms or gate-level application coding. They are about simulation, decoding, benchmarking, runtime packaging, orchestration, and system integration. So the product hypothesis is consistent with where painful engineering work actually lives. 

On the specific pain points:

Calibration bottlenecks: still real, and well evidenced by QUAlibrate and Q-CTRL. 
Decoding bottlenecks: absolutely real; both technical papers and commercial QEC stacks say so. 
Low-latency classical processing bottlenecks: real and central; Quantum Machines and CUDA-Q Realtime are explicit on this. 
Experiment orchestration bottlenecks: real, but a wider category and therefore more dangerous for v1 scope. 
Benchmark / deployment / packaging bottlenecks: not as marketable in buzzword form, but technically very real, because public stacks already expose export, quantisation, and downstream runtime paths that someone still has to evaluate and operationalise. 
So the final cross-check is this.
Calibration triage is technically sound but less commercially attractive for your first build because it is integration-heavy and already occupied by strong workflow/control vendors.
Broader control-plane reliability is commercially tempting but technically too wide for a first serious product.
DecoderOps remains the idea supported by both serious technical understanding and current sector evidence. 

Final recommendation with commercialisation verdict
Final product name: Quantum Error-Correction DecoderOps Workbench.
Target customer: quantum hardware startups, QEC teams, control-stack vendors, and quantum cloud providers with an active interest in decoder selection and deployment-readiness.
Target user: QEC engineers, controls engineers, ML/performance engineers, and research software engineers.
Core stack: Stim, PyMatching, Sinter, NVIDIA Ising-Decoding, ONNX, ONNX Runtime, optional TensorRT, CUDA-Q, CUDA-QX QEC.
Deployment stance: local/on-prem first; customer-cloud optional later; no SaaS-first positioning.
Commercialisation verdict: Yes, commercially buildable, if you keep the core product in a clean local-first benchmark/package/report boundary and avoid making closed NVIDIA runtime pieces or cloud APIs mandatory dependencies.
Mostly Python: Yes.
Requires real hardware for MVP usefulness: No.
Calibration-VLM route: exclude from first build; keep as an optional research module later.
Internal local coding platform: use later as an acceleration layer; do not build first. 

The hard recommendation is therefore:

Build DecoderOps first.
Keep it narrowly centred on decoder benchmarking, artefact packaging, and deployment-readiness.
Use Ising-Decoding as a benchmarked and integrable technical foundation, not as your whole product story.
Treat CUDA-Q QEC as an export target and integration seam.
Keep TensorRT and other NVIDIA-licensed closed pieces optional.
Do not put Ising Calibration/Qwen in your v1 production stack unless a customer explicitly accepts that policy posture. 
Exact source links
The links below are the exact official URLs most relevant to the recommended build.

NVIDIA / Ising / CUDA-Q

https://developer.nvidia.com/ising
https://github.com/NVIDIA/Ising-Decoding
https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast
https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate
https://huggingface.co/nvidia/Ising-Calibration-1-35B-A3B
https://build.nvidia.com/nvidia/ising-calibration-1-35b-a3b
https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint
https://huggingface.co/datasets/nvidia/QCalEval
https://github.com/nvidia/QCalEval
https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes
https://research.nvidia.com/publication/2026-04_qcaleval-benchmarking-vision-language-models-quantum-calibration-plot
https://nvidia.github.io/cuda-quantum/latest/using/install/install.html
https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html
https://github.com/NVIDIA/cuda-quantum
https://nvidia.github.io/cudaqx/
https://nvidia.github.io/cudaqx/quickstart/installation.html
https://nvidia.github.io/cudaqx/components/qec/introduction.html
https://nvidia.github.io/cudaqx/examples_rst/qec/realtime_predecoder_pymatching.html
https://github.com/NVIDIA/cudaqx
Core open-source and runtime dependencies

https://github.com/quantumlib/stim
https://pypi.org/project/stim/
https://github.com/oscarhiggott/PyMatching
https://pymatching.readthedocs.io/
https://onnx.ai/
https://github.com/onnx/onnx
https://onnxruntime.ai/docs/install/
https://github.com/microsoft/onnxruntime
https://developer.nvidia.com/tensorrt
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html
https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
https://pytorch.org/get-started/locally/
https://github.com/pytorch/pytorch
https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html
https://docs.nvidia.com/cuda/cuquantum/latest/python/index.html
https://github.com/NVIDIA/cuQuantum
https://pypi.org/project/cuquantum-python/
Competitor and adjacent-product references

https://qualibrate-docs.quantum-machines.co/
https://www.quantum-machines.co/products/qualibrate/
https://www.quantum-machines.co/blog/shining-a-light-on-qec-controller-requirements-the-crucial-role-of-fast-classical-processing-for-quantum-computation-with-quantum-error-correction/
https://www.quantum-machines.co/blog/scalable-quantum-error-correction/
https://q-ctrl.com/technology/quantum-computer-autocalibration
https://docs.q-ctrl.com/boulder-opal
https://www.riverlane.com/quantum-error-correction-stack
https://www.riverlane.com/deltaflow
https://www.edencode.ai/
Licence references

https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html


Commercial path to a quantum DecoderOps product around NVIDIA Ising
Executive summary
The strongest first product is not a broad “AI for quantum” layer and not a calibration copilot. It is a narrow, on-prem-first QEC DecoderOps Workbench: a Python-heavy product that benchmarks, packages, validates, and deployment-prepares quantum error-correction decoder pipelines, starting with NVIDIA Ising-Decoding as the anchor artefact, plus established baselines such as PyMatching and Stim, and optional downstream validation with CUDA-Q QEC and TensorRT. That recommendation is driven by the public artefacts themselves: NVIDIA’s decoding stack already exposes a training framework, shipped pre-trained models, ONNX export, TensorRT paths, CUDA-Q QEC downstream hooks, and an accompanying research paper. By contrast, the calibration side is newer, more reference-like, and more commercially awkward because the visible public benchmark/model path is tied to a Qwen3.5-35B-A3B base model and an agent blueprint rather than a mature deployment-focused control product. 

The immediate commercial pain is not “invent a better decoder in the abstract”. It is operationalising decoder work: comparing decoder pipelines fairly, checking whether an AI pre-decoder actually improves end-to-end behaviour on a given detector error model, measuring latency and throughput honestly, exporting runtime artefacts, and handing QEC/runtime teams something reproducible enough to plug into a low-latency path. Public evidence across Google, IBM, Riverlane/Rigetti, PyMatching, and NVIDIA all points in the same direction: real-time decoding is now a live engineering bottleneck, especially for superconducting stacks where cycle times are on the order of microseconds and latency budgets are brutal. The current small-but-real product gap is not another research decoder; it is a vendor-neutral, reproducible benchmarking-and-deployment workbench that sits between research models and runtime integration. 

The commercial verdict is therefore straightforward. Technically possible: yes. Production-sensible: yes, if the first version stays focused on offline benchmarking, packaging, and report generation instead of pretending to be a live control-plane or real-time feedback engine. Commercially safe: yes, if you keep the core on permissive/open components and treat proprietary NVIDIA pieces such as TensorRT and cudaq-qec as customer-installed optional integrations, not bundled core dependencies. The calibration-VLM route should be treated as a later optional module, not as the centre of the first product. 

Best product opportunity
What the evidence says the strongest product is
The best first product is decoder benchmarking, packaging, and deployment tooling, not a calibration agent and not a general reliability control plane. The reason is simple: the public NVIDIA Ising artefacts are materially stronger on decoding than on calibration. NVIDIA’s public Ising page exposes Ising Decoding as a training framework with examples for real-time decoding, scripts for quantisation, and CUDA-Q QEC integration. The accompanying paper is explicitly about a modular AI pre-decoder that removes many physical errors before a downstream global decoder, with open-source implementation and end-to-end latency claims on GB300 hardware. The repo also documents export to ONNX, optional TensorRT inference, generation of .bin test data, and downstream ingestion into CUDA-Q QEC. That is already the skeleton of a benchmark-and-deployment product. 

The calibration route is real, but it is weaker as a first commercial foundation. NVIDIA’s public calibration artefacts currently consist of Ising Calibration 1, the QCalEval benchmark, and the Quantum Calibration Agent Blueprint. That is good research and good reference material, but the blueprint is exactly that: a reference agent workflow with CLI/Web UI, SQLite/HDF5 history, plot analysis, and support for external model providers. The benchmark paper itself frames the problem as a first benchmark for quantum calibration plots. That is useful, but it is still much closer to a reference workflow than to a hardened, deployment-centric product wedge. 

Where the pain is actually acute
The highest-pain customer segments are not all the same. The table below is the blunt version.

Customer type	Most urgent public pain	Best first wedge	Why
Superconducting-qubit startups	Low-latency classical decoding and calibration drift	DecoderOps first	Google reports a 1.1 μs cycle time with real-time decoding constraints, and Riverlane/Rigetti report sub-1 μs mean decoding time per round; superconducting calibration drift can also occur on sub-second timescales. That makes decoder deployment and calibration both painful, but decoder operationalisation is less served by incumbent control vendors. 
Error-correction teams	Decoder accuracy/latency trade-off, benchmarking, runtime packaging	DecoderOps first	PyMatching, Google, Riverlane, and NVIDIA all frame decoder speed as a core bottleneck. NVIDIA’s public stack already exposes export and downstream validation paths. 
Quantum hardware startups not yet deep into FT-QEC	Calibration throughput, workflow repeatability, data management	Calibration tooling later	IBM, Qibocal, QUAlibrate, Qruise, and LabOne Q all point to calibration orchestration as a standing operational burden, but this area already has strong incumbents. 
Control-stack vendors	Calibration/workflow orchestration, control usability, integration	Not your first wedge	This is already crowded by Quantum Machines, Zurich Instruments, Q-CTRL, and Qruise. A broad reliability platform is too wide for a first build. 
Quantum cloud providers	Fleet stability, benchmark reproducibility, logical-stack preparation	DecoderOps, but later	IBM explicitly treats monitoring, calibration, and benchmarking as ongoing fleet operations. A decoder deployment workbench becomes relevant once logical services are on the roadmap, but it is not the easiest first beachhead unless the provider is already building QEC services. 
Neutral-atom companies	Architecture-specific control and calibration, measurement/feed-forward constraints	Calibration/workflow first, DecoderOps second	QCalEval spans superconducting and neutral-atom plots, but neutral-atom QEC still faces severe measurement/feed-forward timing penalties; public literature points to measurements on the order of hundreds of microseconds, making real-time decoder work less immediate than in superconducting stacks. 

Candidate product evaluation
Candidate	What it actually does	Buyer	User	Narrow enough for a serious first build	Mostly Python	Needs real hardware to be useful	Commercial verdict
Quantum Error-Correction DecoderOps Workbench	Benchmarks decoder pipelines, compares AI pre-decoders versus classical decoders, manages detector error models and syndrome workflows, exports ONNX/TensorRT/runtime artefacts, generates reproducible reports	Head of QEC, VP Engineering, runtime lead, FT programme lead	QEC researchers, runtime engineers, systems engineers	Yes	Yes	No for MVP; customer hardware/logs improve pilots	Best first product 
Agentic Quantum Calibration Triage and Replay Workbench	Looks at calibration outputs, parameter fits, drift/failure triage, plot reasoning, audit trail, replay/branching	Head of hardware, calibration lead, lab operations lead	Experimental physicists, calibration engineers	Borderline; can drift into generic control software	Yes	Public demo can work without hardware; paid value usually needs customer workflows	Technically sound, but not the strongest first wedge 
Quantum Control-Plane Reliability Workbench	Cross-cutting reliability overlay across calibration, scheduling, experiment orchestration, anomaly review, audit	CTO, platform lead	Mixed lab/control/runtime teams	No; too broad	Yes at first, but scope balloons quickly	Usually yes, to prove value	Commercially tempting, but too wide and too incumbent-contested 

What should be built first
Build Quantum DecoderOps Workbench first. The deciding facts are these:

The public NVIDIA artefacts are strongest on decoding, not calibration. 
Real-time decoding is demonstrably a live sector bottleneck, especially on superconducting hardware. 
A small software team can ship value without owning hardware, because public generators and benchmark tooling already exist. 
The product can be kept commercially conservative by making permissive/open tooling the core and treating proprietary runtime integrations as optional customer-side plugins. 
Verified artefact and licensing matrix
Public NVIDIA Ising stack
The currently visible public Ising stack breaks down as follows. Where I have not been able to verify a point cleanly from the current parsed source, I say so directly instead of pretending certainty.

Artefact	Exact public name	Official page	Official repo	Type	Licence	Commercial use	Redistribution	Local deployment	Cloud deployment	Evidence status
Family landing page	NVIDIA Ising	NVIDIA Developer family page	Support link to GitHub from family page	Product/family page	N/A	N/A	N/A	Documents local flows	Documents model/API links	Verified 
Calibration model	Ising Calibration 1 / nvidia/Ising-Calibration-1-35B-A3B	NVIDIA family page and NIM page	Hugging Face repo	Open-weight VLM	[Unverified current model-card licence in this session]	[Unverified exact weight-card terms]	[Unverified]	Downloadable weights imply local use is intended	NIM/API page exists	Mixed: repo/page existence verified; current exact weight terms unverified 
Calibration benchmark	QCalEval dataset	NVIDIA Research paper page	NVIDIA/QCalEval for scripts; HF dataset for data	Benchmark dataset + scripts	Dataset CC-BY-4.0; scripts Apache-2.0	Dataset card says ready for commercial/non-commercial use	Dataset redistributable with attribution; scripts redistributable under Apache	Yes	Yes	Verified 
Calibration workflow	Quantum Calibration Agent Blueprint	Linked from NVIDIA Ising page	NVIDIA/Quantum-Calibration-Agent-Blueprint	Reference agent blueprint	Apache-2.0	Yes	Yes	Yes; Python 3.11+, local CLI/Web UI	Can use NVIDIA API Catalog or other APIs	Verified 
Decoding framework	Ising Decoding	NVIDIA family page	NVIDIA/Ising-Decoding	Training/inference/export framework	Apache-2.0	Yes	Yes	Yes	Indirectly, via repo/runtime integrations	Verified 
Decoder models	Ising Decoder SurfaceCode 1 Fast and Accurate	NVIDIA family page; shipped .pt files in repo	Available via HF and Git LFS in repo	Pre-trained decoder models	Repo governed by Apache-2.0 unless otherwise noted	Yes, on repo terms	Yes, on repo terms	Yes	Yes via export/runtime paths	Verified for repo-shipped model files; [Unverified current separate HF-card wording] 
CUDA-Q integration point	CUDA-Q	CUDA-Q docs / PyPI	NVIDIA/cuda-quantum	Quantum-classical framework	Apache-2.0, with cuQuantum under separate licence for accelerated simulation	Yes	Yes, on Apache terms; watch cuQuantum runtime terms	Yes	Yes	Verified 
CUDA-QX QEC integration point	cudaq-qec / CUDA-Q QEC	CUDA-QX docs / PyPI	NVIDIA/cudaqx	QEC library/runtime layer	Source mostly Apache-2.0; PyPI wheel marked proprietary because it ships closed libcudaq-qec-nv-qldpc-decoder.so	Yes, but not as a clean permissive dependency	Redistribution constrained by proprietary component	Yes on Linux	Plausible, but not as a carefree bundled dependency	Verified 

Which parts are production-relevant and which are still reference-grade
Production-relevant today: the Ising-Decoding repo, its shipped models, ONNX/TensorRT export path, Stim/PyMatching baseline path, and the downstream CUDA-Q QEC validation hook. Those pieces line up with a real engineering workflow: train or reuse a model, benchmark against baseline decoders, export artefacts, generate syndrome test data, and validate compatibility with a runtime path. That is enough substance to build a commercially credible workbench around. 

Reference-grade today: the Quantum Calibration Agent Blueprint and, to a lesser extent, the calibration VLM route as a deployable first product. The blueprint is useful and well-scoped, but NVIDIA itself describes it as a reference agent blueprint. QCalEval is a proper benchmark and good evidence that calibration-plot understanding matters, but it is still a benchmark paper and dataset, not proof of hardened deployment into customer lab operations. 

Ising Calibration specifics
Here is the honest read.

The exact public Hugging Face repo is nvidia/Ising-Calibration-1-35B-A3B. NVIDIA’s QCalEval paper states that Ising Calibration 1 is “an open-weight model based on Qwen3.5-35B-A3B”. 
Qwen is a model family of the Qwen Team, Alibaba Group / Alibaba Cloud. If you have a policy of no Chinese-origin base model in the active production stack, this route conflicts with that policy. That is not ideology; it is simple provenance. 
The exact current model-card statements on minimum GPU requirement, runtime expectations, and the weight-card licence tag were not cleanly recoverable from parsed lines in this session, so those items remain [Unverified] here and must be re-checked directly on the current model card before you put them into procurement or customer paperwork.
Because of that provenance issue and because the public calibration stack is still more blueprint/benchmark than deployment infrastructure, I would not put Ising Calibration at the centre of the first commercial product. That would be an avoidable self-own. 
Ising Decoding specifics
The decoding side is much cleaner.

The exact repo is NVIDIA/Ising-Decoding. It is an Apache-2.0 repo and explicitly targets Python 3.11, 3.12, 3.13. 
The repo ships two pre-trained public model files: models/Ising-Decoder-SurfaceCode-1-Fast.pt and models/Ising-Decoder-SurfaceCode-1-Accurate.pt. 
The public release describes a pipeline where the neural network consumes detector syndromes across space and time, predicts corrections that reduce syndrome density, and then a standard decoder such as PyMatching makes the final logical decision. That architecture is exactly the right level for a deployment workbench: not replacing everything, but compositing with existing decoders. 
Export/runtime paths are concrete: PyTorch, ONNX, TensorRT, plus .onnx and .bin generation for CUDA-Q QEC downstream ingestion. 
The repo also snapshots config files and logs under structured output directories, which is useful for a reproducibility-focused product. 
Final licensing judgement on the recommended stack
This is the licence picture that actually matters for commercialisation.

Component	Licence	Commercial use	Redistribution implications	Hosted SaaS / cloud implications	On-prem implications	Build verdict
Ising-Decoding repo	Apache-2.0	Yes	Standard Apache notice obligations	Fine	Fine	Safe core 
CUDA-Q	Apache-2.0; accelerated simulation leans on cuQuantum under separate licence	Yes	Apache for code; review cuQuantum if distributing bundled accelerated runtime	Fine if terms respected	Fine	Safe core, with cuQuantum awareness 
Stim	Apache-2.0	Yes	Standard Apache obligations	Fine	Fine	Safe core 
PyMatching	Apache-2.0	Yes	Standard Apache obligations	Fine	Fine	Safe core 
ONNX	Apache-2.0	Yes	Standard Apache obligations, including NOTICE/changes where relevant	Fine	Fine	Safe core 
ONNX Runtime	MIT	Yes	Very permissive	Fine	Fine	Safe core 
TensorRT SDK	Proprietary NVIDIA SDK licence	Yes, but subject to SDK terms	You may distribute only portions identified as distributable, inside an application with material additional functionality; open-source contamination restrictions also apply	Do not casually embed in a public SaaS build without legal review	Better as customer-installed runtime	Use as optional plugin, not bundled core 
cudaq-qec wheel	PyPI package marked proprietary; most source open, one closed library shipped	Yes, but not as a carefree open dependency	Closed libcudaq-qec-nv-qldpc-decoder.so changes redistribution posture	Treat as customer-side integration, not your redistributed base image	Fine if customer installs in their environment	Optional plugin only 
QCalEval dataset	CC-BY-4.0	Yes	Attribution required	Fine	Fine	Safe for optional eval content 
Quantum Calibration Agent Blueprint	Apache-2.0	Yes	Standard Apache obligations	Fine	Fine	Optional later module 

The practical conclusion is harsh and simple: do not build your first commercial product around a dependency stack that forces you to redistribute proprietary NVIDIA binaries by default. Build the product so that proprietary pieces are optional adapters installed by the customer in their own environment. That keeps your own deliverable commercially cleaner and makes procurement discussions easier. 

Product architecture and MVP
Recommended architecture
The recommended first architecture is a local-first benchmark-and-packaging workbench, not a live controller.

The ingestion layer should accept three things: synthetic circuits and detector error models generated with Stim, decoder artefacts from Ising-Decoding, and customer-supplied artefacts such as detector error models, syndrome logs, or baseline decoder outputs when pilots begin. This is enough to support public demos now and private replay later. Stim and PyMatching already expose a well-trodden path for generating circuits and DEMs, while Ising-Decoding provides its own test-data generation utility for downstream CUDA-Q QEC use. 

The experiment runner should be a Python orchestration layer that sweeps code distance, number of rounds, noise parameters, basis choice, backend path, and model choice. The point is not scientific novelty. The point is reproducible execution and result collation. Ising-Decoding already centralises execution through a single user-facing config and local runner script, which is exactly the right pattern to wrap rather than re-invent. 

The model/decoder execution layer should support four backend modes in the first serious build: PyMatching-only baseline, Ising pre-decoder + PyMatching, ONNX-export validation, and optional TensorRT inference. The repo’s documented ONNX_WORKFLOW values map cleanly to this. CUDA-Q QEC should sit one step downstream as a compatibility target, not as a hard MVP dependency. 

The metrics/evaluation layer should track at least: logical error rate, residual syndrome density, latency, throughput, export success, runtime compatibility status, and reproducibility status. NVIDIA’s repo already treats LER / SDR / latency as public-release inference metrics, and it exposes a PyMatching speedup metric as well. 

The artefact packaging layer should write model references, config snapshots, logs, exported ONNX files, any TensorRT engine files, generated syndrome .bin files, and a manifest describing exactly how the run was produced. The repo already writes models, TensorBoard output, config snapshots, and logs into predictable directories. Your product’s job is to make that customer-friendly and auditable. 

The report generation layer should emit an HTML/PDF engineering report, a machine-readable results bundle, compatibility notes, and a risk register. This is where the product earns its keep commercially: converting a pile of QEC runs into a purchaseable engineering deliverable. Qibocal’s emphasis on human- and machine-readable reports is a useful precedent here, even though your product is not a calibration framework. 

The optional AI-assisted analysis layer should come later and should be constrained. It can summarise run deltas, flag suspicious regressions, and draft technical notes, but it should not be central to the trust boundary of v1. The first commercial version has to win on reproducibility, not on “agentic” theatre. That is one of the places you are most likely to waste time if you let yourself. 

MVP boundary
The exact MVP boundary should be this:

Run public surface-code memory benchmarks with PyMatching and Ising-Decoding.
Export ONNX and validate that the artefacts load.
Generate .bin data for downstream CUDA-Q QEC compatibility testing.
Produce a reproducible report with config snapshots, metrics, and artefact manifest.
Support customer-private replay of detector error models or syndrome traces in a paid pilot. 
What v1 should deliberately exclude:

Live low-latency classical feedback on hardware.
Broad calibration orchestration.
Full control-plane reliability dashboards.
Claims of universal decoder superiority.
Native multi-GPU runtime promises. The current CUDA-QX real-time AI predecoder example explicitly says multi-GPU dispatch is not yet supported and clamps --num-gpus to 1. 
What the product should guarantee and what it should not claim
The product should guarantee reproducible decoder benchmarking, artifact export, configuration traceability, and deployment-readiness assessment against named runtime paths. It should also guarantee that the customer can rerun the exact benchmark package locally, on-prem, without shipping raw data out of their environment. Those are sane promises. 

It should not claim that it provides the best decoder for every hardware stack, that it is itself a real-time decoder controller, or that NVIDIA’s public reference benchmarks automatically generalise to customer noise. NVIDIA’s own public Ising benchmark claims are tied to specific noise settings such as d=13, p=0.003 and the surface-code SI1000-style context. Google’s and Riverlane’s public results also show that real-time decoder engineering is tightly coupled to system architecture. If you claim universal wins, you are bluffing. 

Programming reality and deployment model
Python versus quantum-specific coding
For the recommended product, most of the work is ordinary Python systems engineering, not deep quantum-algorithm development. Publicly, Ising-Decoding exposes one user-facing config, one local runner, Python requirements files, and export utilities. The calibration blueprint is also a Python-heavy project. That means your actual burden is orchestration, result management, config handling, packaging, and integration glue. 

The quantum-specific part is real, but narrower than people romanticise. You will need to understand detector error models, syndrome streams, logical observables, code distance, round structure, and what counts as a correct decoding outcome. PyMatching’s documentation and examples are explicit about decoding success being defined via logical observables, and NVIDIA’s repo is explicit about the pre-decoder/global-decoder split. That is specialised domain knowledge, but it is not the same thing as writing novel gate-model algorithms. 

So the honest split is this. Mostly Python systems engineering: runner, config, artefact registry, metrics, reports, manifest generation, CLI/UI, and customer replay support. Quantum software engineering: benchmark design, DEM handling, interpretation of LER/SDR/latency trade-offs, understanding when a comparison is fair. CUDA-Q / CUDA-QX integration: thin wrappers and validation around exported artefacts. Possible C++ requirements: only if you want to own or customise the current CUDA-Q Realtime AI predecoder path, which NVIDIA documents as a C++ demonstration built from source and not part of distributed binaries. 

That means the answer to your own hidden question is: no, you do not need to build a full internal local quantum coding platform first. It is optional. It becomes useful later as an internal acceleration layer for templating, orchestration, repository code generation, and report assembly. But if you build that platform first, you will probably use it as a very sophisticated avoidance mechanism. Build the product, not the internal empire. 

Local, customer-cloud, or hosted
The first build should be local-first with optional customer-cloud deployment, not SaaS-first. Every serious signal in the sector points that way. IBM explicitly describes calibration, monitoring, and benchmarking as part of maintaining a fleet over time. Qibocal is explicitly designed for self-hosted QPUs. QUAlibrate, QruiseOS, and LabOne Q are all built around close integration with hardware and control environments. In other words, the real engineering data is not something these teams are eager to stream into your hosted service. 

The reasons are obvious: detector error models encode device behaviour; syndrome logs can reveal architecture-specific characteristics; calibration histories expose drift and weak points; control-stack details, hardware topologies, pulse settings, and experiment graphs are all commercially sensitive. In some cases, there will also be customer or government constraints around data locality and export. So the realistic posture is on-prem first, with customer-controlled cloud later for less sensitive pilot deployments. 

The only parts that make sense to host centrally are the least sensitive ones: synthetic-public benchmark packs, report templates, optional documentation portals, and maybe a lightweight licensing/update service. Do not design v1 around hosted execution of private calibration or syndrome data. That is strategically stupid for this market. The standard enterprise flavour of this sector is closer to EDA and scientific instrumentation than to generic SaaS. 

Exact toolchain and commands
Recommended core stack
For the first product, the recommended core stack is:

Safe core, technically sufficient for MVP: Python 3.12 on Ubuntu 24.04, Stim, PyMatching, NVIDIA/Ising-Decoding, optional onnxruntime for smoke tests, and your own Python orchestration/reporting layer. 
Production-sensible optional layer: CUDA-Q and customer-installed cudaq-qec for runtime compatibility checks; TensorRT only when you need engine-level deployment validation. 
Excluded from first build: Ising Calibration as a core dependency, because of provenance/policy risk and because the public calibration path is less deployment-ready. 
Ubuntu 24.04 is a sensible choice. CUDA-Q documents Ubuntu 24.04 as supported, Python 3.11+, and Blackwell support for CUDA 13.x only. NVIDIA also documents known Blackwell issues for CUDA 12.8 and notes that some torch-integrator workflows may require nightly torch on Blackwell in that configuration. That means your one-GPU-now setup is fine, but you should avoid pretending that the whole stack is frictionless on every Blackwell/CUDA combination. 

Copy-paste-ready install commands
Chosen path for the first serious build
This path is the one I would actually use first: keep the core open, stay in Python, and do export/validation without forcing proprietary runtime packaging into day one.

bash
Copy
sudo apt-get update
sudo apt-get install -y git git-lfs python3-venv build-essential pkg-config

git lfs install

python3 -m venv ~/venvs/decoderops
source ~/venvs/decoderops/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Core public QEC tooling
pip install stim --upgrade
pip install pymatching --upgrade
pip install onnxruntime

# NVIDIA Ising-Decoding
git clone https://github.com/NVIDIA/Ising-Decoding.git
cd Ising-Decoding
git lfs pull

# NVIDIA's public repo says cu130 is known to work; inference-only install first
export TORCH_CUDA=cu130
pip install -r code/requirements_public_inference.txt
bash code/scripts/check_python_compat.sh
Those commands are grounded in the Ising-Decoding repo’s own documented requirements and install flow, plus the official PyPI install paths for Stim and PyMatching. 

Fallback path if you need CUDA-Q and CUDA-Q QEC validation
bash
Copy
source ~/venvs/decoderops/bin/activate

# CUDA-Q
pip install cudaq

# Optional downstream QEC validation library
pip install cudaq-qec
This is the official pip path documented by CUDA-Q and CUDA-QX. The catch is legal and packaging, not installation difficulty: cudaq-qec is not a clean permissive dependency because the wheel ships a closed library. Use it as a customer-installed integration, not as something you silently bundle and redistribute. 

Optional TensorRT path
bash
Copy
source ~/venvs/decoderops/bin/activate
python -m pip install --upgrade tensorrt
This is the official pip route for TensorRT. Use it only when you are ready to test deployment engines and you have already decided how you will handle redistribution and SDK-term compliance. 

Useful verified commands from the NVIDIA Ising-Decoding workflow
Run inference with the shipped models
bash
Copy
cd Ising-Decoding
source ~/venvs/decoderops/bin/activate

WORKFLOW=inference bash code/scripts/local_run.sh
Export ONNX
bash
Copy
cd Ising-Decoding
source ~/venvs/decoderops/bin/activate

ONNX_WORKFLOW=1 WORKFLOW=inference bash code/scripts/local_run.sh
Export ONNX and test TensorRT inference
bash
Copy
cd Ising-Decoding
source ~/venvs/decoderops/bin/activate

ONNX_WORKFLOW=2 QUANT_FORMAT=int8 WORKFLOW=inference bash code/scripts/local_run.sh
Generate syndrome test data for downstream CUDA-Q QEC ingestion
bash
Copy
cd Ising-Decoding
source ~/venvs/decoderops/bin/activate

python3 code/export/generate_test_data.py \
  --distance 13 \
  --n-rounds 104 \
  --num-samples 10000 \
  --basis X \
  --p-error=0.003 \
  --simple-noise
These are all directly documented in the public repo. 

Support and requirement notes that matter
CUDA-Q: Linux x86_64 and aarch64 are supported; Ubuntu 24.04 is listed; Python 3.11+ on the docs and Python ≥3.10 on PyPI; GPU acceleration is Linux-only; Blackwell is documented as supported for CUDA 13.x only, with known issues on CUDA 12.8. 
Ising-Decoding: Python 3.11–3.13; inference-only and training requirement files are provided. 
cudaq-qec: Linux only, Python ≥3.11, GPU optional for some components, but packaging/licensing is not purely open. 
CUDA-QX real-time AI predecoder example: current public example is a C++ demonstration built from source, not part of distributed binaries; --num-gpus is currently clamped to 1. That is why it is a stage-two compatibility target, not your MVP core. 
Benchmark, pilot, competitor, and knowledge-base and sector cross-check
Meaningful demo and paid pilot design
A technically credible public demo is not “look, an AI model decodes quantum errors”. That is fluff. The credible demo is a reproducible engineering bake-off.

Use public surface-code memory circuits and detector error models generated with Stim. Run PyMatching as the baseline and Ising-Decoding Fast/Accurate as the AI pre-decoder variants. Sweep code distance and noise settings, export ONNX, optionally test TensorRT, and generate CUDA-Q QEC-compatible test artefacts. Then produce a report that says, in plain engineering terms: here is the logical error rate, here is the latency, here is the throughput, here is what exported cleanly, here is what did not, and here is where the pipeline is or is not runtime-ready. That is the sort of thing a real QEC or runtime team can use. 

The most meaningful metrics are these:

Logical error rate: success or failure of the logical observable prediction, which is the operational metric that matters. PyMatching’s docs define decoding success in terms of the logical observables matching the true noise. 
Residual syndrome density: not a universal glamorous KPI, but a useful engineering metric for a pre-decoder because NVIDIA’s public description is explicitly about reducing syndrome density before the downstream decoder. In your product, define it precisely and consistently as a post-predecoder residual-activation metric, not as vague jargon. 
Latency: especially per round or per shot, because real-time systems are constrained by cycle times and reaction-time budgets. Google reports a real-time decoder latency of about 63 μs at distance 5 with a 1.1 μs cycle time; Riverlane/Rigetti report mean decoding time per round below 1 μs. 
Throughput: if the pipeline cannot keep up with syndrome generation, it eventually becomes useless. That is the explicit point made in PyMatching’s sparse blossom work and in real-time hardware papers. 
Exportability and runtime compatibility: can the model be exported, quantised if needed, and ingested downstream? Ising-Decoding’s ONNX/TensorRT/CUDA-Q QEC path makes this measurable, not theoretical. 
Reproducibility: config snapshotting, log capture, and artefact manifesting. Without that, you are selling colourful charts. 
A credible paid pilot would look like this: on-prem installation; ingestion of one or two customer baselines plus one customer DEM or syndrome trace set; reproduction of the baseline; comparison against Ising-style pre-decoder pipelines; export and compatibility tests; final hand-off of a benchmark report, artefact bundle, and risk register. The value is not that you own the customer’s decoder destiny. The value is that you compress their deployment-prep loop from hand-rolled scripts and ad hoc notebooks into something disciplined and repeatable. That is plausible enough to sell. 

Competitor and adjacent-product analysis
Vendor / project	What it appears to offer	Problem solved	Gap that still exists
NVIDIA / CUDA-Q / CUDA-QX / Ising	Core frameworks, decoder training/export, runtime examples, calibration benchmark + blueprint	Enables model development and downstream QEC integration	Does not itself look like a vendor-neutral benchmark-and-packaging product for customer pilots; still exposes customers to raw tooling complexity 
Quantum Machines / QUAlibrate	Web-based calibration workflow management, visualisation, node/graph orchestration; tied to OPX/QUA ecosystem	Calibration execution and orchestration	Strong in calibration, not the same thing as decoder deployment benchmarking; vendor-specific stack 
Q-CTRL / Boulder Opal	Quantum control infrastructure software integrated into research and hardware workflows	Control optimisation and hardware performance	Strong incumbent for control/calibration adjacent work; not the same as a decoder-ops packaging workbench 
Zurich Instruments / LabOne Q	Integrated Python-based quantum experiment/control system for QCCS and allied setups	Instrument and experiment control/orchestration	Control-plane incumbent; does not fill the vendor-neutral decoder benchmarking niche 
QruiseOS / QruiseML	Automated characterisation, calibration, optimal control, digital twins	Bring-up and calibration automation	Calibration-heavy incumbent; again, not your first decoder wedge 
EdenCode	Official public material is still thin; appears positioned around AI-based real-time QEC decoding	Real-time decoder startup	Relevant adjacent threat, but public technical detail is limited; a neutral benchmarking/deployment prep product could still coexist if it validates multiple decoder paths rather than selling a single decoder

The gap is therefore real, but narrow. There is room for a product that is vendor-neutral, benchmark- and deployment-focused, on-prem-first, and aimed at small to mid-sized quantum teams that are not ready to build an internal DecoderOps stack from scratch but do not want to buy themselves wholesale into a single control vendor’s ecosystem. That is the niche. If you drift into general control software, you will get buried by incumbents. If you stay in decoder benchmarking, packaging, and runtime readiness, you have a cleaner wedge. 

Knowledge-base and sector cross-check
After cross-checking the product ideas against established quantum engineering material and current public sector evidence, the recommendation still holds: DecoderOps is the right first product.

The foundational technical picture is stable. Fault-tolerant quantum computing requires error correction; error correction requires classical decoding; and for physical platforms with fast cycles, the decoding path has to keep up. That is not vendor marketing. It is visible in Google’s below-threshold surface-code result, in Riverlane/Rigetti’s low-latency FPGA work, in the PyMatching sparse-blossom paper, and in NVIDIA’s own decoding paper. The older, more general technical knowledge says the same thing: classical decoding is part of the quantum machine, not an optional afterthought. 

The calibration side is also unquestionably real. IBM’s documentation describes calibration as a parameter-heavy, continuously monitored process subject to drift. Recent superconducting work reports sub-second drift and millisecond-scale recalibration/benchmarking workflows. Qibocal, QUAlibrate, Qruise, and LabOne Q all exist because calibration and experiment management are serious engineering burdens. So calibration pain is real; it is not hype. 

Where the older technical picture and the current market picture diverge slightly is in commercial urgency and product wedge, not in technical truth. Both calibration and decoding matter technically. But current market evidence suggests that calibration already has several serious workflow/control incumbents, while the public NVIDIA Ising release has made decoder deployment preparation unusually tangible: there is now a visible open framework, export path, and runtime-adjacent example, but not yet an obvious vendor-neutral product that turns those into customer-grade benchmark and deployment workflows. That makes the calibration path technically sound but less commercially differentiated, and the decoder path both technically sound and commercially better timed. 

The pain-point check comes out like this:

Calibration bottlenecks: still real and current. 
Decoding bottlenecks: still real and current. 
Low-latency classical processing bottlenecks: still real and current, especially for superconducting stacks. 
Experiment orchestration bottlenecks: still real and current, but already more crowded commercially. 
Benchmark / deployment / packaging bottlenecks: real enough to matter commercially because the public tools expose them directly, but no obvious product abstraction solves them cleanly yet. 
So the final cross-check is blunt: the recommended product is still the right one after comparing core knowledge, current sector evidence, and real engineering pain versus hype. Calibration is not fake; it is simply not the cleanest first commercial wedge for you.

Final recommendation with commercialisation verdict
Exact final recommendation
Final product name: Quantum DecoderOps Workbench

One-sentence technical description: A local-first engineering workbench that benchmarks, compares, packages, and deployment-validates QEC decoder pipelines — starting with NVIDIA Ising-Decoding, PyMatching, and Stim — and emits reproducible artefacts and runtime-readiness reports for customer environments. 

Target customer: superconducting-qubit startups, QEC teams inside hardware companies, and quantum platform groups that are moving from decoder research into deployment preparation. Target user: QEC researchers, runtime engineers, and systems engineers. This is strongest for teams that already feel the pain of latency/accuracy trade-offs and packaging work. 

Core stack: Python orchestration + Stim + PyMatching + NVIDIA Ising-Decoding + ONNX export + optional ONNX Runtime smoke tests; optional customer-installed CUDA-Q / cudaq-qec / TensorRT integrations later. 

Deployment stance: local/on-prem first; optional customer-cloud later; not SaaS-first. 

Commercially buildable: yes, with one key boundary: do not make proprietary NVIDIA runtime components part of your redistributed core image by default. Keep those as optional integrations installed in the customer’s environment. 

Implementable mostly in Python: yes. Requires real hardware for MVP usefulness: no. Requires direct quantum algorithm coding: no, beyond understanding QEC data structures and fairness in comparisons. 

Final stack and product boundary
The final recommended stack for v1 is:

Your own Python product layer
Stim
PyMatching
NVIDIA/Ising-Decoding
onnxruntime for basic artifact validation
Report generator + manifest/artefact store
Optional later adapters: cudaq, cudaq-qec, TensorRT 
The final product boundary is equally important:

Include: benchmark runner, decoder comparison, artefact export, reproducible reporting, customer-private replay.
Exclude initially: live control, live hardware feedback, calibration-VLM-first workflows, broad reliability control plane, bundled proprietary runtimes, and universal decoder performance claims. 
Calibration VLM route and internal platform decision
The calibration-VLM route should be excluded initially, or at most treated as a later optional module. It is promising, but the public calibration stack is still more benchmark/blueprint than deployable wedge, and the Qwen base-model provenance is enough on its own to make it an awkward first dependency if you want a conservative commercial posture. 

Your internal local quantum coding/orchestration platform should not be built first. It should be used later as an internal acceleration layer once you already have the product skeleton. The only pieces of that internal platform that materially accelerate this product are environment provisioning, run templating, artifact registry, report assembly, and code generation for benchmark wrappers. Everything else is secondary. 

Appendix of official links, licences, commands, papers, and unresolved risks
The links below are the exact official pages and repos I verified in this research.

text
Copy
NVIDIA Ising family page
https://developer.nvidia.com/ising

NVIDIA Ising-Decoding repository
https://github.com/NVIDIA/Ising-Decoding

NVIDIA Ising Calibration model repo
https://huggingface.co/nvidia/Ising-Calibration-1-35B-A3B

NVIDIA Ising Decoder SurfaceCode 1 Fast
https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast

NVIDIA Ising Decoder SurfaceCode 1 Accurate
https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Accurate

NVIDIA QCalEval dataset
https://huggingface.co/datasets/nvidia/QCalEval

NVIDIA QCalEval evaluation scripts
https://github.com/nvidia/QCalEval

NVIDIA Quantum Calibration Agent Blueprint
https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint

CUDA-Q documentation
https://nvidia.github.io/cuda-quantum/latest/

CUDA-Q PyPI
https://pypi.org/project/cudaq/

CUDA-QX documentation
https://nvidia.github.io/cudaqx/

CUDA-Q QEC PyPI
https://pypi.org/project/cudaq-qec/

CUDA-Q Realtime AI predecoder example
https://nvidia.github.io/cudaqx/examples_rst/qec/realtime_predecoder_pymatching.html

Stim PyPI
https://pypi.org/project/stim/

PyMatching PyPI
https://pypi.org/project/PyMatching/

ONNX repository
https://github.com/onnx/onnx

ONNX Runtime install docs
https://onnxruntime.ai/docs/install/

TensorRT install docs
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html

TensorRT pip install docs
https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html

cuQuantum getting started
https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html

NVIDIA Open Model License
https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

TensorRT SDK licence
https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html
The core papers and primary technical references that matter most here are the NVIDIA Ising decoding paper, the QCalEval paper, Google’s below-threshold surface-code paper, PyMatching’s sparse-blossom paper, and the Riverlane/Rigetti low-latency QEC paper. Those are the load-bearing technical sources behind the recommendation. 

The main unresolved risks are these:

[Unverified] exact current model-card licence and hardware minima for the Ising Calibration weights; re-check before any commercial use.
Verified proprietary packaging risk around TensorRT and cudaq-qec if you redistribute them incorrectly. 
Verified Blackwell friction on some CUDA-Q / torch configurations; do not promise effortless support across all driver/CUDA/torch combinations. 
Inferred but important: a vendor-neutral benchmark workbench is commercially plausible, but it is still a niche product. You will need paid pilots with teams already feeling decoder pain, not generic “quantum AI” curiosity.
