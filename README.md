# ITA0520 Computer Vision — Assignment 2 (CO2, Set B)
## Underwater Coral Image Restoration and Segmentation

A reproducible Python/OpenCV implementation of all ten assignment tasks.

### Dataset and evidence note
The project generates 30 deterministic **synthetic** underwater coral scenes because no real underwater dataset was supplied. The scenes include coral-like branches, pale damage-like patches, algae-like distractors, sandy backgrounds, depth-dependent colour casts, haze and noise. These are controlled illustrations, not real coral photographs or expert-labelled damage masks.

All CSV timings and image statistics are measured by executing the included algorithms. They are diagnostic results on synthetic data and must not be generalized to real underwater footage.

### Scenario
Assignment scenario: 2048×1536 images, grey intensity 35–170, 3000 frames/week, target regions about 12 pixels, CPU-only lab. The project uses 512×384 working images to keep execution practical on a CPU; it can be adapted to full-resolution images.

### Structure
- `src/dataset_generator.py`: deterministic image generation
- `src/run_all.py`: all ten task pipelines
- `data/synthetic/`: 30 labelled images and labels.csv
- `outputs/`: side-by-side comparison images
- `results/`: ten task-wise CSVs and run_summary.json
- `report/`: reserved for report deliverables

### Run
Python 3.10+ recommended:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

### Implemented alternatives
1. Linear stretch, gamma, log mapping.
2. Global equalization, CLAHE, histogram matching.
3. Mean, median, bilateral filters.
4. Laplacian, unsharp masking, high-boost.
5. Ideal, Butterworth, Gaussian low-pass.
6. Ideal, Butterworth, Gaussian high-pass.
7. Sobel, Prewitt, Canny.
8. Global threshold, Otsu, adaptive threshold.
9. Seeded region growing, connected components, watershed.
10. Contour chaining, morphological linking, region-edge fusion.

### Limitations
- Synthetic images are not a representative underwater dataset.
- There are no expert pixel masks; contrast, edge, area and component counts are diagnostic proxies, not damage-segmentation accuracy.
- Task 2 histogram matching uses one generated shallow scene as a reference.
- Task 9 region growing is a basic seeded 4-neighbour implementation.
- Task 10 algae-like distractors are synthetic and need expert-labelled real data for validation.
- Runtime depends on hardware, OpenCV build and image size. Per-operation timing does not prove an end-to-end one-second guarantee.

See `results/` and `outputs/` after execution.
