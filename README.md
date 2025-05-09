# math-156-project
# Math 156 Final Project
## By Jacob Bianchi, Justin Gong, Danelle Lizardo, David Oplatka, Ved Phadke

## Setup

1.  Ensure you have Python and PyTorch installed.
2.  The required datasets (e.g., MNIST) will be downloaded automatically by the scripts if not present in the `data/` directory.

## Running Experiments

Experiments are defined by YAML configuration files in the `experiments/` directory. An example, `experiments/sharp_curvy_mnist.yaml`, is provided for a catastrophic forgetting task (training on "sharp" MNIST digits, then "curvy" MNIST digits).

**Paths in YAML:**
*   `base_model_dir`: Directory to save/load models (e.g., "models/"). Relative to project root.
*   `data_dir`: Root directory for datasets (e.g., "data/"). Relative to project root.

### Example: Sharp vs. Curvy Digits Experiment

The `experiments/sharp_curvy_mnist.yaml` defines two training tasks and four evaluation steps.

**1. Train on Task 1 (Sharp Digits):**
   Open your terminal in the project root directory (`math-156-project/`) and run:
   ```bash
   python src/train.py --config experiments/sharp_curvy_mnist.yaml --task task1
   ```
**2. Train on Task 2 (Curvy Digits):**
   Run command
   ```bash
   python src/train.py --config experiments/sharp_curvy_mnist.yaml --task task1
   ```
**3. Eval on Task 1 with Task 1 model**
   ```bash
   python src/evaluate.py --config experiments/sharp_curvy_mnist.yaml --eval_name eval_task1_after_task1
   ```  
**4. Eval on Task 2 w/Task 1 model**
   ```bash
   python src/evaluate.py --config experiments/sharp_curvy_mnist.yaml --eval_name eval_task2_after_task1
   ```  
**5. Eval on Task 1 w/Task 2 model**
   ```bash
   python src/evaluate.py --config experiments/sharp_curvy_mnist.yaml --eval_name eval_task1_after_task2
   ```  
**6. Eval on Task 2 w/Task 2 model**
   ```bash
   python src/evaluate.py --config experiments/sharp_curvy_mnist.yaml --eval_name eval_task2_after_task2
   ```  
After every evaluation step, a new line is saved in /results/results.csv.