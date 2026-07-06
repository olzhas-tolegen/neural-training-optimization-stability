conda conda env create -f environment.yml
conda activate cs7641-optimization-uncertainty

set -euo pipefail

for nb in \
  01_eda_dataset_original.ipynb \
  part1_randomized_optimization.ipynb \
  part2_adam_family_optimizer.ipynb \
  part3_regularization_study.ipynb \
  part4_integrated_best_combination.ipynb
do
  echo "Executing ${nb}"
  jupyter nbconvert \
    --to notebook \
    --execute "${nb}" \
    --output "executed_${nb}" \
    --ExecutePreprocessor.timeout=0
 done