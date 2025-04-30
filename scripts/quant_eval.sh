#!/bin/bash

DIRPATH=$(dirname $(realpath $0))
cd $DIRPATH/..

# No default values, these must be provided
MODEL=""
OUTPUT_DIR=""

# Parse model and output_dir arguments only
ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --model and --output_dir are required arguments"
  echo "Usage: $0 --model MODEL_PATH --output_dir OUTPUT_PATH [additional args]"
  exit 1
fi

# First run - training
python main.py \
--model $MODEL \
--output_dir $OUTPUT_DIR \
--wbits 4 --abits 4 --lwc --let --multigpu \
$ARGS

# Second run - evaluate perplexity
python main.py --model $MODEL \
--resume $OUTPUT_DIR/rlq_parameters.pth --use_saved_layer 32 \
--output_dir $OUTPUT_DIR \
--wbits 4 --abits 4 --lwc --let --multigpu \
--eval_ppl

# Third run - evaluate on tasks
python main.py --model $MODEL \
--resume $OUTPUT_DIR/rlq_parameters.pth --use_saved_layer 32 \
--output_dir $OUTPUT_DIR \
--wbits 4 --abits 4 --lwc --let --multigpu \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande

