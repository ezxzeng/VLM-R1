#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get the absolute path to the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../../../.." && pwd )"
MULTIMODAL_DIR="$PROJECT_DIR/VLM-R1/src/open-r1-multimodal"

echo "Project directory: $PROJECT_DIR"
echo "Using code from: $MULTIMODAL_DIR"

# Change to the multimodal directory
cd "$MULTIMODAL_DIR" || { echo "Failed to change directory to $MULTIMODAL_DIR"; exit 1; }

# Set environment variables
export DEBUG_MODE="false"
export TORCH_DYNAMO=0
export CUDA_VISIBLE_DEVICES="0"  # Adjust based on available GPUs

# Define common parameters
BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
VAL_FILE="/home/ezzeng/stat946/stat946_final_proj/data/val_qas.jsonl"
IMAGE_FOLDER="/home/ezzeng/stat946/stat946_final_proj/data"
SEGMENTATION_FOLDER="/home/ezzeng/stat946/stat946_final_proj/data/coco/segmentation_masks"
OUTPUT_DIR="$PROJECT_DIR/evaluation_results"
METRICS="accuracy,format,counting_format_reward,count_consistency,counting_accuracy,point_accuracy"

# Enable more verbose output
set -x

# Verify paths exist
echo "Checking if validation file exists: $VAL_FILE"
if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation file not found: $VAL_FILE"
    # Try to find the file
    echo "Searching for val_qas.jsonl..."
    FOUND_FILE=$(find /home/ezzeng/stat946/stat946_final_proj/data -name "val_qas.jsonl" 2>/dev/null | head -n 1)
    if [ -n "$FOUND_FILE" ]; then
        echo "Found validation file at: $FOUND_FILE"
        VAL_FILE="$FOUND_FILE"
    else
        # Try a broader search
        echo "Broadening search for val_qas.jsonl..."
        FOUND_FILE=$(find /home -name "val_qas.jsonl" 2>/dev/null | head -n 1)
        if [ -n "$FOUND_FILE" ]; then
            echo "Found validation file at: $FOUND_FILE"
            VAL_FILE="$FOUND_FILE"
        else
            echo "Could not find val_qas.jsonl. Please check the path."
            exit 1
        fi
    fi
fi

echo "Checking if image folder exists: $IMAGE_FOLDER"
if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Error: Image folder not found: $IMAGE_FOLDER"
    echo "Searching for the correct data folder..."
    PARENT_DIR=$(dirname "$VAL_FILE")
    if [ -d "$PARENT_DIR" ]; then
        echo "Using parent directory of validation file as image folder: $PARENT_DIR"
        IMAGE_FOLDER="$PARENT_DIR"
    else
        exit 1
    fi
fi

# Check if segmentation folder exists, if not, set to null
if [ ! -d "$SEGMENTATION_FOLDER" ]; then
    echo "Warning: Segmentation folder not found: $SEGMENTATION_FOLDER"
    # Try to determine the correct segmentation folder location
    POTENTIAL_SEG_FOLDER="$IMAGE_FOLDER/coco/segmentation_masks"
    if [ -d "$POTENTIAL_SEG_FOLDER" ]; then
        echo "Found potential segmentation folder at: $POTENTIAL_SEG_FOLDER"
        SEGMENTATION_FOLDER="$POTENTIAL_SEG_FOLDER"
    else
        echo "Setting segmentation_folder to null"
        SEGMENTATION_FOLDER="null"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to evaluate a model checkpoint
evaluate_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename "$MODEL_PATH")
    local OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_eval_results.json"
    
    echo "Evaluating $MODEL_NAME..."
    
    # Build segmentation folder argument
    local SEG_ARG=""
    if [ "$SEGMENTATION_FOLDER" != "null" ]; then
        SEG_ARG="--segmentation_folder $SEGMENTATION_FOLDER"
    fi
    
    # Run the evaluation script with absolute paths
    python "$MULTIMODAL_DIR/src/open_r1/grpo_jsonl_eval.py" \
        --checkpoint_path "$MODEL_PATH" \
        --data_file_path "$VAL_FILE" \
        --image_folder "$IMAGE_FOLDER" \
        --base_model_path "$BASE_MODEL" \
        --output_file "$OUTPUT_FILE" \
        --question_task_template count_many_examples_question_last \
        --batch_size 1 \
        --metrics "$METRICS" \
        $SEG_ARG \
        --max_new_tokens 256 \
        --top_p 0.7 \
        --temperature 0.7 \
        --trust_remote_code True \
        --use_fast_tokenizer True
    
    # Check if the evaluation was successful
    local STATUS=$?
    if [ $STATUS -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Evaluation of $MODEL_NAME completed successfully. Results saved to $OUTPUT_FILE"
        return 0
    else
        echo "Error: Evaluation of $MODEL_NAME failed with status code $STATUS"
        return 1
    fi
}

# Path to output directory containing model checkpoints
OUTPUT_BASE_PATH="/pub5/ezzeng/stat946_final_project/output"

# List of models to evaluate
# You can uncomment the models you want to evaluate
MODELS=(
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR"
    "Qwen2.5-VL-3B-GRPO-TALLY-lora-segR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-pointsR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-segR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-pointAccR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-segR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntConsR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntAccR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-segR-pointAccR-cntConsR-cntAccR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-segR-cntConsR-cntAccR"
    # "Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR-pointsR-pointAccR-cntConsR-cntAccR"
)

# Check if model output directory exists
if [ ! -d "$OUTPUT_BASE_PATH" ]; then
    echo "Error: Model output directory not found: $OUTPUT_BASE_PATH"
    # Try to find the output directory
    echo "Searching for output directory..."
    POTENTIAL_DIRS=(
        "/pub0/smnair/stat946_final_project/output"
        "/pub5/ezzeng/stat946_final_project/output"
        "/home/ezzeng/stat946/stat946_final_proj/output"
    )
    for DIR in "${POTENTIAL_DIRS[@]}"; do
        if [ -d "$DIR" ]; then
            echo "Found potential output directory at: $DIR"
            OUTPUT_BASE_PATH="$DIR"
            break
        fi
    done
fi

# Check if output directories exist
for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$OUTPUT_BASE_PATH/$MODEL"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model directory not found: $MODEL_PATH"
    else
        echo "Found model directory: $MODEL_PATH"
        # Check for checkpoints and verify they're accessible
        if [ -d "$MODEL_PATH/checkpoint-50" ]; then
            echo "Found checkpoint at $MODEL_PATH/checkpoint-50"
        fi
    fi
done

# Disable verbose output before running evaluations
set +x

# Track overall success/failure
EVALUATION_SUCCESS=true

# Run evaluation for each model
for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$OUTPUT_BASE_PATH/$MODEL"
    if [ -d "$MODEL_PATH" ]; then
        if ! evaluate_model "$MODEL_PATH"; then
            EVALUATION_SUCCESS=false
        fi
    else
        echo "Model directory not found: $MODEL_PATH"
        EVALUATION_SUCCESS=false
    fi
done

# Optionally, you can also evaluate specific checkpoints
# For example:
# evaluate_model "$OUTPUT_BASE_PATH/Qwen2.5-VL-3B-GRPO-TALLY-lora-cntFormR/checkpoint-100"

# Only report success if all evaluations completed successfully
if [ "$EVALUATION_SUCCESS" = true ]; then
    echo "All evaluations completed successfully. Results saved in $OUTPUT_DIR"

    # Summarize results
    echo "Summary of evaluation results:"
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(basename "$MODEL")
        RESULT_FILE="$OUTPUT_DIR/${MODEL_NAME}_eval_results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "-------------------------------------------"
            echo "Results for $MODEL_NAME:"
            cat "$RESULT_FILE" | grep -o '"metrics":{[^}]*}' | sed 's/"metrics"://' | python -m json.tool
        fi
    done
    exit 0
else
    echo "Error: Some evaluations failed. Check the logs for details."
    exit 1
fi 