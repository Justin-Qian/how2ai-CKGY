print("Starting main.py execution...") # Add this line
import argparse
import torch

# Import necessary functions from other modules within the gnn package
print("Importing GNN modules in main.py...") # Add this line
try:
    import config
    from train import train_model
    from evaluate import evaluate_model
except ImportError as e:
    print(f"Error importing GNN modules in main.py: {e}")
    print("Please ensure you are running this script from the 'gnn' directory or have set up the Python path correctly.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the Static GNN model.")
    parser.add_argument(
        'mode',
        choices=['train', 'evaluate'],
        help="Specify whether to 'train' the model or 'evaluate' a pre-trained model."
    )
    parser.add_argument(
        '--force_cpu',
        action='store_true',
        help="Force execution on CPU, even if CUDA is available."
    )

    args = parser.parse_args()

    # Override device setting if force_cpu is used
    if args.force_cpu:
        print("Forcing execution on CPU.")
        config.DEVICE = "cpu"
    else:
        # Check configured device availability
        if config.DEVICE == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA specified in config but not available. Falling back to CPU.")
            config.DEVICE = "cpu"
        elif config.DEVICE == "cuda":
             print(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")


    if args.mode == 'train':
        print("--- Running in Training Mode ---")
        train_model()
    elif args.mode == 'evaluate':
        print("--- Running in Evaluation Mode ---")
        evaluate_model()

if __name__ == "__main__":
    main()
