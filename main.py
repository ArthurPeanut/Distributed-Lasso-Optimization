import argparse
import subprocess

def run_algorithm(method):
    script_path = f"algorithms/{method}.py"  # Adjusted path to match repo structure
    subprocess.run(["python", script_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different distributed optimization algorithms.")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["proximal_gradient", "admm", "subgradient"],
                        help="Choose the optimization method to run.")
    
    args = parser.parse_args()
    run_algorithm(args.method)
