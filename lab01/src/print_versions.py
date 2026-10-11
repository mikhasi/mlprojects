import sys
import importlib.metadata
import os

packages = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm"
]

def main():
    output_lines = []
    
    py_version_str = f"Python Version: {sys.version}\n"
    output_lines.append(py_version_str)
    print(py_version_str, end="")
    
    header = "Package Versions:\n" + "-" * 30
    output_lines.append(header + "\n")
    print(header)
    
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            line = f"{pkg:20} : {version}"
        except importlib.metadata.PackageNotFoundError:
            line = f"{pkg:20} : Not installed"
        
        output_lines.append(line + "\n")
        print(line)
        
    os.makedirs("results", exist_ok=True)
    os.makedirs("lab01/results", exist_ok=True)
    
    output_path = "results/versions.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
        
    print(f"\nVersions successfully saved to {output_path}")

if __name__ == "__main__":
    main()