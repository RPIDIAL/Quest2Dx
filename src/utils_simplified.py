# Only contains the necessary functions for the released code to simplify environment setup
import subprocess
import colorsys

### General Utils ###
def darken_color(color, amount=0.5):
    c = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def select_gpu(num_gpus=1,verbose=False):
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], capture_output=True, text=True)

    # Parse the output to get GPU index and memory usage
    gpu_info = result.stdout.strip().split('\n')
    gpu_info = [info.split(',') for info in gpu_info]
    gpu_info = [(info[0], int(info[1].split()[0])) for info in gpu_info]

    # Sort the GPU info based on memory usage
    sorted_gpu_info = sorted(gpu_info, key=lambda x: x[1])

    if verbose:
        # Print the GPU info with least memory usage
        for gpu in sorted_gpu_info:
            print(f"GPU {gpu[0]}: Memory Usage {gpu[1]} MB")
    
    # Select the first num_gpus GPUs with least memory usage
    selected_gpus = [gpu[0] for gpu in sorted_gpu_info[:num_gpus]]
    return selected_gpus