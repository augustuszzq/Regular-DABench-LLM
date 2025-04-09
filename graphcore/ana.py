# #!/usr/bin/env python3
# import sys
# import pva

# def main():
#     # Path to the .pop report file
#     report_path = "/home/kevienzzq/graphcore/examples/nlp/gpt2/pytorch/scaling_benchmark_results/popvision_reports/gpt2-test_8layers_run1_0_4_4_0_20250326_222704/training/profile.pop"
#     report1_path ="/home/kevienzzq/graphcore/examples/nlp/gpt2/pytorch/scaling_benchmark_results/popvision_reports/gpt2-test_4layers_run3_0_4_20250326_202114/profile.pop"
  
#     try:
#         report = pva.openReport(report1_path)
#     except Exception as e:
#         print(f"Error opening report: {e}")
#         sys.exit(1)
    
#     # Access target properties from the report
#     try:
#         target = report.compilation.target
#         numIPUs = target.numIPUs
#         tilesPerIpu = target.tilesPerIpu
#         bytesPerIPU = target.bytesPerIPU
        
#         print("=== Target Information ===")
#         print(f"Number of IPUs: {numIPUs}")
#         print(f"Tiles per IPU: {tilesPerIpu}")
#         print(f"Memory per IPU: {bytesPerIPU} bytes")
#     except Exception as e:
#         print(f"Error accessing target properties: {e}")
#         sys.exit(1)
    
#     # Initialize memory usage statistics for each IPU
#     ipu_memory_usage = [0] * numIPUs
    
#     # Access the compilation tiles from the report
#     try:
#         tiles = report.compilation.tiles
#     except Exception as e:
#         print(f"Error accessing compilation tiles: {e}")
#         sys.exit(1)
    
#     # Iterate over each tile and accumulate memory usage per IPU
#     for idx, tile in enumerate(tiles):
#         # Determine the IPU index by assuming tiles are ordered per IPU
#         ipu_index = idx // tilesPerIpu
#         try:
#             tile_used = tile.memory.used
#         except Exception as e:
#             # Fallback: if 'used' attribute is not available, calculate it using total and free memory.
#             try:
#                 tile_total = tile.memory.total.includingGaps
#                 # tile_free = tile.memory.free
#                 tile_used = tile_total 
#             except Exception as e:
#                 tile_used = 0
#         # Accumulate memory usage for the corresponding IPU
#         if ipu_index < numIPUs:
#             ipu_memory_usage[ipu_index] += tile_used
    
#     # Print memory usage information for each IPU
#     print("\n=== Memory Usage per IPU ===")
#     for i in range(numIPUs):
#         print(f"IPU {i}: Used Memory: {ipu_memory_usage[i]} bytes, Total Capacity: {bytesPerIPU} bytes")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import sys
import os
import pva
import csv
from pathlib import Path

def analyze_pop_file(pop_file_path):
    """
    Analyze a single .pop file and return memory usage information.
    
    Args:
        pop_file_path (str): Path to the .pop file
        
    Returns:
        dict: Dictionary containing analysis results or None if analysis failed
    """
    try:
        report = pva.openReport(pop_file_path)
    except Exception as e:
        print(f"Error opening report {pop_file_path}: {e}")
        return None
    
    # Access target properties from the report
    try:
        target = report.compilation.target
        numIPUs = target.numIPUs
        tilesPerIpu = target.tilesPerIpu
        bytesPerIPU = target.bytesPerIPU
        
        # Initialize memory usage statistics for each IPU
        ipu_memory_usage = [0] * numIPUs
        
        # Access the compilation tiles from the report
        tiles = report.compilation.tiles
        
        # Iterate over each tile and accumulate memory usage per IPU
        for idx, tile in enumerate(tiles):
            # Determine the IPU index by assuming tiles are ordered per IPU
            ipu_index = idx // tilesPerIpu
            try:
                tile_used = tile.memory.used
            except Exception as e:
                # Fallback: if 'used' attribute is not available, calculate it using total memory
                try:
                    tile_total = tile.memory.total.includingGaps
                    tile_used = tile_total 
                except Exception as e:
                    tile_used = 0
            
            # Accumulate memory usage for the corresponding IPU
            if ipu_index < numIPUs:
                ipu_memory_usage[ipu_index] += tile_used
        
        # Create result dictionary
        result = {
            'file_path': pop_file_path,
            'num_ipus': numIPUs,
            'tiles_per_ipu': tilesPerIpu,
            'bytes_per_ipu': bytesPerIPU
        }
        
        # Add memory usage for each IPU
        for i in range(numIPUs):
            result[f'ipu_{i}_memory_used'] = ipu_memory_usage[i]
            result[f'ipu_{i}_memory_percent'] = (ipu_memory_usage[i] / bytesPerIPU) * 100
        
        return result
        
    except Exception as e:
        print(f"Error analyzing {pop_file_path}: {e}")
        return None

def find_pop_files(root_dir):
    """
    Find all .pop files in the given directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search
        
    Returns:
        dict: Dictionary mapping subfolder names to lists of .pop file paths
    """
    pop_files_by_subfolder = {}
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue
        
        # Get the subfolder name (relative to root)
        subfolder = os.path.relpath(dirpath, root_dir)
        
        # Find all .pop files in this directory
        pop_files = [os.path.join(dirpath, f) for f in filenames if f.endswith('.pop')]
        
        if pop_files:
            pop_files_by_subfolder[subfolder] = pop_files
    
    return pop_files_by_subfolder

def write_results_to_csv(results, output_file):
    """
    Write analysis results to a CSV file.
    
    Args:
        results (list): List of result dictionaries
        output_file (str): Path to the output CSV file
    """
    if not results:
        print(f"No results to write to {output_file}")
        return
    
    # Get all possible field names from all results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())
    
    # Sort fieldnames for consistent output
    fieldnames = sorted(fieldnames)
    
    # Write results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_file}")

def main():
    # Root directory containing .pop files
    # root_dir = "/home/kevienzzq/graphcore/examples/nlp/gpt2/pytorch/scaling_benchmark_results_full/popvision_reports"
    root_dir = "/home/kevienzzq/graphcore/examples/nlp/gpt2/pytorch/scaling_benchmark_results_fullstep_fullprecision1/popvision_reports"

    
    
    # Create output directory for CSV files if it doesn't exist
    output_dir = os.path.join(root_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .pop files grouped by subfolder
    pop_files_by_subfolder = find_pop_files(root_dir)
    
    # Create a summary of all results
    all_results = []
    
    # Process each subfolder
    for subfolder, pop_files in pop_files_by_subfolder.items():
        print(f"\nProcessing subfolder: {subfolder}")
        
        subfolder_results = []
        for pop_file in pop_files:
            print(f"Analyzing: {pop_file}")
            result = analyze_pop_file(pop_file)
            if result:
                # Add subfolder information to the result
                result['subfolder'] = subfolder
                subfolder_results.append(result)
                all_results.append(result)
        
        if subfolder_results:
            # Create a sanitized subfolder name for the filename
            safe_subfolder = subfolder.replace('/', '_').replace('\\', '_')
            output_file = os.path.join(output_dir, f"{safe_subfolder}_results.csv")
            
            # Write results for this subfolder
            write_results_to_csv(subfolder_results, output_file)
    
    # Write combined results
    if all_results:
        combined_output_file = os.path.join(output_dir, "all_results.csv")
        write_results_to_csv(all_results, combined_output_file)
        print(f"\nCombined results written to {combined_output_file}")
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()

