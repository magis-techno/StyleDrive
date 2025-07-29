# Dataset Caching Scripts

This directory contains scripts for caching StyleDrive training datasets.

## Available Scripts

### 1. `caching_training_sequential.sh` ⭐ (Most Stable)
- **Use case**: First-time setup, systems with Ray issues
- **Performance**: Single-threaded, slower but guaranteed to work
- **Requirements**: Basic Python environment
- **Command**: `bash caching_training_sequential.sh`

### 2. `caching_training_threadpool.sh` ⭐⭐ (Recommended)
- **Use case**: Production environments, regular use
- **Performance**: Multi-threaded, good balance of speed and stability
- **Requirements**: Standard threading support
- **Command**: `bash caching_training_threadpool.sh`

### 3. `caching_training_tran_diff.sh` (Ray Worker)
- **Use case**: High-performance systems with Ray compatibility
- **Performance**: Parallel processing, fastest when working
- **Requirements**: Properly configured Ray environment
- **Command**: `bash caching_training_tran_diff.sh`

### 4. `run_metric_caching.sh` (Original)
- **Use case**: Metric caching (different from dataset caching)
- **Performance**: Depends on configuration
- **Command**: `bash run_metric_caching.sh`

## Quick Start

1. **Set environment variables**:
   ```bash
   source ../../env_vars.sh
   ```

2. **Choose appropriate script**:
   ```bash
   # For most users (recommended)
   bash caching_training_threadpool.sh
   
   # If you encounter issues
   bash caching_training_sequential.sh
   ```

## Troubleshooting

If you encounter errors:
1. Check environment variables are set: `echo $NAVSIM_DEVKIT_ROOT`
2. Try the sequential worker: `bash caching_training_sequential.sh`
3. See detailed troubleshooting guide: `../../docs/troubleshooting.md`

## Performance Comparison

| Script | Speed | Stability | CPU Usage |
|--------|-------|-----------|-----------|
| sequential | 🔴 Slow | 🟢 Excellent | 1 core |
| threadpool | 🟡 Medium | 🟡 Good | Multi-core |
| tran_diff (Ray) | 🟢 Fast | 🔴 Variable | 8+ cores |

## Notes

- All scripts create the cache in `$NAVSIM_EXP_ROOT/training_cache`
- Cache files are large; ensure sufficient disk space
- Caching is a one-time process; subsequent training uses cached data
- Sequential worker is always recommended for troubleshooting 