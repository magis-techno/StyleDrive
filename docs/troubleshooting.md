# Troubleshooting Guide

This document provides solutions for common issues encountered when running StyleDrive.

## Dataset Caching Issues

### Problem: Ray "Resource temporarily unavailable" Error

**Symptoms:**
```
RuntimeError('Resource temporarily unavailable')
Error in call to target 'navsim.planning.utils.multithreading.worker_ray_no_torch.RayDistributedNoTorch'
```

**Root Causes:**
1. Ray trying to allocate too many CPU cores (default: all available cores)
2. System resource limits (ulimit restrictions)
3. Network/port conflicts in Ray cluster initialization
4. Existing Ray processes consuming resources

**Solutions (in order of preference):**

#### Solution 1: Use Sequential Worker ⭐ (Most Stable)
```bash
bash scripts/caching/caching_training_sequential.sh
```
- Single-threaded processing
- No Ray dependencies
- Works on all systems
- Slower but guaranteed to work

#### Solution 2: Use Thread Pool Worker ⭐⭐ (Recommended)
```bash
bash scripts/caching/caching_training_threadpool.sh
```
- Multi-threaded without Ray
- Good balance of speed and stability
- Uses standard Python threading

#### Solution 3: Reduce Ray Resources
The original script has been modified to use fewer resources:
```bash
bash scripts/caching/caching_training_tran_diff.sh
```
- Now uses `ray_low_resource` worker (8 CPUs instead of all)
- May still fail on some systems

### Problem: Environment Variables Not Set

**Symptoms:**
```
provider=hydra.searchpath in main, path=/path/to/StyleDrive/navsim/navsim/planning/script/config/common is not available
```

**Solution:**
```bash
# Set environment variables
source ./env_vars.sh

# Verify variables are set
echo $NAVSIM_DEVKIT_ROOT
echo $NAVSIM_EXP_ROOT
```

### Problem: Hydra Configuration Path Error

**Symptoms:**
```
path=/path/StyleDrive/navsim/navsim/planning/script/config/common is not available
```

**Root Cause:** Duplicate `navsim` in path configuration

**Solution:** 
This has been fixed in `navsim/planning/script/config/training/default_training.yaml`. The path now correctly points to:
```
${oc.env:NAVSIM_DEVKIT_ROOT}/planning/script/config/common
```

## Performance Comparison

| Worker Type | Stability | Speed | CPU Usage | Memory | Use Case |
|-------------|-----------|-------|-----------|---------|----------|
| Sequential | 🟢 Excellent | 🔴 Slow | 1 core | Low | First-time setup, debugging |
| Thread Pool | 🟡 Good | 🟡 Medium | Multi-core | Medium | Production environments |
| Ray (fixed) | 🔴 Variable | 🟢 Fast | 8 cores | High | High-performance systems |
| Ray (default) | 🔴 Poor | 🟢 Fastest | All cores | Highest | Ideal conditions only |

## Quick Diagnosis Commands

```bash
# Check environment variables
echo "NAVSIM_DEVKIT_ROOT: $NAVSIM_DEVKIT_ROOT"
echo "NAVSIM_EXP_ROOT: $NAVSIM_EXP_ROOT"

# Check if directories exist
ls -la $NAVSIM_DEVKIT_ROOT
ls -la $NAVSIM_EXP_ROOT

# Check system resources
echo "CPU cores: $(nproc)"
echo "Available memory: $(free -h | awk '/^Mem:/{print $7}')"

# Check for running Ray processes
ps aux | grep ray | grep -v grep
```

## Best Practices

1. **Always use `sequential` worker for first-time setup** - ensures everything works
2. **Use `thread_pool` worker for production** - good balance of speed and stability  
3. **Only use Ray workers on known-compatible systems** - test thoroughly first
4. **Set environment variables before running any scripts** - prevents path issues
5. **Create cache directory manually if needed**: `mkdir -p $NAVSIM_EXP_ROOT/training_cache`

## Getting Help

If you encounter other issues:
1. Check this troubleshooting guide first
2. Set `export HYDRA_FULL_ERROR=1` for detailed error messages
3. Try the most stable worker type (sequential) to isolate the problem
4. Check system logs for resource-related errors 