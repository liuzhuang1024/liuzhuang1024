# 命令行代理
export https_proxy=http://127.0.0.1:7899 http_proxy=http://127.0.0.1:7899 all_proxy=socks5://127.0.0.1:7899

# CUDA环境配置
export CUDA_ROOT="/data/liuzhuang/cuda"
export CUDA_INC_DIR="/data/liuzhuang/cuda/include"
export PATH="$PATH:/data/liuzhuang/cuda/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/liuzhuang/cuda/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/liuzhuang/tensorrt/lib"

# 导入torch
export TORCH_PATH=$(dirname $(python -c "import torch; print(torch.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH/lib/

# 导入临时文件夹
export TMPDIR=new_path