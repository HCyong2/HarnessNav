启动VLM

```
 conda activate qwen3 
 cd /DATA_HDD/hc/llm/qwen 
 CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 16 --dtype bfloat16 --limit-mm-per-prompt '{"image": 10}' --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --served-model-name Qwen-VL
```

运行测试

```
conda activate harnessnav
python run_HarnessNav.py --episodes 10 --seed 5 --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10
```

运行并行测试

```
  conda activate harnessnav
  python run_multi_agent.py --gpu 0,1 --agent_num 8 --episodes 10 --seed 5 \
    --backend glee --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10

  # 默认 --gpu 0,1、--seg-num 10（两卡共 10 份分割）。少开副本例如：
  python run_multi_agent.py --gpu 0,1 --agent_num 10 --seg-num 2 --episodes 20 --seed 5 \
    --backend glee --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10

  python run_multi_agent.py --gpu 0,1 --agent_num 10 --seg-num 2 --episodes 20 --seed 5 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10

  # 续跑
  python run_multi_agent.py --gpu 0,1 --agent_num 8 --run-id 20260919_122417 --episodes 100 --seed 5 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10
```

脱离终端继续跑（tmux）

关掉笔记本或退出 Cursor 后仍要继续跑时，把上面的命令放进 `tmux`。服务器必须一直开着；关掉这台 GPU 机器则任务会停。

在 SSH 或本机终端里执行（Cursor 内置终端也可以，但启动后必须离开会话再关 Cursor）。`Ctrl+b` 再按 `d` 离开会话，进程继续跑。

```bash
# 推理服务
tmux new -s vllm
conda activate qwen3
cd /DATA_HDD/hc/llm/qwen
CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 16 --dtype bfloat16 --limit-mm-per-prompt '{"image": 10}' --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --served-model-name Qwen-VL
# Ctrl+b，再 d

# 并行测试（另开一个会话；日志写到硬盘，回来可查）
tmux new -s test
conda activate harnessnav
cd /home/xsuper/hc_workplace/HarnessNav
python run_multi_agent.py --gpu 0,1 --agent_num 10 --seg-num 2 --episodes 20 --seed 5 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10
# Ctrl+b，再 d

tmux ls                 # 列出会话
tmux attach -t vllm     # 看推理服务
tmux attach -t test     # 看测试
```

