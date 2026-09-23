启动VLM

```
 conda activate qwen3 
 cd /DATA_HDD/hc/llm/qwen 
 CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 40 --dtype bfloat16 --limit-mm-per-prompt '{"image": 10}' --enable-prefix-caching --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --served-model-name Qwen-VL
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

  python run_multi_agent.py --gpu 0,1 --agent_num 20 --seg-num 2 --episodes 100 --seed 5 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10

  # 需要落盘可视化时加 --debug（默认只写 metrics.json 与 episode.json，不写 nodes/）
  python run_multi_agent.py --gpu 0,1 --agent_num 8 --episodes 10 --seed 5 --debug \
    --backend glee --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10

  # 续跑
  python run_multi_agent.py --gpu 0,1 --agent_num 8 --run-id 20260919_122417 --episodes 100 --seed 5 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10
```

脱离终端继续跑（nohup）

关掉笔记本或退出 Cursor 后仍要继续跑时，用 `nohup` 把进程挂到后台。服务器必须一直开着；关掉这台 GPU 机器则任务会停。

在 SSH 或本机终端里执行（Cursor 内置终端也可以）。启动后关掉终端即可。推理服务不写日志。测试的终端输出写进该次实验目录（与 `summary.json` 同一层），不要落到当前目录的 `nohup.out`。

```bash
# 推理服务：标准输出与错误输出都丢掉
conda activate qwen3
cd /DATA_HDD/hc/llm/qwen
nohup env CUDA_VISIBLE_DEVICES=2,3 vllm serve /DATA_HDD/hc/llm/qwen/Qwen3.6-27B --port 8711 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 16 --dtype bfloat16 --limit-mm-per-prompt '{"image": 10}' --enable-prefix-caching --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --served-model-name Qwen-VL >/dev/null 2>&1 &
VLLM_PID=$!
disown "$VLLM_PID"
echo "vllm pid=$VLLM_PID"

# 并行测试：先定实验目录，终端日志写进该目录
conda activate harnessnav
cd /home/xsuper/hc_workplace/HarnessNav
RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT=/DATA_HDD/hc/harness_outputs/p1/$RUN_ID
mkdir -p "$OUT"
nohup python run_multi_agent.py --gpu 0,1 --agent_num 20 --seg-num 2 --episodes 1000 --seed 1234 \
    --backend gdino_sam --base-url http://127.0.0.1:8711/v1 --model Qwen-VL --max-scans 10 \
    --run-id "$RUN_ID" >"$OUT/run.log" 2>&1 &
echo $! > "$OUT/pid"
disown "$(cat "$OUT/pid")"
echo "pid=$(cat "$OUT/pid")  日志 $OUT/run.log"

# 单进程测试同样写法
# RUN_ID=$(date +%Y%m%d_%H%M%S)
# OUT=/DATA_HDD/hc/harness_outputs/p1/$RUN_ID
# mkdir -p "$OUT"
# nohup python run_HarnessNav.py --episodes 10 --seed 5 --base-url http://127.0.0.1:8711/v1 \
#     --model Qwen-VL --max-scans 10 --run-id "$RUN_ID" >"$OUT/run.log" 2>&1 &
# echo $! > "$OUT/pid"
# disown "$(cat "$OUT/pid")"

# 看测试进度
tail -f "$OUT/run.log"

# 看推理服务是否还在
curl -s http://127.0.0.1:8711/v1/models
pgrep -af 'vllm serve'

# 停掉
kill $(pgrep -f 'vllm serve')
kill "$(cat "$OUT/pid")"
```



## 提交代码到 GitHub（main）

远程仓库：[https://github.com/HCyong2/HarnessNav](https://github.com/HCyong2/HarnessNav)  
约定：只提交自写代码；`thirdparty/`、`model/` 权重、测试产物已由 `.gitignore` 排除。

本机 **SSH（22 / 443）常在握手阶段超时**，已改用 **HTTPS** 远程：

```text
origin  https://github.com/HCyong2/HarnessNav.git
```

```bash
cd /home/xsuper/hc_workplace/HarnessNav

# 1. 看改了什么
git status
git diff

# 2. 暂存要提交的文件（勿把密钥、大权重加进来）
git add -A
git status   # 再确认一遍 staged 列表

# 3. 提交（把引号里的话换成本次说明）
# 若提示 Author identity unknown，在本仓库设一次（不要改 --global，除非你有意为之）：
#   git config user.name "Yong"
#   git config user.email "yong@local"
git commit -m "你的提交说明"

# 4. 推到 main（日常增量用普通 push，不要 --force）
# Cursor 集成终端常注入 GIT_ASKPASS，会连失效的 vscode-git-*.sock，
# 表现为不弹密码框 + Missing or invalid credentials。推送前先清掉：
env -u GIT_ASKPASS -u SSH_ASKPASS \
  -u VSCODE_GIT_ASKPASS_NODE -u VSCODE_GIT_ASKPASS_MAIN \
  -u VSCODE_GIT_ASKPASS_EXTRA_ARGS \
  GIT_TERMINAL_PROMPT=1 git -c credential.helper=store push origin main
# Username: HCyong2
# Password: 粘贴 Personal Access Token（不是登录密码）

# 可选：确认本地与远程一致
git status
git log -1 --oneline
```

说明：

- 第一次备份时用过 `git push --force`，那是为了覆盖远程空壳 README；**以后日常提交只用** `git push origin main`。
- **Token**：打开 [Fine-grained / classic PAT](https://github.com/settings/tokens)，勾选本仓库的 `contents: write`（classic 则勾 `repo`）。推送时密码栏贴 token。上面命令带 `credential.helper=store`，成功一次后会写入 `~/.git-credentials`（明文，注意权限）。
- 若仍出现 `vscode-git-*.sock` / `ECONNREFUSED`：务必用上面的 `env -u GIT_ASKPASS ...` 推送，不要直接 `git push`（Cursor 会抢走交互式密码提示）。
- 若仍想试 SSH：`~/.ssh/config` 可写 `Host github.com` → `Hostname ssh.github.com`、`Port 443`；本机实测 22/443 都会卡在 `SSH2_MSG_KEX_ECDH_REPLY`，优先用 HTTPS。
- 只想提交部分文件时，不要用 `git add -A`，改为 `git add 路径1 路径2`。

