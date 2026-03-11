import os
import subprocess

def main():
    # 设置环境变量
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "https://api.66688777.xyz"
    env["ANTHROPIC_API_KEY"] = "REDACTED_API_KEY"

    print("启动 Claude Code... (已自动注入环境变量)")
    
    # 启动 claude
    try:
        subprocess.run(["claude"], env=env)
    except FileNotFoundError:
        print("错误: 找不到 claude 命令。请确保已经全局安装了 @anthropic-ai/claude-code。")

if __name__ == "__main__":
    main()
