#!/usr/bin/env python3
"""
从 Hugging Face Hub 下载文件或目录

支持通过 URL 直接下载：
- 模型仓库
- 数据集仓库
- Space
- 指定文件或目录

使用方法：
    python download_from_hugging_face.py --url "https://huggingface.co/datasets/username/repo-name"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from huggingface_hub import hf_hub_download, snapshot_download, login
except ImportError:
    print("错误: 未安装 huggingface_hub")
    print("请使用以下命令安装: pip install huggingface_hub")
    sys.exit(1)


def parse_huggingface_url(url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    解析 Hugging Face URL，提取仓库信息

    Args:
        url: Hugging Face URL

    Returns:
        (repo_type, repo_id, revision, path_in_repo)
    """
    # 移除末尾的斜杠
    url = url.rstrip('/')

    # 匹配不同类型的 URL 模式
    patterns = [
        # https://huggingface.co/datasets/username/repo-name/tree/main/path
        r'https://huggingface\.co/datasets/([^/]+/[^/]+)(?:/tree/([^/]+)(?:/(.+))?)?',
        # https://huggingface.co/spaces/username/repo-name/tree/main/path
        r'https://huggingface\.co/spaces/([^/]+/[^/]+)(?:/tree/([^/]+)(?:/(.+))?)?',
        # https://huggingface.co/username/repo-name/tree/main/path (model)
        r'https://huggingface\.co/([^/]+/[^/]+)(?:/tree/([^/]+)(?:/(.+))?)?',
        # https://huggingface.co/datasets/username/repo-name/blob/main/file.txt
        r'https://huggingface\.co/datasets/([^/]+/[^/]+)/blob/([^/]+)/(.+)',
        # https://huggingface.co/spaces/username/repo-name/blob/main/file.txt
        r'https://huggingface\.co/spaces/([^/]+/[^/]+)/blob/([^/]+)/(.+)',
        # https://huggingface.co/username/repo-name/blob/main/file.txt (model)
        r'https://huggingface\.co/([^/]+/[^/]+)/blob/([^/]+)/(.+)',
    ]

    repo_type = "model"  # 默认类型
    repo_id = None
    revision = None
    path_in_repo = None

    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            groups = match.groups()
            repo_id = groups[0]
            revision = groups[1] if len(groups) > 1 and groups[1] else None
            path_in_repo = groups[2] if len(groups) > 2 and groups[2] else None

            # 判断仓库类型
            if '/datasets/' in url:
                repo_type = "dataset"
            elif '/spaces/' in url:
                repo_type = "space"
            else:
                repo_type = "model"

            break

    if not repo_id:
        raise ValueError(f"无法解析 URL: {url}\n请确保 URL 格式正确，例如: https://huggingface.co/username/repo-name")

    return repo_type, repo_id, revision, path_in_repo


def download_from_huggingface(
    url: str,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    从 Hugging Face Hub 通过 URL 下载文件或仓库

    Args:
        url: Hugging Face URL
        local_dir: 本地下载目录（可选，默认为仓库名）
        token: Hugging Face API token（可选）
    """
    # 解析 URL
    try:
        repo_type, repo_id, revision, path_in_repo = parse_huggingface_url(url)
        print(f"仓库类型: {repo_type}")
        print(f"仓库 ID: {repo_id}")
        if revision:
            print(f"版本: {revision}")
        if path_in_repo:
            print(f"路径: {path_in_repo}")
        print()
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # Login if token is provided
    if token:
        login(token=token)

    # 设置默认下载目录
    if local_dir is None:
        repo_name = repo_id.split('/')[-1]
        local_dir = f"./{repo_name}"

    # 创建本地目录
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        # 检查是否是单个文件
        is_single_file = path_in_repo and ('blob/' in url or '.' in path_in_repo.split('/')[-1])

        if is_single_file:
            # 下载单个文件
            print(f"正在从 {repo_id} 下载文件: {path_in_repo}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"文件已下载到: {file_path}")
        else:
            # 下载整个仓库或目录
            if path_in_repo:
                print(f"正在从 {repo_id} 下载目录: {path_in_repo}")
            else:
                print(f"正在下载整个仓库: {repo_id}")

            # 如果指定了路径，需要下载整个仓库然后只保留指定路径
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=f"{path_in_repo}/**" if path_in_repo else None,
            )
            print(f"已下载到: {snapshot_path}")

        print(f"\n原始链接: {url}")
        print(f"下载成功!")

    except Exception as e:
        print(f"下载过程中出错: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从 Hugging Face Hub 通过 URL 下载文件或仓库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载整个数据集
  python download_from_hugging_face.py --url "https://huggingface.co/datasets/puyuan1996/orz_0p5b_ppo_150step"

  # 下载数据集的特定版本
  python download_from_hugging_face.py --url "https://huggingface.co/datasets/puyuan1996/orz_0p5b_ppo_150step/tree/main"

  # 下载整个模型
  python download_from_hugging_face.py --url "https://huggingface.co/bert-base-uncased"

  # 下载单个文件
  python download_from_hugging_face.py --url "https://huggingface.co/bert-base-uncased/blob/main/config.json"

  # 指定下载目录
  python download_from_hugging_face.py --url "https://huggingface.co/datasets/my-dataset" --local_dir "./my_data"

  # 私有仓库（需要 token）
  python download_from_hugging_face.py --url "https://huggingface.co/private-repo" --token "hf_xxx"
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Hugging Face 仓库或文件的 URL"
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="本地下载目录（可选，默认为仓库名）"
    )

    parser.add_argument(
        "--token",
        type=str,
        default="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",
        help="Hugging Face API token（可选，用于私有仓库）"
    )

    args = parser.parse_args()

    # 下载
    try:
        download_from_huggingface(
            url=args.url,
            local_dir=args.local_dir,
            token=args.token,
        )
        print("\n✓ 下载完成!")

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
