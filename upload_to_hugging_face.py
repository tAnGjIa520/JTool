#!/usr/bin/env python3
"""
上传文件或目录到 Hugging Face Hub

支持上传：
- 单个文件
- 整个目录
- 模型检查点
- 数据集

使用方法：
    python upload_to_hugging_face.py --repo_id "username/repo-name" --local_path "./model" --repo_type "model"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


def upload_to_huggingface(
    repo_id: str,
    local_path: str,
    token: Optional[str] = None,
    repo_type: str = "model",
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    create_if_not_exists: bool = True,
):
    """
    Upload files or folders to Hugging Face Hub.

    Args:
        repo_id: Repository ID in format "username/repo-name"
        local_path: Local file or folder path to upload
        token: Hugging Face API token (optional if already logged in)
        repo_type: Type of repo - "model", "dataset", or "space"
        path_in_repo: Path in the repo where to upload (optional)
        commit_message: Custom commit message (optional)
        private: Whether to create a private repository
        create_if_not_exists: Create repo if it doesn't exist
    """
    # Login if token is provided
    if token:
        login(token=token)

    # Initialize the API
    api = HfApi()

    # Check if local path exists
    local_path_obj = Path(local_path)
    if not local_path_obj.exists():
        raise FileNotFoundError(f"Local path not found: {local_path}")

    # Create repository if needed
    if create_if_not_exists:
        try:
            print(f"Creating repository: {repo_id} (type: {repo_type})")
            create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True
            )
            print(f"Repository created/verified: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repository: {e}")

    # Set default commit message
    if commit_message is None:
        if local_path_obj.is_file():
            commit_message = f"Upload {local_path_obj.name}"
        else:
            commit_message = f"Upload {local_path_obj.name} folder"

    # Upload file or folder
    try:
        if local_path_obj.is_file():
            print(f"Uploading file: {local_path}")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo or local_path_obj.name,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            print(f"File uploaded successfully!")
        else:
            print(f"Uploading folder: {local_path}")
            api.upload_folder(
                folder_path=local_path,
                path_in_repo=path_in_repo or "",
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            print(f"Folder uploaded successfully!")

        # Print success message with link
        if repo_type == "model":
            url = f"https://huggingface.co/{repo_id}"
        elif repo_type == "dataset":
            url = f"https://huggingface.co/datasets/{repo_id}"
        elif repo_type == "space":
            url = f"https://huggingface.co/spaces/{repo_id}"
        else:
            url = f"https://huggingface.co/{repo_id}"

        print(f"\nView your upload at: {url}")

    except Exception as e:
        print(f"Error during upload: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="上传文件或目录到 Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 上传模型文件夹
  python upload_to_hugging_face.py --repo_id "my-model" --local_path "./model" --repo_type "model"

  # 上传数据集
  python upload_to_hugging_face.py --repo_id "my-dataset" --local_path "./data" --repo_type "dataset"

  # 上传单个文件到仓库中的指定路径
  python upload_to_hugging_face.py --repo_id "my-repo" --local_path "./config.json" --path_in_repo "configs/config.json"

  # 创建私有仓库
  python upload_to_hugging_face.py --repo_id "my-private-model" --local_path "./model" --private
        """
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='仓库名称（会自动添加用户名 tangjia0424）'
    )

    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="要上传的本地文件或文件夹路径"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token（如果已通过 'huggingface-cli login' 登录则可选）"
    )

    parser.add_argument(
        "--repo_type",
        type=str,
        choices=["model", "dataset", "space"],
        default="model",
        help="仓库类型（默认: model）"
    )

    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="文件在仓库中的路径（可选）"
    )

    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="自定义提交信息（可选）"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="创建私有仓库"
    )

    parser.add_argument(
        "--no-create",
        action="store_true",
        help="如果仓库不存在，不自动创建"
    )

    args = parser.parse_args()

    # 自动添加用户名前缀
    if "/" not in args.repo_id:
        repo_id = f"tangjia0424/{args.repo_id}"
    else:
        repo_id = args.repo_id

    # Upload to Hugging Face
    try:
        upload_to_huggingface(
            repo_id=repo_id,
            local_path=args.local_path,
            token=args.token,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            private=args.private,
            create_if_not_exists=not args.no_create,
        )
        print("\nUpload completed successfully!")

    except Exception as e:
        print(f"\nUpload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
