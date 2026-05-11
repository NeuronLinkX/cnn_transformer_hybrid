#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/home/jiwoo/Desktop/workspace/cnn_transformer_cifar10}"
cd "$REPO_DIR"

echo "[1] 현재 저장소 위치"
pwd

echo "[2] SSH remote를 HTTPS remote로 변경"
git remote set-url origin https://github.com/NeuronLinkX/cnn_transformer_hybrid.git
git remote -v

echo "[3] pull 기본 방식을 rebase로 설정"
git config --global pull.rebase true
git config --global rebase.autoStash true

echo "[4] 원격 상태 갱신"
git fetch origin main

echo "[5] 로컬 변경사항 임시 저장"
git stash push -u -m "auto-stash-before-rebase" || true

echo "[6] 원격 main 기준으로 rebase"
git rebase origin/main

echo "[7] stash 복원"
git stash pop || true

echo "[8] 상태 확인"
git status

echo
echo "이제 아래 명령으로 push:"
echo "git push origin main"
