# Git History Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite repository history to remove tracked large/generated files so the repository can be pushed to GitHub normally.

**Architecture:** Use an isolated project-local worktree for safety, update ignore rules before creating it, then run a single history rewrite that removes known heavy paths from all commits. After the rewrite, aggressively prune unreachable objects and verify repository size and object inventory before describing the future push workflow.

**Tech Stack:** Git, git-filter-repo, project-local `.gitignore`

---

## Chunk 1: Isolated Workspace Setup

### Task 1: Prepare local ignore rules for worktrees and generated artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add ignore rules for `.worktrees/` and generated binary/output paths**

Add entries for:
- `.worktrees/`
- `dataset/`
- `models/`
- `vis_centerline/`
- `training/__pycache__/`
- `__pycache__/`
- `*.pyc`

- [ ] **Step 2: Verify the ignore file reflects the intended exclusions**

Run: `sed -n '1,120p' .gitignore`
Expected: the new ignore entries are present

- [ ] **Step 3: Create the project-local worktree directory and confirm it is ignored**

Run: `mkdir -p .worktrees && git check-ignore .worktrees`
Expected: `.worktrees` is reported as ignored

### Task 2: Create an isolated branch/worktree for the cleanup

**Files:**
- No file changes required

- [ ] **Step 1: Create the worktree**

Run: `git worktree add .worktrees/git-history-cleanup -b git-history-cleanup`
Expected: a new worktree is created on branch `git-history-cleanup`

- [ ] **Step 2: Confirm the worktree starts from the current branch state**

Run: `git -C .worktrees/git-history-cleanup status -sb`
Expected: branch is `git-history-cleanup`

## Chunk 2: History Rewrite

### Task 3: Rewrite history to drop large/generated paths

**Files:**
- No working tree file edits; repository history is rewritten

- [ ] **Step 1: Back up the current branch ref**

Run: `git branch backup/pre-cleanup-2026-03-14`
Expected: backup branch created successfully

- [ ] **Step 2: Rewrite history**

Run: `git filter-repo --force --path dataset --path models --path vis_centerline --path training/__pycache__ --invert-paths`
Expected: history is rewritten and unwanted paths are removed from all commits

- [ ] **Step 3: Inspect the repository status after the rewrite**

Run: `git status -sb`
Expected: repository is on the cleanup branch with rewritten history and no unexpected tracked large paths

## Chunk 3: Repository Pruning and Verification

### Task 4: Remove unreachable objects and verify size reduction

**Files:**
- No file changes required

- [ ] **Step 1: Expire reflogs and run garbage collection**

Run: `git reflog expire --expire=now --all && git gc --prune=now --aggressive`
Expected: old unreachable objects are pruned

- [ ] **Step 2: Measure repository size**

Run: `git count-objects -vH`
Expected: packed size is substantially smaller than before

- [ ] **Step 3: List the largest remaining objects**

Run: `git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | rg '^blob ' | sort -k3 -n | tail -n 20`
Expected: no remaining giant model checkpoints or dataset files

### Task 5: Document future push workflow

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Ensure `.gitignore` is committed with the cleanup branch history**

Run: `git diff -- .gitignore`
Expected: ignore changes are present as intended

- [ ] **Step 2: Prepare the future workflow guidance for the human partner**

Include:
- keep datasets and models outside normal git history
- use `.gitignore` before `git add`
- use `git status` and `git diff --cached` before each commit
- prefer Git LFS or external storage for large binaries
- use `git push --force-with-lease` after history rewrites only when expected
