# Git History Cleanup Design

**Context**

The repository cannot be pushed reliably to GitHub because git history contains large generated artifacts and datasets, including model checkpoints larger than GitHub's 100 MB object limit. Current `.gitignore` rules only prevent future additions and do not remove already-tracked objects from history.

**Goal**

Rewrite repository history so the GitHub-facing repository keeps code and documentation only, while excluding generated artifacts, datasets, cached files, and visualization outputs that do not belong in normal source control history.

**Approach**

We will create an isolated project-local git worktree, update ignore rules to cover local worktrees and generated assets, then use a history-rewrite tool to remove heavy paths from all commits. After rewriting, we will expire reflogs, garbage-collect the repository, verify the largest remaining objects, and document the safe push workflow for future use.

**Scope**

- Remove these paths from git history:
  - `dataset/`
  - `models/`
  - `vis_centerline/`
  - `training/__pycache__/`
- Strengthen ignore rules so these paths do not get reintroduced
- Preserve source code, docs, and lightweight project files
- Verify final repository size and largest tracked objects
- Provide a future GitHub push checklist for large-file-safe workflows

**Non-Goals**

- Preserving old large binary artifacts inside the main GitHub repository
- Migrating model files to Git LFS in this cleanup pass
- Changing application behavior or training code

**Risks**

- History rewrite changes commit SHAs and requires force-pushing
- Any collaborators with old clones will need to re-sync carefully
- If local-only artifacts are not ignored afterward, the problem can return

**Validation**

- `git count-objects -vH` shows a much smaller packed repository size
- Largest remaining git objects are comfortably below GitHub's limits
- `git status -sb` is clean except for intended local changes
- A documented future push workflow is available to prevent recurrence
