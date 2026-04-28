# Git hooks

Version-controlled hooks for this repo. Activated via `core.hooksPath`.

## One-time setup (per clone)

```bash
cd /path/to/wnn
git config core.hooksPath scripts/git-hooks
chmod +x scripts/git-hooks/pre-commit
```

After this, the hooks run automatically on `git commit`.

## Hooks

### `pre-commit` — rebuild paper PDF on .tex/.bib changes

If a commit touches `paper/main.tex` or `paper/references.bib`, the hook:

1. Runs `latexmk -pdf` to rebuild `paper/main.pdf`
2. **Blocks the commit** on:
   - pdflatex errors
   - undefined references (`\ref{foo}` to nonexistent label)
   - undefined citations (`\cite{bar}` to nonexistent bib entry)
   - multiply-defined labels
3. **Reports as INFO** (does not block):
   - overfull/underfull boxes (cosmetic line-break issues)
   - font warnings
   - package warnings
4. Auto-stages the rebuilt `paper/main.pdf` so .tex and .pdf ship together

### Bypass for emergencies

```bash
git commit --no-verify   # skip ALL pre-commit hooks for this commit only
```

## Dependencies

- **`latexmk`** must be on PATH. Install on macOS:
  ```bash
  sudo tlmgr install latexmk
  ```
  (TeX Live's package manager; Homebrew doesn't carry it.)
