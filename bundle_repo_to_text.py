import os

# File extensions to include
INCLUDE_EXTS = {
    '.py', '.ipynb', '.sh', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.java', '.rb', '.go', '.rs', '.m', '.lua',
    '.pl', '.php', '.cs', '.swift', '.scala', '.r', '.jl', '.dart', '.md', '.rst', '.txt'
}

OUTPUT_FILE = "repo_bundle.txt"
REPO_ROOT = "."  # Change if not running from repo root
OUTPUT_PATH = os.path.abspath(os.path.join(REPO_ROOT, OUTPUT_FILE))


def should_skip_file(file_path: str) -> bool:
    abs_path = os.path.abspath(file_path)
    if abs_path == OUTPUT_PATH:
        return True

    file_name = os.path.basename(file_path)
    if file_name == "repo_bundle.txt":
        return True

    if file_name.startswith("Vision_core_repo_bundle_") and file_name.endswith(".txt"):
        return True

    return False

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip hidden directories (like .git, .venv, etc.)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in files:
            if fname.startswith('.'):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in INCLUDE_EXTS:
                fpath = os.path.join(root, fname)
                if should_skip_file(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as fin:
                        out.write(f"\n\n{'='*80}\n# filepath: {os.path.relpath(fpath, REPO_ROOT)}\n{'='*80}\n")
                        out.write(fin.read())
                except Exception as e:
                    out.write(f"\n\n# Could not read {fpath}: {e}\n")

print(f"Bundled repo files into {OUTPUT_FILE}")