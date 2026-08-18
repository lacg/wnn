#!/bin/bash
# Format every Metal shader to the house style (Allman + tabs).
#
# clang-format does not recognise the .metal extension, so each file is fed with
# --assume-filename=x.cpp; Metal is close enough to C++ for the formatter. The
# rules live in the repo-root .clang-format.
#
# --check exits non-zero if any file would change (for a pre-commit hook / CI).
set -u
ROOT="/Users/lacg/wnn"
cd "$ROOT" || exit 1

MODE="${1:-fix}"
# NOT mapfile — macOS ships bash 3.2, where mapfile does not exist and the script
# would die on an unbound array under `set -u`.
FILES=()
while IFS= read -r f
do
	FILES+=("$f")
done < <(find src/wnn/ram/strategies/accelerator -name "*.metal" -not -path "*/target/*" | sort)
if [ ${#FILES[@]} -eq 0 ]
then
	echo "no shaders found"
	exit 1
fi

rc=0
for f in "${FILES[@]}"
do
	if [ "$MODE" = "--check" ]
	then
		if ! clang-format --assume-filename=x.cpp "$f" | diff -q - "$f" >/dev/null
		then
			echo "would reformat: $f"
			rc=1
		fi
	else
		clang-format --assume-filename=x.cpp -i "$f" || rc=1
	fi
done
if [ "$MODE" = "--check" ] && [ $rc -eq 0 ]
then
	echo "all ${#FILES[@]} shaders match the house style"
fi
exit $rc
