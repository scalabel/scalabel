#!/usr/bin/env bash

JSSTYLE=$(which eslint)
JSOPTIONS=""
if [ $? -ne 0 ]; then
    echo "[!] eslint not installed. run script/setup.sh first." >&2
    exit 1
fi

FILES=`git diff --name-only --diff-filter=ACMR | grep -E "\.(js)$"`
for FILE in $FILES; do
    $JSSTYLE $JSOPTIONS $FILE >&2
    if [ $? -ne 0 ]; then
        echo "[!] $FILE does not respect google code style." >&2
        RETURN=1
    fi
done

go fmt "../app/golang/..."

exit $RETURN
