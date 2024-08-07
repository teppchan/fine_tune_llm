#!/usr/bin/env bash

char=80

if [ -z $1 ]; then
    echo specify json file
    exit
fi

cat $1 \
| jq '.entries[].text | select(. != null)' \
| sed -e 's/\!\[\](.\+)//g' \
| sed -e '/^"\\n\\n"$/d' \
> all.txt

cat all.txt \
| awk "{if (length(\$0)>=${char}) {print \$0} }" \
| sed -e 's/\"//g'  \
| sed -e 's/\# //' \
| sed -e 's/^\\n\\n//' \
| sed -e 's/\\n\\n/\\n/g' \
> out.txt

