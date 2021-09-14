#!/bin/bash

echo 'We will add commit and push all changes in chunks of 1000 files'
set -e

git pull &&

while [[ $(git status -s) ]]
do
    git status -s | grep 'M\|??\|A\|U\|D' | cut -c 3- > files_to_commit.txt
    while [[ $(cat files_to_commit.txt) ]]
    do
        head -n 1000 files_to_commit.txt | xargs git add
        git commit -m 'added runs' 
        git push
        tail -n 1001 files_to_commit.txt > files_to_commit.txt
    done
done