#!/bin/bash

echo 'We will add commit and push all changes in chunks of 1000 files'

while [[ $(git status -s) ]]
do
    git status -s | grep 'M\|??\|A\|U' | cut -c 3- | head -n 500 | xargs git add
    git commit -m 'added runs' 
    git push
done