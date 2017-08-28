#!/usr/bin/env sh

set -e

# From https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg
# However, Google Drive always gives an annoying confirmation page
# Bypass it with my own file server temporarily.
wget -c "https://file.szp.io/f/492d03c9c9/?dl=1" -O office_home.tar.gz
echo "db8d358e8c74749baf5230a9d01a4d8d7aad9adb  office_home.tar.gz" | sha1sum -c -
tar -xf office_home.tar.gz

for dataset in Art Clipart Product "Real World"; do
    subfolders=($(ls "$dataset"))
    rm -f "$dataset.txt"
    for ((i = 0; i < ${#subfolders[@]}; ++i)); do
        for file in $(ls "$dataset/${subfolders[$i]}"); do
            echo "$dataset/${subfolders[$i]}/$file $i" >> "$dataset.txt"
        done
    done
done
