#!/usr/bin/env sh

set -e

# From https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE
# However, Google Drive always gives an annoying confirmation page
# Bypass it with my own file server temporarily.
wget -c "https://file.szp.io/f/f6504d6f69/?dl=1" -O domain_adaptation_images.tar.gz
echo "7aa25a86f3880b6e79283227816f9d4d8506d636  domain_adaptation_images.tar.gz" | sha1sum -c -
tar -xf domain_adaptation_images.tar.gz

for dataset in amazon dslr webcam; do
    subfolders=($(ls "$dataset/images"))
    rm -f "$dataset.txt"
    for ((i = 0; i < ${#subfolders[@]}; ++i)); do
        for file in $(ls "$dataset/images/${subfolders[$i]}"); do
            echo "$dataset/images/${subfolders[$i]}/$file $i" >> "$dataset.txt"
        done
    done
done
