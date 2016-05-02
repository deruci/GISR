#!/bin/bash

echo "Download SR91 training dataset for super-resolution to /data/SR91"
wget "https://www.dropbox.com/s/xnwn4o7m6fwtoa3/SR91_PNG.zip"
mkdir data
mkdir data/training
mkdir data/training/SR91
unzip SR91_PNG.zip -d data/training/SR91
rm -f SR91_PNG.zip
