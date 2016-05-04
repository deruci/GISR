#!/bin/bash

echo "Download SR91 training dataset for super-resolution to /data/training/SR91"
wget "https://www.dropbox.com/s/xnwn4o7m6fwtoa3/SR91_PNG.zip"
mkdir data
mkdir data/training
mkdir data/training/SR91
unzip SR91_PNG.zip -d data/training/SR91
rm -f SR91_PNG.zip

echo "Download Set5 validation(test) dataset for super-resolution to /data/test/Set5"
wget "https://www.dropbox.com/s/ters4urci06rl9q/Set5.zip"
mkdir data/test
mkdir data/test/Set5
unzip Set5.zip -d data/test/Set5
rm -f Set5.zip
