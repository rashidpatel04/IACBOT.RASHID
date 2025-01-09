#!/usr/bin/env bash

# Update package list
sudo apt-get update

# Install espeak
sudo apt-get install -y espeak

# Clean up to reduce image size
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*