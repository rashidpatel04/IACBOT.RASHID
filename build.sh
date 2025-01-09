#!/usr/bin/env bash

# Update package list
apt-get update

# Install espeak
apt-get install -y espeak

# Clean up to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*
