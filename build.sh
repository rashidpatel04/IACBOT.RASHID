#!/usr/bin/env bash

# Update package list
apt-get update

# Install both espeak and espeak-ng
apt-get install -y espeak espeak-ng

# Clean up to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*
