#!/bin/bash
# Fix permissions for Docker volumes

echo "Fixing permissions for data directories..."

# Get the UID and GID that the container uses
# The midas user in the container has UID 999 typically
CONTAINER_UID=999
CONTAINER_GID=999

# Alternative: Make directories world-writable (less secure but simpler)
echo "Making data directories writable..."
sudo chown -R $USER:$USER ./data
chmod -R 755 ./data
chmod -R 777 ./data/raw ./data/clean ./data/features

echo "Permissions fixed!"
echo "Directory permissions:"
ls -la ./data/
