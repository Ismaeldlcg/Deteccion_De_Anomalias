#!/bin/bash

set -e

echo "================================"
echo "GitHub Actions Setup"
echo "================================"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status > /dev/null 2>&1; then
    echo "Authenticating with GitHub..."
    gh auth login
fi

echo ""
echo "Available secret configuration options:"
echo ""
echo "1. Discord Webhook (for notifications)"
echo "2. PyPI Token (for publishing to PyPI)"
echo "3. Custom Registry Token"
echo ""
read -p "Enter secret name (e.g., DISCORD_WEBHOOK, PYPI_API_TOKEN): " SECRET_NAME

if [ -z "$SECRET_NAME" ]; then
    echo "Error: Secret name cannot be empty"
    exit 1
fi

read -sp "Enter secret value: " SECRET_VALUE
echo ""

# Get repo info
REPO=$(gh repo view --json nameWithOwner -q)

echo ""
echo "Setting secret $SECRET_NAME for $REPO..."

# Create secret
echo "$SECRET_VALUE" | gh secret set "$SECRET_NAME" --body -

echo "✓ Secret $SECRET_NAME set successfully"
echo ""

# List all secrets
echo "Current secrets:"
gh secret list

echo ""
echo "================================"
echo "Setup Complete"
echo "================================"
