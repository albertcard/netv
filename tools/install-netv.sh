#!/bin/bash
# Install netv systemd service
# Prerequisites: uv, install-letsencrypt.sh
set -e

IPTV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER="${SUDO_USER:-$USER}"
HOME_DIR=$(eval echo "~$USER")

# Validate
if [ "$USER" = "root" ]; then
    echo "Error: Run with sudo, not as root directly"
    echo "Usage: sudo $0"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install from https://docs.astral.sh/uv/"
    exit 1
fi

UV_PATH=$(which uv)

if [ ! -d /etc/letsencrypt/live ]; then
    echo "Warning: Let's Encrypt not configured. Run install-letsencrypt.sh first for HTTPS."
    echo "Continuing with HTTP-only setup..."
    HTTPS_FLAG=""
else
    HTTPS_FLAG="--https"
fi

echo "=== Installing netv for user: $USER ==="

echo "=== Adding $USER to ssl-cert group ==="
sudo usermod -aG ssl-cert "$USER"

echo "=== Installing netv systemd service ==="

cat <<EOF | sudo tee /etc/systemd/system/netv.service
[Unit]
Description=NetV IPTV Server
After=network.target

[Service]
Type=simple
User=$USER
Group=ssl-cert
WorkingDirectory=$IPTV_DIR
ExecStart=$UV_PATH --quiet run --no-sync ./main.py $HTTPS_FLAG
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable netv
sudo systemctl start netv

if [ -n "$HTTPS_FLAG" ]; then
    echo "=== Installing certbot deploy hook (restart netv on renewal) ==="
    cat <<'EOF' | sudo tee /etc/letsencrypt/renewal-hooks/deploy/netv
#!/bin/bash
# Restart netv after cert renewal
systemctl restart netv
EOF
    sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/netv
fi

echo ""
echo "=== Done ==="
echo ""
echo "Commands:"
echo "  sudo systemctl status netv     # Check status"
echo "  sudo systemctl restart netv    # Restart after code changes"
echo "  journalctl -u netv -f          # View logs"
