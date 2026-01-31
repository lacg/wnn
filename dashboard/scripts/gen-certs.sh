#!/bin/bash
# Generate self-signed TLS certificates for local development
# Uses ECDSA P-384 for security and rustls compatibility

set -e

CERT_DIR="${1:-$(dirname "$0")/../certs}"
DAYS=3650  # 10 years

mkdir -p "$CERT_DIR"

echo "Generating TLS certificates in $CERT_DIR..."

# Generate private key (ECDSA P-384)
openssl ecparam -genkey -name secp384r1 -out "$CERT_DIR/key.pem"

# Generate self-signed certificate
openssl req -new -x509 \
    -key "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -days $DAYS \
    -subj "/CN=localhost/O=WNN Dashboard/C=US" \
    -addext "subjectAltName=DNS:localhost,DNS:macstudio.local,IP:127.0.0.1,IP:0.0.0.0"

chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"

echo "Certificates generated in $CERT_DIR (valid for 10 years)"
echo "  - cert.pem: Certificate"
echo "  - key.pem: Private key (keep secure!)"
