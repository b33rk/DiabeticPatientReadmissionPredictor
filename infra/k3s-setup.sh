#!/bin/bash
# Run once on the server node to bootstrap K3s and required components.
# Usage: sudo bash infra/k3s-setup.sh
set -euo pipefail

echo "==> Installing K3s (single-node, server mode)"
# Disable Traefik's default TLS redirect since we're HTTP-only for this project
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -

echo "==> Waiting for K3s to be ready"
until kubectl get nodes 2>/dev/null | grep -q "Ready"; do
  sleep 3
done
echo "K3s node is ready"

echo "==> Setting up kubeconfig for current user"
mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
chmod 600 "$HOME/.kube/config"

echo "==> Installing NVIDIA device plugin (for P100 GPU)"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

echo "==> Verifying local-path provisioner (built into K3s)"
kubectl get storageclass local-path

echo "==> Installing Traefik as ingress controller"
kubectl apply -f https://raw.githubusercontent.com/traefik/traefik/v3.0/docs/content/reference/dynamic-configuration/kubernetes-crd-definition-v1.yml
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server" sh -

echo ""
echo "==> Setup complete. Next steps:"
echo "    1. Base64-encode your kubeconfig and add it to GitHub Secrets as KUBECONFIG_B64:"
echo "       cat ~/.kube/config | base64 | tr -d '\\n'"
echo "    2. Run: make deploy"
