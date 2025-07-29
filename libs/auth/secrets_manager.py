"""
Secrets management for Ω-PHR framework.

Provides secure storage and retrieval of secrets using platform-specific
secure storage (macOS Keychain locally, GCP Secret Manager in research).
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SecretsManagerError(Exception):
    """Secrets management errors."""

    pass


class SecretsManager:
    """Enterprise-grade secrets management with multi-backend support."""

    def __init__(self, backend: str = "auto"):
        """
        Initialize secrets manager.

        Args:
            backend: Backend to use ("keychain", "gcp", "file", "auto")
        """
        self.backend = self._detect_backend() if backend == "auto" else backend
        self._init_backend()

    def _detect_backend(self) -> str:
        """Auto-detect the best available backend."""
        # Check for GCP environment
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GCP_PROJECT"):
            return "gcp"

        # Check for macOS
        if os.uname().sysname == "Darwin":
            return "keychain"

        # Fallback to file-based storage
        return "file"

    def _init_backend(self) -> None:
        """Initialize the selected backend."""
        if self.backend == "keychain":
            self._init_keychain()
        elif self.backend == "gcp":
            self._init_gcp()
        elif self.backend == "file":
            self._init_file()
        else:
            raise SecretsManagerError(f"Unknown backend: {self.backend}")

    def _init_keychain(self) -> None:
        """Initialize macOS Keychain backend."""
        try:
            import keyring

            self.keyring = keyring
            # Test keychain access
            self.keyring.get_password("omega-phr-test", "test")
        except ImportError:
            logger.warning("keyring not available, falling back to file backend")
            self.backend = "file"
            self._init_file()
        except Exception as e:
            logger.warning(f"Keychain access failed: {e}, falling back to file backend")
            self.backend = "file"
            self._init_file()

    def _init_gcp(self) -> None:
        """Initialize GCP Secret Manager backend."""
        try:
            from google.cloud import secretmanager

            self.client = secretmanager.SecretManagerServiceClient()
            self.project_id = os.getenv("GCP_PROJECT") or self._detect_gcp_project()
            if not self.project_id:
                raise SecretsManagerError("GCP project not detected")
        except ImportError:
            logger.warning(
                "google-cloud-secret-manager not available, falling back to file backend"
            )
            self.backend = "file"
            self._init_file()
        except Exception as e:
            logger.warning(
                f"GCP Secret Manager initialization failed: {e}, falling back to file backend"
            )
            self.backend = "file"
            self._init_file()

    def _init_file(self) -> None:
        """Initialize file-based backend (development only)."""
        self.secrets_dir = Path.home() / ".phr" / "secrets"
        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        # Secure permissions
        os.chmod(self.secrets_dir, 0o700)

        logger.warning("Using file-based secrets storage - NOT SUITABLE FOR ENTERPRISE")

    def _detect_gcp_project(self) -> str | None:
        """Detect GCP project from metadata service."""
        try:
            import requests

            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
                timeout=2,
            )
            return response.text if response.status_code == 200 else None
        except Exception:
            return None

    def get_secret(self, name: str) -> str:
        """
        Retrieve a secret by name.

        Args:
            name: Secret name/key

        Returns:
            Secret value

        Raises:
            SecretsManagerError: If secret not found or access fails
        """
        try:
            if self.backend == "keychain":
                return self._get_keychain_secret(name)
            elif self.backend == "gcp":
                return self._get_gcp_secret(name)
            elif self.backend == "file":
                return self._get_file_secret(name)
            else:
                raise SecretsManagerError(f"Backend not supported: {self.backend}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to retrieve secret '{name}': {e}")

    def put_secret(self, name: str, value: str) -> None:
        """
        Store a secret.

        Args:
            name: Secret name/key
            value: Secret value

        Raises:
            SecretsManagerError: If storage fails
        """
        try:
            if self.backend == "keychain":
                self._put_keychain_secret(name, value)
            elif self.backend == "gcp":
                self._put_gcp_secret(name, value)
            elif self.backend == "file":
                self._put_file_secret(name, value)
            else:
                raise SecretsManagerError(f"Backend not supported: {self.backend}")
        except Exception as e:
            raise SecretsManagerError(f"Failed to store secret '{name}': {e}")

    def delete_secret(self, name: str) -> None:
        """Delete a secret."""
        try:
            if self.backend == "keychain":
                self._delete_keychain_secret(name)
            elif self.backend == "gcp":
                self._delete_gcp_secret(name)
            elif self.backend == "file":
                self._delete_file_secret(name)
        except Exception as e:
            raise SecretsManagerError(f"Failed to delete secret '{name}': {e}")

    def list_secrets(self) -> list[str]:
        """List available secret names."""
        try:
            if self.backend == "keychain":
                return self._list_keychain_secrets()
            elif self.backend == "gcp":
                return self._list_gcp_secrets()
            elif self.backend == "file":
                return self._list_file_secrets()
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []

    # Keychain backend methods
    def _get_keychain_secret(self, name: str) -> str:
        """Get secret from macOS Keychain."""
        secret = self.keyring.get_password("omega-phr", name)
        if secret is None:
            raise SecretsManagerError(f"Secret '{name}' not found in keychain")
        return secret

    def _put_keychain_secret(self, name: str, value: str) -> None:
        """Store secret in macOS Keychain."""
        self.keyring.set_password("omega-phr", name, value)

    def _delete_keychain_secret(self, name: str) -> None:
        """Delete secret from macOS Keychain."""
        try:
            self.keyring.delete_password("omega-phr", name)
        except Exception:
            pass  # Secret might not exist

    def _list_keychain_secrets(self) -> list[str]:
        """List secrets in macOS Keychain."""
        # Keyring doesn't provide list functionality
        return []

    # GCP Secret Manager backend methods
    def _get_gcp_secret(self, name: str) -> str:
        """Get secret from GCP Secret Manager."""
        secret_name = f"projects/{self.project_id}/secrets/{name}/versions/latest"
        response = self.client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode("UTF-8")

    def _put_gcp_secret(self, name: str, value: str) -> None:
        """Store secret in GCP Secret Manager."""
        parent = f"projects/{self.project_id}"

        # Create secret if it doesn't exist
        try:
            secret = self.client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
        except Exception:
            # Secret already exists
            secret_path = f"projects/{self.project_id}/secrets/{name}"
        else:
            secret_path = secret.name

        # Add secret version
        self.client.add_secret_version(
            request={
                "parent": secret_path,
                "payload": {"data": value.encode("UTF-8")},
            }
        )

    def _delete_gcp_secret(self, name: str) -> None:
        """Delete secret from GCP Secret Manager."""
        secret_name = f"projects/{self.project_id}/secrets/{name}"
        self.client.delete_secret(request={"name": secret_name})

    def _list_gcp_secrets(self) -> list[str]:
        """List secrets in GCP Secret Manager."""
        parent = f"projects/{self.project_id}"
        secrets = self.client.list_secrets(request={"parent": parent})
        return [secret.name.split("/")[-1] for secret in secrets]

    # File backend methods (development only)
    def _get_file_secret(self, name: str) -> str:
        """Get secret from file."""
        secret_file = self.secrets_dir / f"{name}.json"
        if not secret_file.exists():
            raise SecretsManagerError(f"Secret '{name}' not found")

        with open(secret_file) as f:
            data = json.load(f)

        return data["value"]

    def _put_file_secret(self, name: str, value: str) -> None:
        """Store secret in file."""
        secret_file = self.secrets_dir / f"{name}.json"

        data = {
            "name": name,
            "value": value,
            "created_at": time.time(),
            "backend": "file",
        }

        with open(secret_file, "w") as f:
            json.dump(data, f, indent=2)

        # Secure file permissions
        os.chmod(secret_file, 0o600)

    def _delete_file_secret(self, name: str) -> None:
        """Delete secret file."""
        secret_file = self.secrets_dir / f"{name}.json"
        if secret_file.exists():
            secret_file.unlink()

    def _list_file_secrets(self) -> list[str]:
        """List secrets in file backend."""
        return [f.stem for f in self.secrets_dir.glob("*.json")]


# Global instance
_secrets_manager = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(name: str) -> str:
    """Convenience function to get secret."""
    return get_secrets_manager().get_secret(name)


def put_secret(name: str, value: str) -> None:
    """Convenience function to store secret."""
    return get_secrets_manager().put_secret(name, value)
