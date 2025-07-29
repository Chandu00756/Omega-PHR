"""
JWT utilities for service-to-service authentication in Ω-PHR framework.

Implements ES256 (ECDSA with P-256 curve and SHA-256) for research-grade
security with short-lived tokens and proper key management.
"""

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


class AuthError(Exception):
    """Authentication-related errors."""

    pass


class JWTUtils:
    """Enterprise-grade JWT utilities with ES256 signing."""

    def __init__(self, key_dir: Path | None = None):
        """Initialize JWT utilities with key management."""
        self.key_dir = key_dir or Path.home() / ".phr" / "keys"
        self.key_dir.mkdir(parents=True, exist_ok=True)

        self.private_key_path = self.key_dir / "jwt_private.pem"
        self.public_key_path = self.key_dir / "jwt_public.pem"

        # Load or generate keys
        self._ensure_keys()

    def _ensure_keys(self) -> None:
        """Ensure ES256 key pair exists."""
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            self._generate_key_pair()

        # Load keys
        with open(self.private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

        with open(self.public_key_path, "rb") as f:
            self.public_key = serialization.load_pem_public_key(f.read())

    def _generate_key_pair(self) -> None:
        """Generate new ES256 key pair."""
        # Generate private key
        private_key = ec.generate_private_key(ec.SECP256R1())

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        # Save keys
        with open(self.private_key_path, "wb") as f:
            f.write(private_pem)
        with open(self.public_key_path, "wb") as f:
            f.write(public_pem)

        # Secure permissions
        self.private_key_path.chmod(0o600)
        self.public_key_path.chmod(0o644)

    def _base64url_encode(self, data: bytes) -> str:
        """Base64URL encode without padding."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _base64url_decode(self, data: str) -> bytes:
        """Base64URL decode with padding."""
        # Add padding if needed
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def issue_token(self, sub: str, ttl_s: int = 3600, **claims) -> str:
        """
        Issue a JWT token with ES256 signature.

        Args:
            sub: Subject (user/service identifier)
            ttl_s: Time to live in seconds
            **claims: Additional claims to include

        Returns:
            Compact JWS token
        """
        now = int(time.time())

        # JWT header
        header = {"alg": "ES256", "typ": "JWT", "kid": self._get_key_id()}

        # JWT payload
        payload = {
            "iss": "omega-phr",
            "sub": sub,
            "aud": "omega-phr-services",
            "iat": now,
            "exp": now + ttl_s,
            "jti": self._generate_jti(),
            **claims,
        }

        # Encode header and payload
        encoded_header = self._base64url_encode(
            json.dumps(header, separators=(",", ":")).encode()
        )
        encoded_payload = self._base64url_encode(
            json.dumps(payload, separators=(",", ":")).encode()
        )

        # Create signing input
        signing_input = f"{encoded_header}.{encoded_payload}".encode()

        # Sign with ES256
        signature = self.private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        encoded_signature = self._base64url_encode(signature)

        # Return compact JWS
        return f"{encoded_header}.{encoded_payload}.{encoded_signature}"

    def verify_token(self, token: str) -> dict[str, Any]:
        """
        Verify JWT token and return payload.

        Args:
            token: Compact JWS token

        Returns:
            Decoded payload

        Raises:
            AuthError: If token is invalid
        """
        try:
            # Split token
            parts = token.split(".")
            if len(parts) != 3:
                raise AuthError("Invalid token format")

            encoded_header, encoded_payload, encoded_signature = parts

            # Decode header
            header = json.loads(self._base64url_decode(encoded_header))

            # Verify algorithm
            if header.get("alg") != "ES256":
                raise AuthError("Invalid algorithm")

            # Verify key ID if present
            if "kid" in header and header["kid"] != self._get_key_id():
                raise AuthError("Invalid key ID")

            # Decode signature
            signature = self._base64url_decode(encoded_signature)

            # Verify signature
            signing_input = f"{encoded_header}.{encoded_payload}".encode()
            try:
                self.public_key.verify(
                    signature, signing_input, ec.ECDSA(hashes.SHA256())
                )
            except InvalidSignature:
                raise AuthError("Invalid signature")

            # Decode payload
            payload = json.loads(self._base64url_decode(encoded_payload))

            # Verify expiration
            now = int(time.time())
            if payload.get("exp", 0) < now:
                raise AuthError("Token expired")

            # Verify not before
            if payload.get("nbf", 0) > now:
                raise AuthError("Token not yet valid")

            return payload

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            raise AuthError(f"Token parsing error: {e}")

    def _get_key_id(self) -> str:
        """Get key identifier for this key pair."""
        # Use SHA-256 of public key DER bytes as key ID
        public_der = self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return hashlib.sha256(public_der).hexdigest()[:16]

    def _generate_jti(self) -> str:
        """Generate unique JWT ID."""
        import secrets

        return secrets.token_urlsafe(16)

    def create_service_token(
        self, service_name: str, permissions: list[str] = None
    ) -> str:
        """Create service-to-service authentication token."""
        claims = {
            "service": service_name,
            "type": "service",
            "permissions": permissions or [],
        }
        return self.issue_token(sub=f"service:{service_name}", ttl_s=3600, **claims)

    def extract_service_info(self, token: str) -> dict[str, Any]:
        """Extract service information from token."""
        payload = self.verify_token(token)

        if payload.get("type") != "service":
            raise AuthError("Not a service token")

        return {
            "service": payload.get("service"),
            "permissions": payload.get("permissions", []),
            "expires_at": payload.get("exp"),
            "issued_at": payload.get("iat"),
        }


# Global instance for convenience
_jwt_utils = None


def get_jwt_utils() -> JWTUtils:
    """Get global JWT utilities instance."""
    global _jwt_utils
    if _jwt_utils is None:
        _jwt_utils = JWTUtils()
    return _jwt_utils


def issue_token(sub: str, ttl_s: int = 3600, **claims) -> str:
    """Convenience function to issue token."""
    return get_jwt_utils().issue_token(sub, ttl_s, **claims)


def verify_token(token: str) -> dict[str, Any]:
    """Convenience function to verify token."""
    return get_jwt_utils().verify_token(token)
