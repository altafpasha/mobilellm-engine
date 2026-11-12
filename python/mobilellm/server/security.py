"""
Security assertions, loopback validation, Bearer token authentication,
and path traversal prevention for MobileLLM server.
"""

import hmac
import ipaddress
import os
from pathlib import Path
from typing import Sequence

from mobilellm.exceptions import SecurityViolationError


def is_loopback_address(host: str) -> bool:
    """Returns True if host is strictly a loopback interface."""
    normalized = host.strip().lower()
    if normalized in ("127.0.0.1", "localhost", "::1"):
        return True

    try:
        ip = ipaddress.ip_address(normalized)
        return ip.is_loopback
    except ValueError:
        return False


def validate_bind_security(
    host: str,
    allow_remote: bool = False,
    api_key: str | None = None,
) -> None:
    """
    Enforces ADR-0007 security controls:
    Refuses to start on non-loopback interfaces unless --allow-remote-connections
    is explicitly supplied together with a non-empty --api-key.
    """
    if is_loopback_address(host):
        return

    # Non-loopback interface (e.g. 0.0.0.0 or LAN IP)
    if not allow_remote:
        raise SecurityViolationError(
            f"Binding to non-loopback interface '{host}' is forbidden by default. "
            "To expose MobileLLM to the network, you must supply --allow-remote-connections "
            "and configure a non-empty --api-key."
        )

    if not api_key or not api_key.strip():
        raise SecurityViolationError(
            f"Binding to remote interface '{host}' with --allow-remote-connections "
            "strictly requires configuring a non-empty --api-key."
        )


def validate_auth_token(
    auth_header: str | None,
    required_api_key: str | None,
) -> bool:
    """
    Validates Authorization: Bearer <key> header against required secret
    using constant-time comparison to prevent timing side-channels.
    If no required_api_key is configured, requests are allowed.
    """
    if not required_api_key or not required_api_key.strip():
        return True

    if not auth_header:
        return False

    parts = auth_header.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
    else:
        token = auth_header.strip()

    return hmac.compare_digest(token.encode("utf-8"), required_api_key.encode("utf-8"))


def resolve_and_validate_model_path(
    model_path: str | Path,
    allowed_dirs: Sequence[str | Path] | None = None,
) -> Path:
    """
    Canonicalizes model_path and enforces strict containment within allowed model directories.
    Rejects directory traversal (e.g. '..'), symlink escapes, null bytes,
    and paths outside the allowed model directories.

    Raises:
        SecurityViolationError: If model_path escapes allowed directories, contains null bytes,
                                or uses symlinks to target unauthorized locations.
    """
    if not model_path:
        raise SecurityViolationError("Model path cannot be empty.")

    raw_path_str = str(model_path).strip()
    if not raw_path_str:
        raise SecurityViolationError("Model path cannot be empty.")

    if "\0" in raw_path_str:
        raise SecurityViolationError("Null bytes forbidden in model path.")

    # Canonicalize allowed directories
    canonical_allowed: list[Path] = []
    if allowed_dirs:
        for d in allowed_dirs:
            p = Path(d).resolve()
            canonical_allowed.append(p)
    else:
        env_dir = os.environ.get("MOBILELLM_MODEL_DIR")
        if env_dir:
            canonical_allowed.append(Path(env_dir).resolve())
        else:
            canonical_allowed.append(Path.cwd().resolve())
            cache_models = Path.home().resolve() / ".cache" / "mobilellm" / "models"
            canonical_allowed.append(cache_models)

    path_obj = Path(raw_path_str)
    candidate_paths: list[Path] = []
    if path_obj.is_absolute():
        candidate_paths.append(path_obj)
    else:
        for ad in canonical_allowed:
            candidate_paths.append(ad / path_obj)
        candidate_paths.append(path_obj)

    resolved_path: Path | None = None
    is_contained = False

    for candidate in candidate_paths:
        try:
            canonical_cand = candidate.resolve(strict=False)
        except (ValueError, RuntimeError):
            continue

        for ad in canonical_allowed:
            try:
                if canonical_cand.is_relative_to(ad):
                    # Check realpath for symlink escapes
                    if candidate.exists():
                        real_p = Path(os.path.realpath(candidate))
                        real_ad = Path(os.path.realpath(ad))
                        if not real_p.is_relative_to(real_ad):
                            raise SecurityViolationError(
                                f"Symlink escape rejected: '{raw_path_str}' targets outside '{ad}'"
                            )
                    resolved_path = canonical_cand
                    is_contained = True
                    break
            except AttributeError:
                if str(canonical_cand).startswith(str(ad)):
                    resolved_path = canonical_cand
                    is_contained = True
                    break
        if is_contained:
            break

    if not is_contained or resolved_path is None:
        raise SecurityViolationError(
            f"Access denied: Model path '{raw_path_str}' is outside allowed directories: "
            f"{[str(d) for d in canonical_allowed]}"
        )

    return resolved_path
