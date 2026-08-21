"""Checksum-addressed, atomic acquisition of the locked official sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urlparse

import requests

from .config import ONET_FILENAMES, ONET_TABLES, ProjectConfig
from .schemas import SchemaError, read_bls_workbook, read_onet_csv, schema_fingerprint

MANIFEST_VERSION = 1
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_SOURCE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4_096
CHUNK_SIZE = 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")

EXPECTED_SOURCE_LAYOUT: Mapping[str, tuple[str, str, str, str]] = {
    **{f"onet.{table}": ("onet", table, ONET_FILENAMES[table], "csv") for table in ONET_TABLES},
    "bls.occupation": ("bls", "occupation", "occupation.xlsx", "xlsx"),
    "bls.crosswalk": (
        "bls",
        "crosswalk",
        "nem-onet-to-soc-crosswalk.xlsx",
        "xlsx",
    ),
}

CSV_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "text/csv",
        "application/csv",
        "text/plain",
    }
)
XLSX_CONTENT_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/octet-stream",
        "binary/octet-stream",
    }
)


class AcquisitionError(RuntimeError):
    """Base class for source acquisition failures."""


class PinMismatchError(AcquisitionError):
    """Raised when refreshed publisher bytes differ from an immutable pin."""


class ManifestError(AcquisitionError):
    """Raised when a manifest or checksum-addressed object is invalid."""


class HttpSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    family: str
    table: str
    publisher: str
    vintage: str
    requested_url: str
    filename: str
    media_type: str

    @property
    def allowed_content_types(self) -> frozenset[str]:
        return CSV_CONTENT_TYPES if self.media_type == "csv" else XLSX_CONTENT_TYPES


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    path: str
    byte_count: int
    compressed_byte_count: int
    crc32: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceObject:
    source_id: str
    family: str
    table: str
    publisher: str
    vintage: str
    filename: str
    media_type: str
    requested_url: str
    resolved_url: str
    redirect_chain: tuple[str, ...]
    retrieved_at_utc: str
    status_code: int
    content_type: str
    etag: str | None
    last_modified: str | None
    response_date: str | None
    content_disposition: str | None
    byte_count: int
    sha256: str
    object_path: str
    row_count: int
    schema_fingerprint: str
    parser_name: str
    parser_version: str
    archive_members: tuple[ArchiveMember, ...]


@dataclass(frozen=True, slots=True)
class SourceManifest:
    manifest_version: int
    snapshot_id: str
    created_at_utc: str
    project_name: str
    project_artifact_version: str
    project_snapshot_date: str
    config_file_sha256: str
    config_semantic_sha256: str
    repository_commit: str | None
    sources: tuple[SourceObject, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sources"] = [asdict(source) for source in self.sources]
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SourceManifest:
        try:
            expected_manifest_fields = {field.name for field in fields(cls)}
            if set(raw) != expected_manifest_fields:
                raise ManifestError(
                    "Source manifest fields differ from schema v1: "
                    f"missing={sorted(expected_manifest_fields - set(raw))}, "
                    f"unknown={sorted(set(raw) - expected_manifest_fields)}"
                )
            raw_version = raw["manifest_version"]
            if isinstance(raw_version, bool) or not isinstance(raw_version, int):
                raise ManifestError("Source manifest version must be an integer")
            version = raw_version
            if version != MANIFEST_VERSION:
                raise ManifestError(
                    f"Unsupported source manifest version {version}; expected {MANIFEST_VERSION}"
                )
            raw_sources = raw["sources"]
            if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
                raise ManifestError("Source manifest sources must be an ordered array")
            expected_source_fields = {field.name for field in fields(SourceObject)}
            sources: list[SourceObject] = []
            for index, item in enumerate(raw_sources):
                if not isinstance(item, Mapping):
                    raise ManifestError(f"Source entry {index} must be an object")
                if set(item) != expected_source_fields:
                    raise ManifestError(
                        f"Source entry {index} fields differ from schema v1: "
                        f"missing={sorted(expected_source_fields - set(item))}, "
                        f"unknown={sorted(set(item) - expected_source_fields)}"
                    )
                raw_members = item["archive_members"]
                if isinstance(raw_members, (str, bytes)) or not isinstance(raw_members, Sequence):
                    raise ManifestError(
                        f"Source entry {index} archive_members must be an ordered array"
                    )
                expected_member_fields = {field.name for field in fields(ArchiveMember)}
                members: list[ArchiveMember] = []
                for member_index, member in enumerate(raw_members):
                    if not isinstance(member, Mapping) or set(member) != expected_member_fields:
                        raise ManifestError(
                            f"Source entry {index} archive member {member_index} "
                            "differs from schema v1"
                        )
                    members.append(ArchiveMember(**dict(member)))
                source_values = dict(item)
                raw_redirects = item["redirect_chain"]
                if isinstance(raw_redirects, (str, bytes)) or not isinstance(
                    raw_redirects, Sequence
                ):
                    raise ManifestError(
                        f"Source entry {index} redirect_chain must be an ordered array"
                    )
                source_values["redirect_chain"] = tuple(raw_redirects)
                source_values["archive_members"] = tuple(members)
                sources.append(SourceObject(**source_values))
            manifest = cls(
                manifest_version=version,
                snapshot_id=raw["snapshot_id"],
                created_at_utc=raw["created_at_utc"],
                project_name=raw["project_name"],
                project_artifact_version=raw["project_artifact_version"],
                project_snapshot_date=raw["project_snapshot_date"],
                config_file_sha256=raw["config_file_sha256"],
                config_semantic_sha256=raw["config_semantic_sha256"],
                repository_commit=(
                    None if raw["repository_commit"] is None else raw["repository_commit"]
                ),
                sources=tuple(sources),
            )
            _validate_manifest_metadata(manifest)
            return manifest
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"Malformed source manifest: {exc}") from exc


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ManifestError(f"{field} must be a trimmed, nonempty string")
    if any(ord(character) < 32 for character in value):
        raise ManifestError(f"{field} contains control characters")
    return value


def _require_integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_utc_timestamp(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field)
    if not text.endswith("Z"):
        raise ManifestError(f"{field} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ManifestError(f"{field} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ManifestError(f"{field} must be UTC")
    if _format_utc(parsed) != text:
        raise ManifestError(f"{field} is not in canonical second-resolution form")
    return text


def _validate_optional_header(value: Any, field: str) -> None:
    if value is None:
        return
    _require_nonempty_string(value, field)


def _validate_manifest_url(value: Any, field: str) -> None:
    try:
        _validate_url(value, field)
    except AcquisitionError as exc:
        raise ManifestError(str(exc)) from exc


def _validate_manifest_metadata(manifest: SourceManifest) -> None:
    """Validate the complete, versioned manifest before trusting object paths."""

    try:
        _safe_snapshot_id(_require_nonempty_string(manifest.snapshot_id, "snapshot_id"))
    except AcquisitionError as exc:
        raise ManifestError(str(exc)) from exc
    _require_nonempty_string(manifest.project_name, "project_name")
    try:
        artifact_version = _safe_snapshot_id(
            _require_nonempty_string(manifest.project_artifact_version, "project_artifact_version")
        )
    except AcquisitionError as exc:
        raise ManifestError(str(exc)) from exc
    if artifact_version != manifest.snapshot_id:
        raise ManifestError(
            "snapshot_id must equal project_artifact_version; bump the locked config "
            "for a new snapshot"
        )
    try:
        snapshot_date = datetime.strptime(
            _require_nonempty_string(manifest.project_snapshot_date, "project_snapshot_date"),
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise ManifestError("project_snapshot_date must be an ISO calendar date") from exc
    if snapshot_date.isoformat() != manifest.project_snapshot_date:
        raise ManifestError("project_snapshot_date must use canonical YYYY-MM-DD form")
    created_at = _validate_utc_timestamp(manifest.created_at_utc, "created_at_utc")
    for field_name, digest in (
        ("config_file_sha256", manifest.config_file_sha256),
        ("config_semantic_sha256", manifest.config_semantic_sha256),
    ):
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ManifestError(f"{field_name} must be a lowercase SHA-256 digest")
    if manifest.repository_commit is not None and (
        not isinstance(manifest.repository_commit, str)
        or not COMMIT_PATTERN.fullmatch(manifest.repository_commit)
    ):
        raise ManifestError("repository_commit must be null or a lowercase Git commit hash")

    expected_ids = tuple(EXPECTED_SOURCE_LAYOUT)
    actual_ids = tuple(source.source_id for source in manifest.sources)
    if actual_ids != expected_ids:
        raise ManifestError(
            "Manifest must contain the nine locked sources in protocol order; "
            f"expected {expected_ids}, got {actual_ids}"
        )
    for source in manifest.sources:
        context = source.source_id
        family, table, filename, media_type = EXPECTED_SOURCE_LAYOUT[context]
        if (source.family, source.table, source.filename, source.media_type) != (
            family,
            table,
            filename,
            media_type,
        ):
            raise ManifestError(f"{context}: source identity differs from the locked layout")
        for field_name in ("publisher", "vintage", "content_type", "parser_name", "parser_version"):
            _require_nonempty_string(getattr(source, field_name), f"{context}.{field_name}")
        _validate_manifest_url(source.requested_url, f"{context} requested URL")
        _validate_manifest_url(source.resolved_url, f"{context} resolved URL")
        if not isinstance(source.redirect_chain, tuple):
            raise ManifestError(f"{context}.redirect_chain must be an ordered array")
        for index, redirect_url in enumerate(source.redirect_chain):
            _validate_manifest_url(redirect_url, f"{context}.redirect_chain[{index}]")
        _validate_utc_timestamp(source.retrieved_at_utc, f"{context}.retrieved_at_utc")
        if source.retrieved_at_utc != created_at:
            raise ManifestError(f"{context}: retrieval timestamp differs from snapshot timestamp")
        _require_integer(source.status_code, f"{context}.status_code", minimum=200)
        if source.status_code >= 300:
            raise ManifestError(f"{context}.status_code must be a successful 2xx response")
        _require_integer(source.byte_count, f"{context}.byte_count", minimum=1)
        _require_integer(source.row_count, f"{context}.row_count", minimum=1)
        if not isinstance(source.sha256, str) or not SHA256_PATTERN.fullmatch(source.sha256):
            raise ManifestError(f"{context}.sha256 must be a lowercase SHA-256 digest")
        if not isinstance(source.schema_fingerprint, str) or not SHA256_PATTERN.fullmatch(
            source.schema_fingerprint
        ):
            raise ManifestError(f"{context}.schema_fingerprint must be a lowercase SHA-256 digest")
        expected_path = f"objects/sha256/{source.sha256}"
        if source.object_path != expected_path:
            raise ManifestError(f"{context}: object path is not checksum-addressed")
        allowed_types = CSV_CONTENT_TYPES if media_type == "csv" else XLSX_CONTENT_TYPES
        if source.content_type not in allowed_types:
            raise ManifestError(f"{context}: recorded Content-Type is not allowed")
        for header_name in (
            "etag",
            "last_modified",
            "response_date",
            "content_disposition",
        ):
            _validate_optional_header(getattr(source, header_name), f"{context}.{header_name}")
        if not isinstance(source.archive_members, tuple):
            raise ManifestError(f"{context}.archive_members must be an ordered array")
        if media_type == "csv" and source.archive_members:
            raise ManifestError(f"{context}: CSV sources cannot contain archive members")
        if media_type == "xlsx" and not source.archive_members:
            raise ManifestError(f"{context}: workbook archive-member provenance is missing")
        member_paths: set[str] = set()
        for index, member in enumerate(source.archive_members):
            member_context = f"{context}.archive_members[{index}]"
            member_path = _require_nonempty_string(member.path, f"{member_context}.path")
            relative = PurePosixPath(member_path)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member_path:
                raise ManifestError(f"{member_context}.path is unsafe")
            if member_path in member_paths:
                raise ManifestError(f"{context}: duplicate archive member path {member_path!r}")
            member_paths.add(member_path)
            _require_integer(member.byte_count, f"{member_context}.byte_count", minimum=0)
            _require_integer(
                member.compressed_byte_count,
                f"{member_context}.compressed_byte_count",
                minimum=0,
            )
            if not isinstance(member.crc32, str) or not re.fullmatch(r"[0-9a-f]{8}", member.crc32):
                raise ManifestError(f"{member_context}.crc32 must be eight lowercase hex digits")
            if not isinstance(member.sha256, str) or not SHA256_PATTERN.fullmatch(member.sha256):
                raise ManifestError(f"{member_context}.sha256 must be a lowercase SHA-256 digest")


def build_source_specs(config: ProjectConfig) -> tuple[SourceSpec, ...]:
    """Create the seven O*NET CSV and two BLS workbook requests."""

    specs: list[SourceSpec] = []
    for table in config.onet.tables:
        filename = config.onet.filename(table)
        specs.append(
            SourceSpec(
                source_id=f"onet.{table}",
                family="onet",
                table=table,
                publisher=config.onet.publisher,
                vintage=config.onet.release,
                requested_url=f"{config.onet.base_url}/{quote(filename, safe='._-')}",
                filename=filename,
                media_type="csv",
            )
        )
    specs.extend(
        (
            SourceSpec(
                source_id="bls.occupation",
                family="bls",
                table="occupation",
                publisher=config.bls.publisher,
                vintage=config.bls.projection_vintage,
                requested_url=config.bls.occupation_url,
                filename=Path(urlparse(config.bls.occupation_url).path).name,
                media_type="xlsx",
            ),
            SourceSpec(
                source_id="bls.crosswalk",
                family="bls",
                table="crosswalk",
                publisher=config.bls.publisher,
                vintage=config.bls.projection_vintage,
                requested_url=config.bls.crosswalk_url,
                filename=Path(urlparse(config.bls.crosswalk_url).path).name,
                media_type="xlsx",
            ),
        )
    )
    if len(specs) != 9 or len({spec.source_id for spec in specs}) != 9:
        raise AcquisitionError("The locked acquisition plan must contain nine unique sources")
    return tuple(specs)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_snapshot_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value):
        raise AcquisitionError(f"Unsafe snapshot identifier: {value!r}")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise AcquisitionError(
            f"Required parser distribution is not installed: {distribution}"
        ) from exc


def _parser_identity(spec: SourceSpec) -> tuple[str, str]:
    pandas_version = _installed_version("pandas")
    if spec.media_type == "csv":
        return "pandas.read_csv", f"pandas=={pandas_version}"
    return (
        "pandas.read_excel[openpyxl]",
        f"pandas=={pandas_version};openpyxl=={_installed_version('openpyxl')}",
    )


def _repository_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value).strip() or None
    return None


def _validate_url(url: str, context: str) -> None:
    if not isinstance(url, str):
        raise AcquisitionError(f"{context} must be a string")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise AcquisitionError(f"{context} is not a credential-free HTTPS URL: {url!r}")


def _validate_xlsx(path: Path, table: str) -> tuple[ArchiveMember, ...]:
    if not zipfile.is_zipfile(path):
        raise AcquisitionError(f"bls.{table}: response is not an OpenXML workbook")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
                raise AcquisitionError(
                    f"bls.{table}: workbook member count is outside the safety limit"
                )
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise AcquisitionError(f"bls.{table}: workbook contains duplicate member paths")
            required = {"[Content_Types].xml", "xl/workbook.xml"}
            if not required.issubset(names):
                raise AcquisitionError(f"bls.{table}: incomplete OpenXML workbook")
            total_uncompressed = 0
            members: list[ArchiveMember] = []
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or "\\" in info.filename
                    or not info.filename
                ):
                    raise AcquisitionError(f"bls.{table}: unsafe workbook member {info.filename!r}")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or info.flag_bits & 0x1:
                    raise AcquisitionError(
                        f"bls.{table}: encrypted or linked workbook member {info.filename!r}"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_EXPANDED_SOURCE_BYTES:
                    raise AcquisitionError(f"bls.{table}: implausibly large expanded workbook")
                digest = hashlib.sha256()
                observed_size = 0
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                        observed_size += len(chunk)
                        digest.update(chunk)
                if observed_size != info.file_size:
                    raise AcquisitionError(
                        f"bls.{table}: member size mismatch for {info.filename!r}"
                    )
                members.append(
                    ArchiveMember(
                        path=info.filename,
                        byte_count=info.file_size,
                        compressed_byte_count=info.compress_size,
                        crc32=f"{info.CRC:08x}",
                        sha256=digest.hexdigest(),
                    )
                )
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"bls.{table}: invalid OpenXML ZIP container") from exc
    return tuple(members)


def _validate_download(path: Path, spec: SourceSpec) -> tuple[int, str, tuple[ArchiveMember, ...]]:
    if spec.media_type == "csv":
        with path.open("rb") as handle:
            prefix = handle.read(8192).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"\x00" in prefix:
            raise AcquisitionError(f"{spec.source_id}: non-CSV response body")
        try:
            frame = read_onet_csv(path, spec.table)
        except SchemaError as exc:
            raise AcquisitionError(str(exc)) from exc
        archive_members: tuple[ArchiveMember, ...] = ()
    else:
        archive_members = _validate_xlsx(path, spec.table)
        try:
            frame = read_bls_workbook(path, spec.table)
        except SchemaError as exc:
            raise AcquisitionError(str(exc)) from exc
    return (
        len(frame),
        schema_fingerprint(frame.columns.tolist(), frame.dtypes.tolist()),
        archive_members,
    )


def _download_one(
    session: HttpSession,
    spec: SourceSpec,
    staging: Path,
    *,
    timeout: tuple[float, float],
    retrieved_at: str,
) -> SourceObject:
    _validate_url(spec.requested_url, f"{spec.source_id} requested URL")
    try:
        response = session.get(
            spec.requested_url,
            allow_redirects=True,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": "ai-job-market-intelligence/0.1 official-data-acquisition"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise AcquisitionError(f"{spec.source_id}: HTTP acquisition failed: {exc}") from exc
    except Exception as exc:
        raise AcquisitionError(f"{spec.source_id}: HTTP acquisition failed: {exc}") from exc

    resolved_url = str(getattr(response, "url", spec.requested_url))
    _validate_url(resolved_url, f"{spec.source_id} resolved URL")
    headers = getattr(response, "headers", {})
    content_type_header = _header(headers, "Content-Type")
    content_type = (content_type_header or "").split(";", 1)[0].strip().lower()
    if not content_type or content_type not in spec.allowed_content_types:
        raise AcquisitionError(
            f"{spec.source_id}: unexpected Content-Type {content_type_header!r}; "
            f"allowed {sorted(spec.allowed_content_types)}"
        )
    content_length = _header(headers, "Content-Length")
    if content_length is not None:
        try:
            advertised_size = int(content_length)
        except ValueError as exc:
            raise AcquisitionError(f"{spec.source_id}: invalid Content-Length") from exc
        if advertised_size <= 0 or advertised_size > MAX_SOURCE_BYTES:
            raise AcquisitionError(f"{spec.source_id}: rejected Content-Length {advertised_size}")

    download_path = staging / f"download-{spec.source_id.replace('.', '-')}.tmp"
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with download_path.open("xb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                byte_count += len(chunk)
                if byte_count > MAX_SOURCE_BYTES:
                    raise AcquisitionError(f"{spec.source_id}: response exceeded size limit")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if byte_count == 0:
        raise AcquisitionError(f"{spec.source_id}: empty response")
    if content_length is not None and byte_count != int(content_length):
        raise AcquisitionError(
            f"{spec.source_id}: Content-Length mismatch ({byte_count} != {content_length})"
        )

    row_count, fingerprint, archive_members = _validate_download(download_path, spec)
    parser_name, parser_version = _parser_identity(spec)
    sha256 = digest.hexdigest()
    relative_object = PurePosixPath("objects") / "sha256" / sha256
    object_path = staging.joinpath(*relative_object.parts)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if object_path.exists():
        if sha256_file(object_path) != sha256:
            raise AcquisitionError(f"{spec.source_id}: checksum-address collision")
        download_path.unlink()
    else:
        os.replace(download_path, object_path)
        object_path.chmod(0o444)

    history = getattr(response, "history", ()) or ()
    redirect_chain = tuple(str(getattr(item, "url", "")) for item in history)
    return SourceObject(
        source_id=spec.source_id,
        family=spec.family,
        table=spec.table,
        publisher=spec.publisher,
        vintage=spec.vintage,
        filename=spec.filename,
        media_type=spec.media_type,
        requested_url=spec.requested_url,
        resolved_url=resolved_url,
        redirect_chain=redirect_chain,
        retrieved_at_utc=retrieved_at,
        status_code=int(getattr(response, "status_code", 200)),
        content_type=content_type,
        etag=_header(headers, "ETag"),
        last_modified=_header(headers, "Last-Modified"),
        response_date=_header(headers, "Date"),
        content_disposition=_header(headers, "Content-Disposition"),
        byte_count=byte_count,
        sha256=sha256,
        object_path=relative_object.as_posix(),
        row_count=row_count,
        schema_fingerprint=fingerprint,
        parser_name=parser_name,
        parser_version=parser_version,
        archive_members=archive_members,
    )


def _write_manifest(path: Path, manifest: SourceManifest) -> None:
    payload = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _acquire_staging(
    config: ProjectConfig,
    staging: Path,
    snapshot_id: str,
    session: HttpSession,
    timeout: tuple[float, float],
    clock: Callable[[], datetime],
) -> SourceManifest:
    retrieved_at = _format_utc(clock())
    if datetime.fromisoformat(retrieved_at[:-1] + "+00:00").date() != config.project.snapshot_date:
        raise AcquisitionError(
            "Acquisition UTC date differs from project.snapshot_date; update and review the "
            "locked configuration before downloading"
        )
    objects = tuple(
        _download_one(
            session,
            spec,
            staging,
            timeout=timeout,
            retrieved_at=retrieved_at,
        )
        for spec in build_source_specs(config)
    )
    manifest = SourceManifest(
        manifest_version=MANIFEST_VERSION,
        snapshot_id=snapshot_id,
        created_at_utc=retrieved_at,
        project_name=config.project.name,
        project_artifact_version=config.project.artifact_version,
        project_snapshot_date=config.project.snapshot_date.isoformat(),
        config_file_sha256=config.fingerprint,
        config_semantic_sha256=config.semantic_fingerprint,
        repository_commit=_repository_commit(config.root),
        sources=objects,
    )
    _write_manifest(staging / "manifest.json", manifest)
    return manifest


def load_manifest(path: str | Path) -> SourceManifest:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ManifestError(f"Source manifest path cannot be a symbolic link: {manifest_path}")
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if manifest_path.is_symlink():
        raise ManifestError(f"Source manifest file cannot be a symbolic link: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read source manifest {manifest_path}: {exc}") from exc
    return SourceManifest.from_dict(raw)


def manifest_directory(path: str | Path) -> Path:
    value = Path(path).resolve()
    return value if value.is_dir() else value.parent


def source_path(manifest_path: str | Path, source: SourceObject) -> Path:
    root = manifest_directory(manifest_path)
    relative = PurePosixPath(source.object_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestError(f"Unsafe object path in manifest: {source.object_path!r}")
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"Object escapes snapshot directory: {source.object_path!r}") from exc
    unresolved = root
    for part in relative.parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ManifestError(f"Object path contains a symbolic link: {source.object_path!r}")
    return candidate


def verify_snapshot(
    path: str | Path,
    *,
    config: ProjectConfig | None = None,
) -> SourceManifest:
    """Verify IDs, object paths, sizes, digests, and optional config pin."""

    manifest = load_manifest(path)
    if config is not None:
        if manifest.project_name != config.project.name:
            raise ManifestError("Manifest project name does not match configuration")
        if manifest.snapshot_id != config.snapshot_id:
            raise ManifestError("Manifest snapshot ID does not match configured artifact version")
        if manifest.project_artifact_version != config.project.artifact_version:
            raise ManifestError("Manifest artifact version does not match configuration")
        if manifest.project_snapshot_date != config.project.snapshot_date.isoformat():
            raise ManifestError("Manifest snapshot date does not match configuration")
        retrieval_date = datetime.fromisoformat(manifest.created_at_utc[:-1] + "+00:00").date()
        if retrieval_date != config.project.snapshot_date:
            raise ManifestError(
                "Manifest retrieval date does not match the configured source snapshot date"
            )
        if manifest.config_file_sha256 != config.fingerprint:
            raise ManifestError("Manifest config-file checksum does not match locked config bytes")
        if manifest.config_semantic_sha256 != config.semantic_fingerprint:
            raise ManifestError(
                "Manifest semantic configuration checksum does not match locked config"
            )
        expected_specs = {spec.source_id: spec for spec in build_source_specs(config)}
        for source in manifest.sources:
            spec = expected_specs[source.source_id]
            expected_identity = (
                spec.family,
                spec.table,
                spec.publisher,
                spec.vintage,
                spec.filename,
                spec.media_type,
                spec.requested_url,
            )
            actual_identity = (
                source.family,
                source.table,
                source.publisher,
                source.vintage,
                source.filename,
                source.media_type,
                source.requested_url,
            )
            if actual_identity != expected_identity:
                raise ManifestError(
                    f"{source.source_id}: manifest identity differs from locked config"
                )
            if (source.parser_name, source.parser_version) != _parser_identity(spec):
                raise ManifestError(
                    f"{source.source_id}: parser provenance differs from locked environment"
                )
    for source in manifest.sources:
        object_path = source_path(path, source)
        if object_path.is_symlink() or not object_path.is_file():
            raise ManifestError(f"Missing object for {source.source_id}: {object_path}")
        if object_path.stat().st_size != source.byte_count:
            raise ManifestError(f"Byte-count mismatch for {source.source_id}")
        if sha256_file(object_path) != source.sha256:
            raise ManifestError(f"SHA-256 mismatch for {source.source_id}")
        spec = SourceSpec(
            source_id=source.source_id,
            family=source.family,
            table=source.table,
            publisher=source.publisher,
            vintage=source.vintage,
            requested_url=source.requested_url,
            filename=source.filename,
            media_type=source.media_type,
        )
        try:
            row_count, fingerprint, archive_members = _validate_download(object_path, spec)
        except AcquisitionError as exc:
            raise ManifestError(f"{source.source_id}: content verification failed: {exc}") from exc
        if row_count != source.row_count:
            raise ManifestError(f"Row-count provenance mismatch for {source.source_id}")
        if fingerprint != source.schema_fingerprint:
            raise ManifestError(f"Schema provenance mismatch for {source.source_id}")
        if archive_members != source.archive_members:
            raise ManifestError(f"Archive-member provenance mismatch for {source.source_id}")

    snapshot_root = manifest_directory(path)
    expected_files = {
        (snapshot_root / "manifest.json").resolve(),
        *(source_path(path, source) for source in manifest.sources),
    }
    actual_files: set[Path] = set()
    for candidate in snapshot_root.rglob("*"):
        if candidate.is_symlink():
            raise ManifestError(f"Snapshot contains a symbolic link: {candidate}")
        if candidate.is_file():
            actual_files.add(candidate.resolve())
    if actual_files != expected_files:
        unexpected = sorted(
            str(value.relative_to(snapshot_root)) for value in actual_files - expected_files
        )
        missing = sorted(
            str(value.relative_to(snapshot_root)) for value in expected_files - actual_files
        )
        raise ManifestError(
            f"Snapshot file set differs from manifest; unexpected={unexpected}, missing={missing}"
        )
    return manifest


def _pin_signature(manifest: SourceManifest) -> dict[str, tuple[str, int, str, int]]:
    return {
        source.source_id: (
            source.sha256,
            source.byte_count,
            source.schema_fingerprint,
            source.row_count,
        )
        for source in manifest.sources
    }


def acquire_sources(
    config: ProjectConfig,
    raw_root: str | Path | None = None,
    *,
    snapshot_id: str | None = None,
    new_snapshot: bool = False,
    refresh: bool = False,
    session: HttpSession | None = None,
    timeout: tuple[float, float] = (10.0, 120.0),
    clock: Callable[[], datetime] = _utc_now,
) -> SourceManifest:
    """Acquire or verify one immutable snapshot.

    The first acquisition of the configured artifact version establishes its
    local pin.  Subsequent default calls verify local objects without network
    access.  ``refresh=True`` re-downloads all nine responses and rejects any
    byte/schema change. ``new_snapshot=True`` asserts that the explicitly named,
    config-matching version does not already exist. A genuinely new snapshot
    requires a reviewed ``artifact_version`` and ``snapshot_date`` config bump.
    """

    chosen_id = _safe_snapshot_id(snapshot_id or config.snapshot_id)
    if chosen_id != config.snapshot_id:
        raise AcquisitionError(
            "snapshot_id must equal the configured artifact_version; update the locked config "
            "for a new source snapshot"
        )
    if new_snapshot and snapshot_id is None:
        raise AcquisitionError("--new-snapshot requires an explicit snapshot_id")
    if new_snapshot and refresh:
        raise AcquisitionError("new_snapshot and refresh are mutually exclusive")

    root = Path(raw_root or config.root / "data" / "raw").expanduser().resolve()
    snapshots_root = root / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    target = snapshots_root / chosen_id
    if target.exists() and new_snapshot:
        raise AcquisitionError(f"Snapshot already exists and is immutable: {target}")
    if refresh and not target.exists():
        raise AcquisitionError("--refresh requires an existing immutable snapshot pin")
    if target.exists() and not refresh:
        return verify_snapshot(target, config=config)

    http_session = session or requests.Session()
    stage = Path(tempfile.mkdtemp(prefix=f".{chosen_id}.staging-", dir=snapshots_root))
    try:
        candidate = _acquire_staging(
            config,
            stage,
            chosen_id,
            http_session,
            timeout,
            clock,
        )
        verify_snapshot(stage, config=config)
        if target.exists():
            pinned = verify_snapshot(target, config=config)
            if _pin_signature(candidate) != _pin_signature(pinned):
                changed = sorted(
                    source_id
                    for source_id in set(_pin_signature(candidate)) | set(_pin_signature(pinned))
                    if _pin_signature(candidate).get(source_id)
                    != _pin_signature(pinned).get(source_id)
                )
                raise PinMismatchError(
                    "Publisher bytes/schema differ from the locked snapshot for: "
                    + ", ".join(changed)
                    + ". Use a new project snapshot/version; the pin was not overwritten."
                )
            return pinned
        os.replace(stage, target)
        stage = target  # prevent cleanup of the published directory
        return verify_snapshot(target, config=config)
    finally:
        if stage.exists() and stage != target:
            shutil.rmtree(stage)
        if session is None:
            close = getattr(http_session, "close", None)
            if callable(close):
                close()


def iter_source_paths(
    manifest_path: str | Path, manifest: SourceManifest | None = None
) -> Iterable[tuple[SourceObject, Path]]:
    loaded = manifest or verify_snapshot(manifest_path)
    for source in loaded.sources:
        yield source, source_path(manifest_path, source)


__all__ = [
    "AcquisitionError",
    "ArchiveMember",
    "CSV_CONTENT_TYPES",
    "EXPECTED_SOURCE_LAYOUT",
    "MANIFEST_VERSION",
    "ManifestError",
    "PinMismatchError",
    "SourceManifest",
    "SourceObject",
    "SourceSpec",
    "XLSX_CONTENT_TYPES",
    "acquire_sources",
    "build_source_specs",
    "iter_source_paths",
    "load_manifest",
    "manifest_directory",
    "sha256_file",
    "source_path",
    "verify_snapshot",
]
