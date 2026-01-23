"""SharePoint integration tools for Recruitment Deep Agent.

This module provides tools for interacting with SharePoint document libraries
for recruitment document management including resumes, JDs, and scoring files.

Following Enterprise Development Standards:
- Software Architect: Modular SharePoint operations with abstraction
- Security Architect: OAuth2 authentication, secure token handling
- Data Architect: Structured document metadata and caching
- Software Engineer: Type-safe with comprehensive error handling
"""

import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.deepagents.config.recruitment_config import get_recruitment_config

logger = logging.getLogger(__name__)

# =============================================================================
# SharePoint Client Management
# =============================================================================

# Module-level client cache per session
_sharepoint_clients: dict[str, Any] = {}
_document_cache: dict[str, dict[str, Any]] = {}


class SharePointClient:
    """SharePoint Graph API client for document operations.

    Uses Microsoft Graph API for SharePoint operations.
    Requires Azure AD application with Sites.ReadWrite.All permission.
    """

    def __init__(
        self,
        site_url: str | None = None,
        tenant_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        """Initialize SharePoint client.

        Args:
            site_url: SharePoint site URL.
            tenant_id: Azure AD tenant ID.
            client_id: Azure AD application client ID.
            client_secret: Azure AD application client secret.
        """
        config = get_recruitment_config()
        self.site_url = site_url or config.sharepoint.site_url
        self.tenant_id = tenant_id or config.sharepoint.tenant_id
        self.client_id = client_id or config.sharepoint.client_id
        self.client_secret = client_secret or config.sharepoint.client_secret

        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._site_id: str | None = None

        # Demo mode flag
        self._demo_mode = not all([
            self.site_url,
            self.tenant_id,
            self.client_id,
            self.client_secret,
        ])

        if self._demo_mode:
            logger.warning(
                "SharePoint credentials not configured. Running in demo mode."
            )

    @property
    def is_configured(self) -> bool:
        """Check if SharePoint is properly configured."""
        return not self._demo_mode

    def _get_access_token(self) -> str:
        """Get or refresh OAuth2 access token.

        Returns:
            Access token string.

        Raises:
            ValueError: If credentials not configured.
        """
        if self._demo_mode:
            return "demo_token"

        # Check if token is still valid
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                return self._access_token

        try:
            import requests

            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            }

            response = requests.post(token_url, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]

            # Token typically valid for 1 hour, refresh at 50 minutes
            from datetime import timedelta
            self._token_expiry = datetime.now() + timedelta(minutes=50)

            return self._access_token

        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise ValueError(f"SharePoint authentication failed: {e}") from e

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json",
        }

    def _get_site_id(self) -> str:
        """Get SharePoint site ID from site URL.

        Returns:
            Site ID string.
        """
        if self._site_id:
            return self._site_id

        if self._demo_mode:
            return "demo_site_id"

        try:
            import requests
            from urllib.parse import urlparse

            parsed = urlparse(self.site_url)
            hostname = parsed.netloc
            site_path = parsed.path.rstrip("/")

            # Graph API to get site by path
            graph_url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{site_path}"

            response = requests.get(
                graph_url,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()

            self._site_id = response.json()["id"]
            return self._site_id

        except Exception as e:
            logger.error(f"Failed to get site ID: {e}")
            raise ValueError(f"Could not resolve SharePoint site: {e}") from e

    def list_folder(self, folder_path: str) -> list[dict[str, Any]]:
        """List files in a SharePoint folder.

        Args:
            folder_path: Relative folder path (e.g., "Recruitment/Resumes").

        Returns:
            List of file metadata dictionaries.
        """
        if self._demo_mode:
            return self._get_demo_folder_contents(folder_path)

        try:
            import requests

            site_id = self._get_site_id()

            # Encode folder path
            encoded_path = folder_path.replace("/", ":/").replace(" ", "%20")

            graph_url = (
                f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                f"/drive/root:/{encoded_path}:/children"
            )

            response = requests.get(
                graph_url,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()

            items = response.json().get("value", [])

            return [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "size": item.get("size", 0),
                    "created": item.get("createdDateTime"),
                    "modified": item.get("lastModifiedDateTime"),
                    "type": "folder" if "folder" in item else "file",
                    "web_url": item.get("webUrl"),
                    "download_url": item.get("@microsoft.graph.downloadUrl"),
                }
                for item in items
            ]

        except Exception as e:
            logger.error(f"Failed to list folder {folder_path}: {e}")
            return []

    def download_file(self, folder_path: str, filename: str) -> bytes | None:
        """Download a file from SharePoint.

        Args:
            folder_path: Relative folder path.
            filename: Name of the file to download.

        Returns:
            File content as bytes, or None if not found.
        """
        if self._demo_mode:
            return self._get_demo_file_content(folder_path, filename)

        try:
            import requests

            site_id = self._get_site_id()
            file_path = f"{folder_path}/{filename}".replace(" ", "%20")
            encoded_path = file_path.replace("/", ":/")

            graph_url = (
                f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                f"/drive/root:/{encoded_path}:/content"
            )

            response = requests.get(
                graph_url,
                headers=self._get_headers(),
                timeout=60,
            )
            response.raise_for_status()

            return response.content

        except Exception as e:
            logger.error(f"Failed to download {folder_path}/{filename}: {e}")
            return None

    def upload_file(
        self,
        folder_path: str,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any] | None:
        """Upload a file to SharePoint.

        Args:
            folder_path: Relative folder path.
            filename: Name for the uploaded file.
            content: File content as bytes.
            content_type: MIME type of the content.

        Returns:
            Uploaded file metadata, or None if failed.
        """
        if self._demo_mode:
            return self._demo_upload_file(folder_path, filename, content)

        try:
            import requests

            site_id = self._get_site_id()
            file_path = f"{folder_path}/{filename}".replace(" ", "%20")
            encoded_path = file_path.replace("/", ":/")

            # For files < 4MB, use simple upload
            if len(content) < 4 * 1024 * 1024:
                graph_url = (
                    f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                    f"/drive/root:/{encoded_path}:/content"
                )

                headers = self._get_headers()
                headers["Content-Type"] = content_type

                response = requests.put(
                    graph_url,
                    headers=headers,
                    data=content,
                    timeout=120,
                )
                response.raise_for_status()

                return response.json()
            else:
                # For larger files, use upload session
                return self._upload_large_file(site_id, folder_path, filename, content)

        except Exception as e:
            logger.error(f"Failed to upload {folder_path}/{filename}: {e}")
            return None

    def _upload_large_file(
        self,
        site_id: str,
        folder_path: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any] | None:
        """Upload large file using upload session.

        Args:
            site_id: SharePoint site ID.
            folder_path: Folder path.
            filename: File name.
            content: File content.

        Returns:
            Uploaded file metadata.
        """
        try:
            import requests

            file_path = f"{folder_path}/{filename}".replace(" ", "%20")
            encoded_path = file_path.replace("/", ":/")

            # Create upload session
            session_url = (
                f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                f"/drive/root:/{encoded_path}:/createUploadSession"
            )

            response = requests.post(
                session_url,
                headers=self._get_headers(),
                json={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
                timeout=30,
            )
            response.raise_for_status()

            upload_url = response.json()["uploadUrl"]

            # Upload in chunks
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            total_size = len(content)

            for start in range(0, total_size, chunk_size):
                end = min(start + chunk_size, total_size)
                chunk = content[start:end]

                headers = {
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {start}-{end - 1}/{total_size}",
                }

                response = requests.put(
                    upload_url,
                    headers=headers,
                    data=chunk,
                    timeout=120,
                )
                response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Failed large file upload: {e}")
            return None

    def create_folder(self, folder_path: str) -> dict[str, Any] | None:
        """Create a folder in SharePoint.

        Args:
            folder_path: Relative folder path to create.

        Returns:
            Created folder metadata, or None if failed.
        """
        if self._demo_mode:
            return {"name": folder_path.split("/")[-1], "type": "folder"}

        try:
            import requests

            site_id = self._get_site_id()

            # Split path into parent and new folder name
            parts = folder_path.rsplit("/", 1)
            if len(parts) == 2:
                parent_path, folder_name = parts
                encoded_parent = parent_path.replace("/", ":/").replace(" ", "%20")
                graph_url = (
                    f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                    f"/drive/root:/{encoded_parent}:/children"
                )
            else:
                folder_name = parts[0]
                graph_url = (
                    f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                    f"/drive/root/children"
                )

            response = requests.post(
                graph_url,
                headers=self._get_headers(),
                json={
                    "name": folder_name,
                    "folder": {},
                    "@microsoft.graph.conflictBehavior": "rename",
                },
                timeout=30,
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Failed to create folder {folder_path}: {e}")
            return None

    def search_files(
        self,
        query: str,
        folder_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for files in SharePoint.

        Args:
            query: Search query string.
            folder_path: Optional folder to limit search scope.

        Returns:
            List of matching file metadata.
        """
        if self._demo_mode:
            return self._demo_search_files(query, folder_path)

        try:
            import requests

            site_id = self._get_site_id()

            graph_url = (
                f"https://graph.microsoft.com/v1.0/sites/{site_id}"
                f"/drive/root/search(q='{query}')"
            )

            response = requests.get(
                graph_url,
                headers=self._get_headers(),
                timeout=30,
            )
            response.raise_for_status()

            items = response.json().get("value", [])

            # Filter by folder if specified
            if folder_path:
                items = [
                    item for item in items
                    if folder_path.lower() in item.get("parentReference", {}).get("path", "").lower()
                ]

            return [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "size": item.get("size", 0),
                    "modified": item.get("lastModifiedDateTime"),
                    "web_url": item.get("webUrl"),
                    "path": item.get("parentReference", {}).get("path", ""),
                }
                for item in items
            ]

        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []

    # =========================================================================
    # Demo Mode Methods
    # =========================================================================

    def _get_demo_folder_contents(self, folder_path: str) -> list[dict[str, Any]]:
        """Get demo folder contents for testing."""
        demo_data = {
            "Recruitment/JobDescriptions": [
                {
                    "id": "jd_001",
                    "name": "Senior_Python_Developer_JD.pdf",
                    "size": 125000,
                    "type": "file",
                    "created": "2025-01-15T10:00:00Z",
                    "modified": "2025-01-15T10:00:00Z",
                },
                {
                    "id": "jd_002",
                    "name": "DevOps_Engineer_JD.docx",
                    "size": 98000,
                    "type": "file",
                    "created": "2025-01-10T14:30:00Z",
                    "modified": "2025-01-12T09:00:00Z",
                },
            ],
            "Recruitment/Resumes": [
                {
                    "id": "resume_001",
                    "name": "John_Doe_Resume.pdf",
                    "size": 250000,
                    "type": "file",
                    "created": "2025-01-18T08:00:00Z",
                    "modified": "2025-01-18T08:00:00Z",
                },
                {
                    "id": "resume_002",
                    "name": "Jane_Smith_Resume.docx",
                    "size": 180000,
                    "type": "file",
                    "created": "2025-01-17T11:00:00Z",
                    "modified": "2025-01-17T11:00:00Z",
                },
                {
                    "id": "resume_003",
                    "name": "Bob_Johnson_Resume.pdf",
                    "size": 220000,
                    "type": "file",
                    "created": "2025-01-19T09:30:00Z",
                    "modified": "2025-01-19T09:30:00Z",
                },
            ],
            "Recruitment/InterviewQuestions": [],
            "Recruitment/Scoring": [],
        }
        return demo_data.get(folder_path, [])

    def _get_demo_file_content(self, folder_path: str, filename: str) -> bytes:
        """Get demo file content for testing."""
        # Return placeholder content
        demo_content = f"""
Demo file content for: {filename}
Folder: {folder_path}
Generated: {datetime.now().isoformat()}

This is simulated content for demonstration purposes.
In production, actual SharePoint file content would be returned.
"""
        return demo_content.encode("utf-8")

    def _demo_upload_file(
        self,
        folder_path: str,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """Demo file upload."""
        return {
            "id": f"demo_{datetime.now().timestamp()}",
            "name": filename,
            "size": len(content),
            "folder_path": folder_path,
            "created": datetime.now().isoformat(),
            "demo_mode": True,
        }

    def _demo_search_files(
        self,
        query: str,
        folder_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Demo file search."""
        all_files = []
        for folder, files in [
            ("Recruitment/Resumes", self._get_demo_folder_contents("Recruitment/Resumes")),
            ("Recruitment/JobDescriptions", self._get_demo_folder_contents("Recruitment/JobDescriptions")),
        ]:
            if not folder_path or folder_path in folder:
                for f in files:
                    if query.lower() in f["name"].lower():
                        all_files.append({**f, "path": folder})
        return all_files


# =============================================================================
# SharePoint Client Factory
# =============================================================================

def get_sharepoint_client(session_id: str = "default") -> SharePointClient:
    """Get or create SharePoint client for session.

    Args:
        session_id: Session identifier for client isolation.

    Returns:
        SharePointClient instance.
    """
    if session_id not in _sharepoint_clients:
        _sharepoint_clients[session_id] = SharePointClient()
    return _sharepoint_clients[session_id]


# =============================================================================
# Tool Functions
# =============================================================================

@tool
def list_sharepoint_folder(
    folder_type: str,
    session_id: str = "default",
) -> str:
    """List files in a recruitment SharePoint folder.

    Use this tool to see what documents are available in SharePoint.

    Args:
        folder_type: Type of folder - "jd" (job descriptions), "resumes",
                     "roles", "questions", "answers", "scoring", "shortlist".
        session_id: Session identifier.

    Returns:
        Formatted list of files in the folder.
    """
    config = get_recruitment_config()

    folder_map = {
        "jd": config.sharepoint.jd_folder,
        "resumes": config.sharepoint.resumes_folder,
        "roles": config.sharepoint.roles_folder,
        "questions": config.sharepoint.interview_questions_folder,
        "answers": config.sharepoint.candidate_answers_folder,
        "scoring": config.sharepoint.scoring_folder,
        "shortlist": config.sharepoint.shortlist_folder,
    }

    folder_path = folder_map.get(folder_type.lower())
    if not folder_path:
        return f"Unknown folder type: {folder_type}. Valid types: {', '.join(folder_map.keys())}"

    client = get_sharepoint_client(session_id)
    files = client.list_folder(folder_path)

    if not files:
        return f"No files found in {folder_path}"

    # Format output
    output = f"## Files in {folder_path}\n\n"
    output += f"**Total: {len(files)} items**\n\n"

    for f in files:
        icon = "📁" if f["type"] == "folder" else "📄"
        size_kb = f.get("size", 0) / 1024
        output += f"- {icon} **{f['name']}** ({size_kb:.1f} KB)\n"
        if f.get("modified"):
            output += f"  Modified: {f['modified']}\n"

    return output


@tool
def download_sharepoint_document(
    folder_type: str,
    filename: str,
    session_id: str = "default",
) -> str:
    """Download a document from SharePoint.

    Use this tool to retrieve document content for processing.

    Args:
        folder_type: Type of folder - "jd", "resumes", "roles", "questions",
                     "answers", "scoring", "shortlist".
        filename: Name of the file to download.
        session_id: Session identifier.

    Returns:
        Status message indicating success or failure.
    """
    config = get_recruitment_config()

    folder_map = {
        "jd": config.sharepoint.jd_folder,
        "resumes": config.sharepoint.resumes_folder,
        "roles": config.sharepoint.roles_folder,
        "questions": config.sharepoint.interview_questions_folder,
        "answers": config.sharepoint.candidate_answers_folder,
        "scoring": config.sharepoint.scoring_folder,
        "shortlist": config.sharepoint.shortlist_folder,
    }

    folder_path = folder_map.get(folder_type.lower())
    if not folder_path:
        return f"Unknown folder type: {folder_type}"

    client = get_sharepoint_client(session_id)
    content = client.download_file(folder_path, filename)

    if content is None:
        return f"Failed to download {filename} from {folder_path}"

    # Cache the downloaded content for further processing
    cache_key = f"{folder_path}/{filename}"
    if session_id not in _document_cache:
        _document_cache[session_id] = {}
    _document_cache[session_id][cache_key] = {
        "content": content,
        "filename": filename,
        "folder_path": folder_path,
        "downloaded_at": datetime.now().isoformat(),
    }

    return (
        f"Successfully downloaded {filename} ({len(content):,} bytes) from {folder_path}. "
        f"Document is now available for processing."
    )


@tool
def upload_to_sharepoint(
    folder_type: str,
    filename: str,
    content: str,
    session_id: str = "default",
) -> str:
    """Upload a document to SharePoint.

    Use this tool to save generated documents like interview questions
    or scoring reports to SharePoint.

    Args:
        folder_type: Type of folder - "questions", "scoring", "shortlist".
        filename: Name for the uploaded file.
        content: Content to upload (text content will be encoded as UTF-8).
        session_id: Session identifier.

    Returns:
        Status message indicating success or failure.
    """
    config = get_recruitment_config()

    # Only allow uploads to specific folders
    allowed_folders = {
        "questions": config.sharepoint.interview_questions_folder,
        "scoring": config.sharepoint.scoring_folder,
        "shortlist": config.sharepoint.shortlist_folder,
    }

    folder_path = allowed_folders.get(folder_type.lower())
    if not folder_path:
        return f"Upload not allowed to folder type: {folder_type}. Allowed: {', '.join(allowed_folders.keys())}"

    client = get_sharepoint_client(session_id)

    # Encode content as bytes
    content_bytes = content.encode("utf-8")

    result = client.upload_file(folder_path, filename, content_bytes)

    if result:
        return f"Successfully uploaded {filename} to {folder_path}"
    else:
        return f"Failed to upload {filename} to {folder_path}"


@tool
def search_sharepoint_documents(
    query: str,
    folder_type: str | None = None,
    session_id: str = "default",
) -> str:
    """Search for documents in SharePoint.

    Use this tool to find specific documents by name or content.

    Args:
        query: Search query string.
        folder_type: Optional folder type to limit search scope.
        session_id: Session identifier.

    Returns:
        Formatted search results.
    """
    config = get_recruitment_config()

    folder_path = None
    if folder_type:
        folder_map = {
            "jd": config.sharepoint.jd_folder,
            "resumes": config.sharepoint.resumes_folder,
            "roles": config.sharepoint.roles_folder,
            "questions": config.sharepoint.interview_questions_folder,
            "answers": config.sharepoint.candidate_answers_folder,
            "scoring": config.sharepoint.scoring_folder,
            "shortlist": config.sharepoint.shortlist_folder,
        }
        folder_path = folder_map.get(folder_type.lower())

    client = get_sharepoint_client(session_id)
    results = client.search_files(query, folder_path)

    if not results:
        scope = f" in {folder_type}" if folder_type else ""
        return f"No documents found matching '{query}'{scope}"

    output = f"## Search Results for '{query}'\n\n"
    output += f"**Found: {len(results)} documents**\n\n"

    for r in results:
        output += f"- **{r['name']}**\n"
        output += f"  Path: {r.get('path', 'N/A')}\n"
        if r.get("modified"):
            output += f"  Modified: {r['modified']}\n"
        output += "\n"

    return output


@tool
def get_cached_document(
    folder_type: str,
    filename: str,
    session_id: str = "default",
) -> str:
    """Get a previously downloaded document from cache.

    Use this tool to access content of documents that were already downloaded.

    Args:
        folder_type: Type of folder the document was downloaded from.
        filename: Name of the file.
        session_id: Session identifier.

    Returns:
        Document content or error message.
    """
    config = get_recruitment_config()

    folder_map = {
        "jd": config.sharepoint.jd_folder,
        "resumes": config.sharepoint.resumes_folder,
        "roles": config.sharepoint.roles_folder,
        "questions": config.sharepoint.interview_questions_folder,
        "answers": config.sharepoint.candidate_answers_folder,
        "scoring": config.sharepoint.scoring_folder,
        "shortlist": config.sharepoint.shortlist_folder,
    }

    folder_path = folder_map.get(folder_type.lower())
    if not folder_path:
        return f"Unknown folder type: {folder_type}"

    cache_key = f"{folder_path}/{filename}"

    if session_id not in _document_cache:
        return f"No cached documents for session {session_id}"

    if cache_key not in _document_cache[session_id]:
        return f"Document not in cache: {filename}. Use download_sharepoint_document first."

    cached = _document_cache[session_id][cache_key]
    content = cached["content"]

    # Try to decode as text
    try:
        text_content = content.decode("utf-8")
        return f"## Document: {filename}\n\n{text_content}"
    except UnicodeDecodeError:
        return f"Document {filename} is binary ({len(content):,} bytes). Use document processing tools to extract content."


@tool
def create_sharepoint_folder(
    folder_path: str,
    session_id: str = "default",
) -> str:
    """Create a new folder in SharePoint.

    Use this tool to create folders for organizing recruitment documents.

    Args:
        folder_path: Full path of the folder to create.
        session_id: Session identifier.

    Returns:
        Status message indicating success or failure.
    """
    client = get_sharepoint_client(session_id)
    result = client.create_folder(folder_path)

    if result:
        return f"Successfully created folder: {folder_path}"
    else:
        return f"Failed to create folder: {folder_path}"


__all__ = [
    "SharePointClient",
    "get_sharepoint_client",
    "list_sharepoint_folder",
    "download_sharepoint_document",
    "upload_to_sharepoint",
    "search_sharepoint_documents",
    "get_cached_document",
    "create_sharepoint_folder",
]
